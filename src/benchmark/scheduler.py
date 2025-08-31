"""
Benchmark Scheduler

Handles batch operations and concurrent benchmarking across multiple models
with queue management, resource control, and progress tracking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from .runner import BenchmarkRunner, BenchmarkConfig, BenchmarkResult, RunMode
from src.core.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerState(str, Enum):
    """States of the benchmark scheduler."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ScheduledBenchmark:
    """A scheduled benchmark task."""
    model_name: str
    config: BenchmarkConfig
    benchmark_name: Optional[str] = None
    priority: int = 1  # Lower numbers = higher priority
    scheduled_time: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Model names that must complete first
    retry_count: int = 0
    max_retries: int = 2
    timeout_minutes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerProgress:
    """Overall progress across all scheduled benchmarks."""
    total_benchmarks: int
    completed_benchmarks: int
    failed_benchmarks: int
    running_benchmarks: int
    queued_benchmarks: int
    current_models: List[str]
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    
    @property
    def completion_percentage(self) -> float:
        if self.total_benchmarks == 0:
            return 0.0
        return ((self.completed_benchmarks + self.failed_benchmarks) / self.total_benchmarks) * 100.0
    
    @property
    def success_rate(self) -> float:
        completed_total = self.completed_benchmarks + self.failed_benchmarks
        if completed_total == 0:
            return 0.0
        return (self.completed_benchmarks / completed_total) * 100.0


@dataclass
class ResourceLimits:
    """Resource limits for scheduler operation."""
    max_concurrent_models: int = 3
    max_concurrent_requests_per_model: int = 5
    max_total_requests_per_minute: int = 100
    max_memory_usage_mb: int = 2048
    max_disk_usage_mb: int = 5120


class BenchmarkScheduler:
    """Schedules and manages multiple benchmark runs."""
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """
        Initialize the benchmark scheduler.
        
        Args:
            resource_limits: Resource usage limits
        """
        self.config = get_config()
        self.resource_limits = resource_limits or ResourceLimits()
        
        # Scheduler state
        self.state = SchedulerState.IDLE
        self.lock = threading.RLock()
        
        # Task management
        self.pending_queue: List[ScheduledBenchmark] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}  # model_name -> task
        self.completed_results: Dict[str, BenchmarkResult] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        
        # Progress tracking
        self.progress: Optional[SchedulerProgress] = None
        self.start_time: Optional[datetime] = None
        
        # Resource tracking
        self.request_timestamps: List[float] = []
        self.memory_usage_mb = 0
        self.disk_usage_mb = 0
        
        # Event callbacks
        self.on_benchmark_started: Optional[Callable[[str], None]] = None
        self.on_benchmark_completed: Optional[Callable[[str, BenchmarkResult], None]] = None
        self.on_benchmark_failed: Optional[Callable[[str, Exception], None]] = None
        self.on_all_completed: Optional[Callable[[Dict[str, BenchmarkResult]], None]] = None
        
        # Internal components
        self._runner_pool: Dict[str, BenchmarkRunner] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
    
    def schedule_benchmark(self, 
                          model_name: str,
                          config: BenchmarkConfig,
                          benchmark_name: Optional[str] = None,
                          priority: int = 1,
                          dependencies: Optional[List[str]] = None,
                          timeout_minutes: Optional[int] = None) -> str:
        """
        Schedule a benchmark for execution.
        
        Args:
            model_name: Name of the model to benchmark
            config: Benchmark configuration
            benchmark_name: Optional custom name for the benchmark
            priority: Priority (lower = higher priority)
            dependencies: Models that must complete before this one
            timeout_minutes: Maximum execution time in minutes
            
        Returns:
            Unique identifier for the scheduled benchmark
        """
        with self.lock:
            if self.state == SchedulerState.STOPPED:
                raise RuntimeError("Scheduler is stopped, cannot schedule new benchmarks")
            
            # Create scheduled benchmark
            scheduled = ScheduledBenchmark(
                model_name=model_name,
                config=config,
                benchmark_name=benchmark_name,
                priority=priority,
                dependencies=dependencies or [],
                timeout_minutes=timeout_minutes
            )
            
            # Add to queue (will be sorted by priority)
            self.pending_queue.append(scheduled)
            self.pending_queue.sort(key=lambda x: x.priority)
            
            logger.info(f"Scheduled benchmark for {model_name} (priority: {priority})")
            
            return f"{model_name}_{int(time.time())}"
    
    def schedule_multiple(self, 
                         models: List[str],
                         mode: RunMode = RunMode.STANDARD,
                         benchmark_name_prefix: Optional[str] = None,
                         concurrent_limit: Optional[int] = None) -> List[str]:
        """
        Schedule benchmarks for multiple models.
        
        Args:
            models: List of model names
            mode: Benchmark mode to use for all models
            benchmark_name_prefix: Prefix for benchmark names
            concurrent_limit: Override default concurrency limit
            
        Returns:
            List of scheduled benchmark identifiers
        """
        if concurrent_limit:
            self.resource_limits.max_concurrent_models = concurrent_limit
        
        scheduled_ids = []
        runner = BenchmarkRunner()
        
        for i, model_name in enumerate(models):
            config = runner.get_default_config(mode)
            
            benchmark_name = None
            if benchmark_name_prefix:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                benchmark_name = f"{benchmark_name_prefix}_{model_name}_{timestamp}"
            
            # Stagger priorities to ensure some ordering if needed
            priority = i + 1
            
            scheduled_id = self.schedule_benchmark(
                model_name=model_name,
                config=config,
                benchmark_name=benchmark_name,
                priority=priority
            )
            scheduled_ids.append(scheduled_id)
        
        logger.info(f"Scheduled {len(models)} benchmarks with mode {mode.value}")
        return scheduled_ids
    
    async def run_scheduled_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """
        Execute all scheduled benchmarks.
        
        Returns:
            Dictionary of model_name -> BenchmarkResult
        """
        with self.lock:
            if self.state != SchedulerState.IDLE:
                raise RuntimeError(f"Scheduler is not idle (current state: {self.state})")
            
            if not self.pending_queue:
                logger.warning("No benchmarks scheduled")
                return {}
            
            self.state = SchedulerState.RUNNING
            self.start_time = datetime.now()
            
            # Initialize progress tracking
            self.progress = SchedulerProgress(
                total_benchmarks=len(self.pending_queue),
                completed_benchmarks=0,
                failed_benchmarks=0,
                running_benchmarks=0,
                queued_benchmarks=len(self.pending_queue),
                current_models=[],
                start_time=self.start_time
            )
        
        logger.info(f"Starting scheduled benchmark execution: {self.progress.total_benchmarks} benchmarks")
        
        try:
            # Start background monitoring
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            
            # Main execution loop
            while True:
                with self.lock:
                    if self.state == SchedulerState.STOPPING:
                        logger.info("Scheduler stopping requested")
                        break
                    
                    if self.state == SchedulerState.PAUSED:
                        await asyncio.sleep(1.0)
                        continue
                    
                    # Check if we're done
                    if not self.pending_queue and not self.running_tasks:
                        logger.info("All benchmarks completed")
                        break
                
                # Try to start new benchmarks
                await self._process_queue()
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Wait a bit before next iteration
                await asyncio.sleep(0.5)
            
            # Wait for any remaining tasks to complete
            if self.running_tasks:
                logger.info(f"Waiting for {len(self.running_tasks)} running tasks to complete...")
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
            
            # Final cleanup
            await self._cleanup_completed_tasks()
            
        except Exception as e:
            logger.error(f"Scheduler execution failed: {str(e)}")
            raise
        finally:
            # Clean up monitoring
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            with self.lock:
                self.state = SchedulerState.IDLE
                self._update_progress()
                
                # Trigger completion callback
                if self.on_all_completed:
                    self.on_all_completed(self.completed_results.copy())
        
        logger.info(f"Scheduler execution completed: {len(self.completed_results)} successful, "
                   f"{len(self.failed_tasks)} failed")
        
        return self.completed_results.copy()
    
    async def _process_queue(self):
        """Process the pending queue and start new benchmarks."""
        with self.lock:
            if not self.pending_queue:
                return
            
            # Check resource limits
            if len(self.running_tasks) >= self.resource_limits.max_concurrent_models:
                return
            
            # Check rate limits
            if not self._check_rate_limits():
                return
            
            # Find next benchmark to start
            ready_benchmarks = []
            for benchmark in self.pending_queue:
                if self._can_start_benchmark(benchmark):
                    ready_benchmarks.append(benchmark)
            
            if not ready_benchmarks:
                return
            
            # Start the highest priority benchmark
            benchmark_to_start = ready_benchmarks[0]
            self.pending_queue.remove(benchmark_to_start)
        
        # Start the benchmark (outside of lock to avoid blocking)
        await self._start_benchmark(benchmark_to_start)
    
    def _can_start_benchmark(self, benchmark: ScheduledBenchmark) -> bool:
        """Check if a benchmark can be started based on dependencies."""
        # Check if model is already running
        if benchmark.model_name in self.running_tasks:
            return False
        
        # Check dependencies
        for dep_model in benchmark.dependencies:
            if (dep_model not in self.completed_results and 
                dep_model not in self.failed_tasks):
                return False
        
        # Check scheduled time
        if benchmark.scheduled_time and datetime.now() < benchmark.scheduled_time:
            return False
        
        return True
    
    async def _start_benchmark(self, scheduled: ScheduledBenchmark):
        """Start a single benchmark."""
        logger.info(f"Starting benchmark for {scheduled.model_name}")
        
        # Get or create runner
        runner = self._get_runner(scheduled.model_name)
        
        # Create task with timeout
        async def run_with_timeout():
            if scheduled.timeout_minutes:
                return await asyncio.wait_for(
                    runner.run_benchmark(
                        model_name=scheduled.model_name,
                        mode=scheduled.config.mode,
                        custom_config=scheduled.config,
                        benchmark_name=scheduled.benchmark_name
                    ),
                    timeout=scheduled.timeout_minutes * 60
                )
            else:
                return await runner.run_benchmark(
                    model_name=scheduled.model_name,
                    mode=scheduled.config.mode,
                    custom_config=scheduled.config,
                    benchmark_name=scheduled.benchmark_name
                )
        
        # Store task
        task = asyncio.create_task(run_with_timeout())
        
        with self.lock:
            self.running_tasks[scheduled.model_name] = task
            self.progress.running_benchmarks = len(self.running_tasks)
            self.progress.queued_benchmarks = len(self.pending_queue)
            self.progress.current_models = list(self.running_tasks.keys())
        
        # Trigger callback
        if self.on_benchmark_started:
            self.on_benchmark_started(scheduled.model_name)
        
        logger.info(f"Benchmark task started for {scheduled.model_name}")
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed benchmark tasks."""
        completed_models = []
        
        with self.lock:
            for model_name, task in list(self.running_tasks.items()):
                if task.done():
                    completed_models.append(model_name)
        
        # Process completed tasks
        for model_name in completed_models:
            task = self.running_tasks.pop(model_name)
            
            try:
                result = await task
                
                with self.lock:
                    self.completed_results[model_name] = result
                    self.progress.completed_benchmarks += 1
                    self._update_progress()
                
                logger.info(f"Benchmark completed successfully for {model_name}")
                
                # Trigger callback
                if self.on_benchmark_completed:
                    self.on_benchmark_completed(model_name, result)
                    
            except Exception as e:
                with self.lock:
                    self.failed_tasks[model_name] = e
                    self.progress.failed_benchmarks += 1
                    self._update_progress()
                
                logger.error(f"Benchmark failed for {model_name}: {str(e)}")
                
                # Trigger callback
                if self.on_benchmark_failed:
                    self.on_benchmark_failed(model_name, e)
                
                # Check for retry
                await self._handle_retry(model_name, e)
    
    async def _handle_retry(self, model_name: str, error: Exception):
        """Handle retry logic for failed benchmarks."""
        # Find the original scheduled benchmark (this is simplified)
        # In a full implementation, you'd store more metadata to handle retries properly
        logger.info(f"Retry logic not fully implemented for {model_name}")
    
    def _get_runner(self, model_name: str) -> BenchmarkRunner:
        """Get or create a benchmark runner for a model."""
        if model_name not in self._runner_pool:
            self._runner_pool[model_name] = BenchmarkRunner()
        return self._runner_pool[model_name]
    
    def _check_rate_limits(self) -> bool:
        """Check if we can make more requests without hitting rate limits."""
        current_time = time.time()
        
        # Clean old timestamps (keep only last minute)
        cutoff_time = current_time - 60
        self.request_timestamps = [t for t in self.request_timestamps if t > cutoff_time]
        
        # Check if we're at the limit
        if len(self.request_timestamps) >= self.resource_limits.max_total_requests_per_minute:
            return False
        
        # Record this potential request
        self.request_timestamps.append(current_time)
        return True
    
    def _update_progress(self):
        """Update progress calculations."""
        if not self.progress:
            return
        
        self.progress.running_benchmarks = len(self.running_tasks)
        self.progress.queued_benchmarks = len(self.pending_queue)
        self.progress.current_models = list(self.running_tasks.keys())
        
        # Estimate completion time
        if self.progress.completed_benchmarks > 0 and self.start_time:
            elapsed = datetime.now() - self.start_time
            avg_time_per_benchmark = elapsed / self.progress.completed_benchmarks
            remaining_benchmarks = self.progress.total_benchmarks - self.progress.completed_benchmarks - self.progress.failed_benchmarks
            estimated_remaining_time = avg_time_per_benchmark * remaining_benchmarks
            self.progress.estimated_completion = datetime.now() + estimated_remaining_time
    
    async def _monitor_resources(self):
        """Monitor resource usage and enforce limits."""
        while self.state == SchedulerState.RUNNING:
            try:
                # Monitor memory usage (simplified)
                # In a real implementation, you'd use psutil or similar
                
                # Monitor disk usage
                # Check available disk space
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Back off on error
    
    def pause(self):
        """Pause the scheduler (current tasks continue, no new ones start)."""
        with self.lock:
            if self.state == SchedulerState.RUNNING:
                self.state = SchedulerState.PAUSED
                logger.info("Scheduler paused")
    
    def resume(self):
        """Resume the scheduler."""
        with self.lock:
            if self.state == SchedulerState.PAUSED:
                self.state = SchedulerState.RUNNING
                logger.info("Scheduler resumed")
    
    def stop(self):
        """Stop the scheduler (cancels running tasks)."""
        with self.lock:
            if self.state in [SchedulerState.RUNNING, SchedulerState.PAUSED]:
                self.state = SchedulerState.STOPPING
                logger.info("Scheduler stopping...")
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
    
    def get_progress(self) -> Optional[SchedulerProgress]:
        """Get current scheduler progress."""
        return self.progress
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status."""
        with self.lock:
            return {
                'state': self.state.value,
                'pending_count': len(self.pending_queue),
                'running_count': len(self.running_tasks),
                'completed_count': len(self.completed_results),
                'failed_count': len(self.failed_tasks),
                'pending_models': [b.model_name for b in self.pending_queue],
                'running_models': list(self.running_tasks.keys()),
                'completed_models': list(self.completed_results.keys()),
                'failed_models': list(self.failed_tasks.keys())
            }
    
    def clear_queue(self):
        """Clear all pending benchmarks."""
        with self.lock:
            if self.state == SchedulerState.RUNNING:
                raise RuntimeError("Cannot clear queue while scheduler is running")
            
            self.pending_queue.clear()
            logger.info("Benchmark queue cleared")
    
    def remove_benchmark(self, model_name: str) -> bool:
        """Remove a benchmark from the pending queue."""
        with self.lock:
            original_length = len(self.pending_queue)
            self.pending_queue = [b for b in self.pending_queue if b.model_name != model_name]
            removed = len(self.pending_queue) < original_length
            
            if removed:
                logger.info(f"Removed benchmark for {model_name} from queue")
            
            return removed