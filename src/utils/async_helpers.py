"""
Async Utility Functions

Async utility functions for rate limiting, retry mechanisms,
and other async operations used throughout the benchmarking system.
"""

import asyncio
import time
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
import random

from core.exceptions import ModelAPIError, RateLimitError
from utils.logging import get_logger

logger = get_logger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, 
                      max_delay: float = 60.0, backoff_factor: float = 2.0,
                      jitter: bool = True):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (RateLimitError, ModelAPIError, asyncio.TimeoutError) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    # Handle rate limit specific delays
                    if isinstance(e, RateLimitError) and hasattr(e, 'retry_after'):
                        delay = max(delay, e.retry_after)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                except Exception as e:
                    # Non-retryable exceptions
                    logger.error(f"Function {func.__name__} failed with non-retryable error: {str(e)}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class AsyncThrottler:
    """Async throttler for rate limiting requests."""
    
    def __init__(self, rate_limit: int, time_window: float = 60.0):
        """
        Initialize throttler.
        
        Args:
            rate_limit: Maximum number of requests per time window
            time_window: Time window in seconds (default: 60s)
        """
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            
            # Remove old requests outside the time window
            cutoff_time = current_time - self.time_window
            self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
            
            # Check if we can make a request
            if len(self.requests) >= self.rate_limit:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                sleep_time = self.time_window - (current_time - oldest_request)
                
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Recursively try again
                    return await self.acquire()
            
            # Record this request
            self.requests.append(current_time)


def throttle_requests(rate_limit: int, time_window: float = 60.0):
    """
    Decorator to throttle async function calls.
    
    Args:
        rate_limit: Maximum number of requests per time window
        time_window: Time window in seconds
    """
    throttler = AsyncThrottler(rate_limit, time_window)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            await throttler.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncBatch:
    """Helper for processing items in batches asynchronously."""
    
    def __init__(self, batch_size: int = 10, max_concurrent: int = 5):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
    
    async def process(self, items: List[Any], process_func: Callable, 
                     **kwargs) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Async function to process each item
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results
        """
        logger.info(f"Processing {len(items)} items in batches of {self.batch_size}")
        
        # Create batches
        batches = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batches.append(batch)
        
        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch: List[Any]) -> List[Any]:
            async with semaphore:
                tasks = [process_func(item, **kwargs) for item in batch]
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process all batches
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results


class AsyncQueue:
    """Thread-safe async queue for producer-consumer patterns."""
    
    def __init__(self, maxsize: int = 0):
        """Initialize queue with optional size limit."""
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.active_workers = 0
        self._lock = asyncio.Lock()
    
    async def put(self, item: Any) -> None:
        """Put item in queue."""
        await self.queue.put(item)
    
    async def get(self) -> Any:
        """Get item from queue."""
        return await self.queue.get()
    
    async def worker(self, worker_func: Callable, worker_id: int) -> None:
        """
        Worker coroutine that processes items from the queue.
        
        Args:
            worker_func: Async function to process queue items
            worker_id: Unique worker identifier
        """
        async with self._lock:
            self.active_workers += 1
        
        logger.debug(f"Worker {worker_id} started")
        
        try:
            while True:
                try:
                    # Get item with timeout to check for shutdown
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    
                    # Process item
                    try:
                        await worker_func(item)
                    except Exception as e:
                        logger.error(f"Worker {worker_id} failed to process item: {str(e)}")
                    finally:
                        self.queue.task_done()
                        
                except asyncio.TimeoutError:
                    # Check if queue is empty and no more items are expected
                    if self.queue.empty():
                        break
                
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
        finally:
            async with self._lock:
                self.active_workers -= 1
            logger.debug(f"Worker {worker_id} stopped")
    
    async def wait_completion(self) -> None:
        """Wait for all queued items to be processed."""
        await self.queue.join()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self.queue.qsize()


async def gather_with_concurrency(tasks: List[Callable], max_concurrent: int = 5) -> List[Any]:
    """
    Execute tasks with limited concurrency.
    
    Args:
        tasks: List of async functions to execute
        max_concurrent: Maximum concurrent executions
        
    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_task(task: Callable) -> Any:
        async with semaphore:
            if asyncio.iscoroutinefunction(task):
                return await task()
            else:
                return await task
    
    return await asyncio.gather(*[execute_task(task) for task in tasks])


async def timeout_after(coro: Callable, timeout: float, 
                       timeout_message: str = "Operation timed out") -> Any:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        timeout_message: Message to include in timeout exception
        
    Returns:
        Result of coroutine
        
    Raises:
        asyncio.TimeoutError: If operation times out
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"Timeout after {timeout}s: {timeout_message}")
        raise asyncio.TimeoutError(timeout_message)


class AsyncContextManager:
    """Base class for async context managers."""
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


def create_task_with_name(coro: Callable, name: str) -> asyncio.Task:
    """
    Create a named task for better debugging.
    
    Args:
        coro: Coroutine to execute
        name: Task name
        
    Returns:
        Named asyncio Task
    """
    task = asyncio.create_task(coro)
    task.set_name(name)
    return task


async def cancel_tasks(tasks: List[asyncio.Task]) -> None:
    """
    Cancel a list of tasks gracefully.
    
    Args:
        tasks: List of tasks to cancel
    """
    if not tasks:
        return
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    
    # Wait for them to finish cancellation
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.debug(f"Cancelled {len(tasks)} tasks")