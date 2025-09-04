"""
Debug Logger for Model Interactions

Specialized logging for debugging model prompts, responses, and evaluation.
Creates detailed logs to help diagnose issues with model performance.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DebugLogEntry:
    """Structure for debug log entries."""
    timestamp: str
    benchmark_id: Optional[int]
    question_id: Any
    model_name: str
    category: Optional[str]
    value: Optional[Any]
    question_text: str
    correct_answer: str
    formatted_prompt: str
    raw_response: str
    parsed_answer: str
    is_correct: bool
    match_score: float
    match_type: str
    confidence_score: float
    response_time_ms: float
    cost_usd: float
    tokens_input: int
    tokens_output: int
    grading_details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DebugLogger:
    """
    Specialized logger for debugging model interactions.
    
    Creates detailed JSON logs of prompts, responses, and evaluations
    to help diagnose model performance issues.
    """
    
    def __init__(self, debug_enabled: bool = False, log_dir: str = "logs/debug"):
        """
        Initialize debug logger.
        
        Args:
            debug_enabled: Whether debug logging is enabled
            log_dir: Directory for debug log files
        """
        self.debug_enabled = debug_enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create separate loggers for different types of debug info
        self._setup_loggers()
    
    def _setup_loggers(self) -> None:
        """Set up specialized debug loggers."""
        if not self.debug_enabled:
            return
        
        # Main debug logger - detailed JSON logs
        self.debug_logger = logging.getLogger('debug.model_interactions')
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.debug_logger.handlers[:]:
            self.debug_logger.removeHandler(handler)
        
        # Create file handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = self.log_dir / f"model_interactions_{timestamp}.jsonl"
        
        file_handler = logging.FileHandler(debug_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Simple formatter for JSON lines
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.debug_logger.addHandler(file_handler)
        self.debug_logger.propagate = False  # Don't propagate to root logger
        
        # Summary logger - readable summary
        self.summary_logger = logging.getLogger('debug.summary')
        self.summary_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.summary_logger.handlers[:]:
            self.summary_logger.removeHandler(handler)
        
        summary_file = self.log_dir / f"debug_summary_{timestamp}.log"
        summary_handler = logging.FileHandler(summary_file)
        summary_handler.setLevel(logging.DEBUG)
        
        summary_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        summary_handler.setFormatter(summary_formatter)
        
        self.summary_logger.addHandler(summary_handler)
        self.summary_logger.propagate = False
        
        logger.info(f"Debug logging enabled - files: {debug_file}, {summary_file}")
    
    def log_model_interaction(
        self,
        benchmark_id: Optional[int],
        question_id: Any,
        model_name: str,
        category: Optional[str],
        value: Optional[Any],
        question_text: str,
        correct_answer: str,
        formatted_prompt: str,
        raw_response: str,
        parsed_answer: str,
        is_correct: bool,
        match_score: float,
        match_type: str,
        confidence_score: float,
        response_time_ms: float,
        cost_usd: float,
        tokens_input: int,
        tokens_output: int,
        grading_details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a complete model interaction for debugging.
        
        Args:
            benchmark_id: ID of the benchmark run
            question_id: Question identifier
            model_name: Name of the model
            category: Question category
            value: Question value
            question_text: The original question text
            correct_answer: The correct answer
            formatted_prompt: The formatted prompt sent to model
            raw_response: Raw response from model
            parsed_answer: Parsed answer extracted from response
            is_correct: Whether the answer was graded as correct
            match_score: Fuzzy matching score
            match_type: Type of match found
            confidence_score: Model confidence score
            response_time_ms: Response time in milliseconds
            cost_usd: Cost of the API call
            tokens_input: Input token count
            tokens_output: Output token count
            grading_details: Additional grading information
            error: Error message if any
        """
        if not self.debug_enabled:
            return
        
        try:
            # Create structured log entry
            entry = DebugLogEntry(
                timestamp=datetime.now().isoformat(),
                benchmark_id=benchmark_id,
                question_id=question_id,
                model_name=model_name,
                category=category,
                value=value,
                question_text=question_text,
                correct_answer=correct_answer,
                formatted_prompt=formatted_prompt,
                raw_response=raw_response,
                parsed_answer=parsed_answer,
                is_correct=is_correct,
                match_score=match_score,
                match_type=match_type,
                confidence_score=confidence_score,
                response_time_ms=response_time_ms,
                cost_usd=cost_usd,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                grading_details=grading_details,
                error=error
            )
            
            # Log as JSON line
            json_entry = json.dumps(asdict(entry), ensure_ascii=False)
            self.debug_logger.debug(json_entry)
            
            # Log readable summary
            status = "✓" if is_correct else "✗"
            error_info = f" ERROR: {error}" if error else ""
            
            self.summary_logger.debug(
                f"{status} Q{question_id} [{model_name}] {category or 'N/A'} "
                f"Score: {match_score:.3f} ({match_type}) "
                f"Time: {response_time_ms:.0f}ms Cost: ${cost_usd:.6f}{error_info}"
            )
            
            # Log detailed prompt/response for incorrect answers or errors
            if not is_correct or error:
                self.summary_logger.debug(f"  Question: {question_text}")
                self.summary_logger.debug(f"  Correct:  {correct_answer}")
                self.summary_logger.debug(f"  Response: {raw_response}")
                self.summary_logger.debug(f"  Parsed:   {parsed_answer}")
                if grading_details:
                    self.summary_logger.debug(f"  Grading:  {grading_details}")
                self.summary_logger.debug("-" * 80)
            
        except Exception as e:
            logger.error(f"Failed to log debug entry: {str(e)}")
    
    def log_prompt_only(
        self,
        question_id: Any,
        model_name: str,
        question_text: str,
        formatted_prompt: str
    ) -> None:
        """Log just the prompt for debugging prompt formatting issues."""
        if not self.debug_enabled:
            return
        
        try:
            self.summary_logger.debug(f"PROMPT Q{question_id} [{model_name}]:")
            self.summary_logger.debug(f"  Question: {question_text}")
            self.summary_logger.debug(f"  Prompt:")
            for line in formatted_prompt.split('\n'):
                self.summary_logger.debug(f"    {line}")
            self.summary_logger.debug("-" * 40)
        except Exception as e:
            logger.error(f"Failed to log prompt: {str(e)}")
    
    def log_response_only(
        self,
        question_id: Any,
        model_name: str,
        raw_response: str,
        parsed_answer: str
    ) -> None:
        """Log just the response for debugging parsing issues."""
        if not self.debug_enabled:
            return
        
        try:
            self.summary_logger.debug(f"RESPONSE Q{question_id} [{model_name}]:")
            self.summary_logger.debug(f"  Raw: {raw_response}")
            self.summary_logger.debug(f"  Parsed: {parsed_answer}")
            self.summary_logger.debug("-" * 40)
        except Exception as e:
            logger.error(f"Failed to log response: {str(e)}")
    
    def log_grading_details(
        self,
        question_id: Any,
        model_name: str,
        parsed_answer: str,
        correct_answer: str,
        is_correct: bool,
        match_score: float,
        match_type: str,
        grading_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log detailed grading information."""
        if not self.debug_enabled:
            return
        
        try:
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            self.summary_logger.debug(f"GRADING Q{question_id} [{model_name}] {status}:")
            self.summary_logger.debug(f"  Answer:   {parsed_answer}")
            self.summary_logger.debug(f"  Expected: {correct_answer}")
            self.summary_logger.debug(f"  Score:    {match_score:.3f} ({match_type})")
            if grading_details:
                for key, value in grading_details.items():
                    self.summary_logger.debug(f"  {key}: {value}")
            self.summary_logger.debug("-" * 40)
        except Exception as e:
            logger.error(f"Failed to log grading details: {str(e)}")
    
    def close(self) -> None:
        """Close debug logger and clean up resources."""
        if not self.debug_enabled:
            return
        
        try:
            for handler in self.debug_logger.handlers[:]:
                handler.close()
                self.debug_logger.removeHandler(handler)
            
            for handler in self.summary_logger.handlers[:]:
                handler.close()
                self.summary_logger.removeHandler(handler)
                
        except Exception as e:
            logger.error(f"Failed to close debug logger: {str(e)}")


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get the global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        # Initialize with default settings - will be reconfigured when needed
        _debug_logger = DebugLogger(debug_enabled=False)
    return _debug_logger


def initialize_debug_logger(debug_enabled: bool, log_dir: str = "logs/debug") -> DebugLogger:
    """Initialize the global debug logger with specific settings."""
    global _debug_logger
    _debug_logger = DebugLogger(debug_enabled=debug_enabled, log_dir=log_dir)
    return _debug_logger