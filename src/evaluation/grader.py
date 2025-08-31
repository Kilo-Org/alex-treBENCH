"""
Answer Grading System

Grades model responses using fuzzy matching with support for different grading modes,
partial credit calculation, and comprehensive metadata tracking.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime

from .matcher import FuzzyMatcher, MatchResult, MatchType
from ..models.response_parser import ResponseParser, ParsedResponse, ResponseType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GradingMode(str, Enum):
    """Grading modes for answer evaluation."""
    STRICT = "strict"           # Exact match required
    LENIENT = "lenient"        # Fuzzy match acceptable
    JEOPARDY = "jeopardy"      # Must be in question format
    ADAPTIVE = "adaptive"       # Adapts based on question type


@dataclass
class GradingCriteria:
    """Criteria for answer grading."""
    mode: GradingMode
    confidence_threshold: float = 0.70
    partial_credit_enabled: bool = True
    jeopardy_format_required: bool = False
    fuzzy_threshold: float = 0.80
    semantic_threshold: float = 0.70
    
    # Partial credit thresholds
    partial_credit_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.partial_credit_thresholds is None:
            self.partial_credit_thresholds = {
                'high': 0.90,      # 90-100% credit
                'medium': 0.70,    # 70-89% credit
                'low': 0.50,       # 50-69% credit
                'minimal': 0.30    # 30-49% credit
            }


@dataclass
class GradedResponse:
    """Result of grading a response."""
    is_correct: bool
    score: float  # 0.0 to 1.0
    confidence: float
    partial_credit: float
    match_result: MatchResult
    parsed_response: ParsedResponse
    grading_metadata: Dict[str, Any]
    grade_explanation: str
    timestamp: datetime


class AnswerGrader:
    """Grades model responses using fuzzy matching and multiple criteria."""
    
    def __init__(self, default_criteria: Optional[GradingCriteria] = None):
        """
        Initialize the answer grader.
        
        Args:
            default_criteria: Default grading criteria
        """
        self.default_criteria = default_criteria or GradingCriteria(mode=GradingMode.LENIENT)
        
        # Initialize components
        self.response_parser = ResponseParser()
        
        # Track grading statistics
        self.grading_stats = {
            'total_graded': 0,
            'correct_count': 0,
            'incorrect_count': 0,
            'partial_credit_count': 0,
            'avg_confidence': 0.0,
            'match_type_distribution': {},
            'response_type_distribution': {}
        }
    
    def grade_response(self, 
                      response_text: str,
                      correct_answer: str,
                      question_context: Optional[Dict[str, Any]] = None,
                      multiple_acceptable: Optional[List[str]] = None,
                      criteria: Optional[GradingCriteria] = None) -> GradedResponse:
        """
        Grade a single response.
        
        Args:
            response_text: The model's response text
            correct_answer: The expected correct answer
            question_context: Additional context about the question
            multiple_acceptable: List of alternative acceptable answers
            criteria: Grading criteria (uses default if None)
            
        Returns:
            GradedResponse with detailed grading information
        """
        start_time = time.time()
        criteria = criteria or self.default_criteria
        
        try:
            # Parse the response
            parsed_response = self.response_parser.parse_response(response_text)
            
            # Handle special cases first
            if parsed_response.response_type == ResponseType.REFUSAL:
                return self._create_failed_grade(
                    parsed_response, correct_answer, "Response was a refusal", criteria
                )
            
            if parsed_response.response_type == ResponseType.ERROR:
                return self._create_failed_grade(
                    parsed_response, correct_answer, "Response contained errors", criteria
                )
            
            # Create matcher with appropriate settings
            matcher = self._create_matcher(criteria, question_context)
            
            # Match the answer
            match_result = matcher.match_answer(
                parsed_response.extracted_answer,
                correct_answer,
                multiple_acceptable
            )
            
            # Determine if correct based on grading mode
            is_correct, score, explanation = self._evaluate_match(
                match_result, parsed_response, criteria
            )
            
            # Calculate partial credit
            partial_credit = self._calculate_partial_credit(
                match_result, parsed_response, criteria
            )
            
            # Create grading metadata
            grading_metadata = self._create_grading_metadata(
                match_result, parsed_response, criteria, question_context, 
                time.time() - start_time
            )
            
            # Update statistics
            self._update_stats(match_result, parsed_response, is_correct)
            
            return GradedResponse(
                is_correct=is_correct,
                score=score,
                confidence=match_result.confidence,
                partial_credit=partial_credit,
                match_result=match_result,
                parsed_response=parsed_response,
                grading_metadata=grading_metadata,
                grade_explanation=explanation,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Grading failed for response: {str(e)}")
            parsed_response = ParsedResponse(
                original_text=response_text,
                extracted_answer="",
                response_type=ResponseType.ERROR,
                confidence_indicators=[]
            )
            return self._create_failed_grade(
                parsed_response, correct_answer, f"Grading error: {str(e)}", criteria
            )
    
    def grade_batch(self,
                   responses: List[str],
                   correct_answers: List[str],
                   question_contexts: Optional[List[Dict[str, Any]]] = None,
                   multiple_acceptable: Optional[List[List[str]]] = None,
                   criteria: Optional[GradingCriteria] = None) -> List[GradedResponse]:
        """
        Grade multiple responses in batch.
        
        Args:
            responses: List of response texts
            correct_answers: List of correct answers
            question_contexts: Optional list of question contexts
            multiple_acceptable: Optional list of alternative answers
            criteria: Grading criteria
            
        Returns:
            List of GradedResponse objects
        """
        if len(responses) != len(correct_answers):
            raise ValueError("Responses and correct answers must have the same length")
        
        logger.info(f"Grading batch of {len(responses)} responses")
        
        graded_responses = []
        for i, (response, correct) in enumerate(zip(responses, correct_answers)):
            context = question_contexts[i] if question_contexts else None
            acceptable = multiple_acceptable[i] if multiple_acceptable else None
            
            graded = self.grade_response(response, correct, context, acceptable, criteria)
            graded_responses.append(graded)
        
        logger.info(f"Batch grading complete: {sum(1 for g in graded_responses if g.is_correct)}/{len(graded_responses)} correct")
        return graded_responses
    
    def _create_matcher(self, criteria: GradingCriteria, 
                       question_context: Optional[Dict[str, Any]]) -> FuzzyMatcher:
        """Create a matcher with appropriate settings."""
        jeopardy_required = criteria.jeopardy_format_required
        
        # Adjust requirements based on mode
        if criteria.mode == GradingMode.JEOPARDY:
            jeopardy_required = True
        elif criteria.mode == GradingMode.STRICT:
            jeopardy_required = False
        
        # Consider question context
        if question_context:
            category = question_context.get('category', '').lower()
            if 'jeopardy' in category or question_context.get('format') == 'jeopardy':
                jeopardy_required = True
        
        return FuzzyMatcher(
            fuzzy_threshold=criteria.fuzzy_threshold,
            semantic_threshold=criteria.semantic_threshold,
            jeopardy_format_required=jeopardy_required
        )
    
    def _evaluate_match(self, match_result: MatchResult, parsed_response: ParsedResponse,
                       criteria: GradingCriteria) -> Tuple[bool, float, str]:
        """
        Evaluate whether a match constitutes a correct answer.
        
        Returns:
            Tuple of (is_correct, score, explanation)
        """
        if criteria.mode == GradingMode.STRICT:
            # Strict mode: only exact matches count
            is_correct = (match_result.is_match and 
                         match_result.match_type == MatchType.EXACT)
            score = 1.0 if is_correct else 0.0
            explanation = "Strict mode: exact match required"
            
        elif criteria.mode == GradingMode.LENIENT:
            # Lenient mode: any high-confidence match counts
            is_correct = (match_result.is_match and 
                         match_result.confidence >= criteria.confidence_threshold)
            score = match_result.confidence if is_correct else 0.0
            explanation = f"Lenient mode: confidence {match_result.confidence:.3f}"
            
        elif criteria.mode == GradingMode.JEOPARDY:
            # Jeopardy mode: must be in question format with correct content
            has_format = parsed_response.response_type == ResponseType.JEOPARDY_FORMAT
            is_correct = (match_result.is_match and has_format and
                         match_result.confidence >= criteria.confidence_threshold)
            score = match_result.confidence * (1.0 if has_format else 0.5)
            explanation = f"Jeopardy mode: format={has_format}, confidence={match_result.confidence:.3f}"
            
        elif criteria.mode == GradingMode.ADAPTIVE:
            # Adaptive mode: adjust based on response and question characteristics
            base_threshold = criteria.confidence_threshold
            
            # Lower threshold for uncertain responses
            if parsed_response.confidence_indicators:
                base_threshold *= 0.9
            
            # Higher threshold for refusals that somehow matched
            if parsed_response.response_type == ResponseType.REFUSAL:
                base_threshold *= 1.2
            
            is_correct = (match_result.is_match and 
                         match_result.confidence >= base_threshold)
            score = match_result.confidence
            explanation = f"Adaptive mode: adjusted threshold {base_threshold:.3f}"
            
        else:
            # Default fallback
            is_correct = match_result.is_match
            score = match_result.confidence
            explanation = "Default evaluation"
        
        return is_correct, score, explanation
    
    def _calculate_partial_credit(self, match_result: MatchResult, 
                                parsed_response: ParsedResponse,
                                criteria: GradingCriteria) -> float:
        """Calculate partial credit score."""
        if not criteria.partial_credit_enabled:
            return 1.0 if match_result.is_match else 0.0
        
        confidence = match_result.confidence
        thresholds = criteria.partial_credit_thresholds
        
        # Full credit for high confidence
        if confidence >= thresholds['high']:
            return 1.0
        elif confidence >= thresholds['medium']:
            return 0.85
        elif confidence >= thresholds['low']:
            return 0.65
        elif confidence >= thresholds['minimal']:
            return 0.45
        else:
            return 0.0
    
    def _create_grading_metadata(self, match_result: MatchResult,
                               parsed_response: ParsedResponse,
                               criteria: GradingCriteria,
                               question_context: Optional[Dict[str, Any]],
                               grading_time: float) -> Dict[str, Any]:
        """Create comprehensive grading metadata."""
        return {
            'grading_mode': criteria.mode.value,
            'confidence_threshold': criteria.confidence_threshold,
            'partial_credit_enabled': criteria.partial_credit_enabled,
            'match_type': match_result.match_type.value,
            'match_confidence': match_result.confidence,
            'match_details': match_result.details,
            'response_type': parsed_response.response_type.value,
            'response_quality_score': self.response_parser.get_response_quality_score(parsed_response),
            'confidence_indicators': parsed_response.confidence_indicators,
            'has_reasoning': parsed_response.reasoning is not None,
            'grading_time_ms': int(grading_time * 1000),
            'normalized_answer': match_result.normalized_answer,
            'normalized_expected': match_result.normalized_expected,
            'question_context': question_context or {}
        }
    
    def _create_failed_grade(self, parsed_response: ParsedResponse,
                           correct_answer: str, reason: str,
                           criteria: GradingCriteria) -> GradedResponse:
        """Create a failed grade result."""
        dummy_match = MatchResult(
            is_match=False,
            confidence=0.0,
            match_type=MatchType.EXACT,
            details={'failure_reason': reason},
            normalized_answer=parsed_response.extracted_answer,
            normalized_expected=correct_answer
        )
        
        return GradedResponse(
            is_correct=False,
            score=0.0,
            confidence=0.0,
            partial_credit=0.0,
            match_result=dummy_match,
            parsed_response=parsed_response,
            grading_metadata={
                'failure_reason': reason,
                'grading_mode': criteria.mode.value
            },
            grade_explanation=reason,
            timestamp=datetime.now()
        )
    
    def _update_stats(self, match_result: MatchResult, 
                     parsed_response: ParsedResponse, is_correct: bool):
        """Update grading statistics."""
        self.grading_stats['total_graded'] += 1
        
        if is_correct:
            self.grading_stats['correct_count'] += 1
        else:
            self.grading_stats['incorrect_count'] += 1
        
        # Update confidence average
        total = self.grading_stats['total_graded']
        old_avg = self.grading_stats['avg_confidence']
        new_confidence = match_result.confidence
        self.grading_stats['avg_confidence'] = (old_avg * (total - 1) + new_confidence) / total
        
        # Update distributions
        match_type = match_result.match_type.value
        self.grading_stats['match_type_distribution'][match_type] = \
            self.grading_stats['match_type_distribution'].get(match_type, 0) + 1
        
        response_type = parsed_response.response_type.value
        self.grading_stats['response_type_distribution'][response_type] = \
            self.grading_stats['response_type_distribution'].get(response_type, 0) + 1
    
    def get_grading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive grading statistics."""
        total = self.grading_stats['total_graded']
        
        if total == 0:
            return {'message': 'No responses graded yet'}
        
        stats = self.grading_stats.copy()
        stats['accuracy_rate'] = self.grading_stats['correct_count'] / total
        stats['error_rate'] = self.grading_stats['incorrect_count'] / total
        
        # Convert counts to percentages for distributions
        for dist_key in ['match_type_distribution', 'response_type_distribution']:
            distribution = stats[dist_key]
            stats[dist_key + '_pct'] = {
                k: (v / total) * 100 for k, v in distribution.items()
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset grading statistics."""
        self.grading_stats = {
            'total_graded': 0,
            'correct_count': 0,
            'incorrect_count': 0,
            'partial_credit_count': 0,
            'avg_confidence': 0.0,
            'match_type_distribution': {},
            'response_type_distribution': {}
        }
    
    def create_adaptive_criteria(self, question_context: Dict[str, Any]) -> GradingCriteria:
        """Create adaptive grading criteria based on question context."""
        category = question_context.get('category', '').lower()
        difficulty = question_context.get('difficulty_level', '').lower()
        value = question_context.get('value', 0)
        
        # Base criteria
        criteria = GradingCriteria(mode=GradingMode.ADAPTIVE)
        
        # Adjust thresholds based on category
        if 'science' in category or 'history' in category:
            # More lenient for factual categories
            criteria.confidence_threshold = 0.65
            criteria.fuzzy_threshold = 0.75
        elif 'wordplay' in category or 'language' in category:
            # Stricter for wordplay
            criteria.confidence_threshold = 0.80
            criteria.fuzzy_threshold = 0.85
        
        # Adjust based on difficulty
        if difficulty == 'hard' or value > 1000:
            # More lenient for hard questions
            criteria.confidence_threshold *= 0.9
        elif difficulty == 'easy' or value < 400:
            # Stricter for easy questions
            criteria.confidence_threshold *= 1.1
        
        return criteria
    
    def explain_grade(self, graded_response: GradedResponse) -> str:
        """Generate human-readable explanation of the grade."""
        lines = []
        
        lines.append(f"Grade: {'CORRECT' if graded_response.is_correct else 'INCORRECT'}")
        lines.append(f"Score: {graded_response.score:.3f}")
        lines.append(f"Confidence: {graded_response.confidence:.3f}")
        
        if graded_response.partial_credit > 0:
            lines.append(f"Partial Credit: {graded_response.partial_credit:.3f}")
        
        lines.append(f"Match Type: {graded_response.match_result.match_type.value}")
        lines.append(f"Response Type: {graded_response.parsed_response.response_type.value}")
        
        if graded_response.parsed_response.confidence_indicators:
            lines.append(f"Uncertainty Indicators: {', '.join(graded_response.parsed_response.confidence_indicators)}")
        
        lines.append(f"Explanation: {graded_response.grade_explanation}")
        
        return "\n".join(lines)