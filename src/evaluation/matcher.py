"""
Answer Matching System

Implements fuzzy matching algorithms for comparing model answers to correct answers
with support for multiple matching strategies and confidence scoring.
"""

import re
import string
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import difflib
from fuzzywuzzy import fuzz, process
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {str(e)}")


class MatchType(str, Enum):
    """Types of answer matching strategies."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    JEOPARDY_FORMAT = "jeopardy_format"
    NUMBER_DATE = "number_date"
    ABBREVIATION = "abbreviation"


@dataclass
class MatchResult:
    """Result of answer matching."""
    is_match: bool
    confidence: float
    match_type: MatchType
    details: Dict[str, Any]
    normalized_answer: str
    normalized_expected: str


class FuzzyMatcher:
    """Implements fuzzy matching algorithms for answer comparison."""
    
    def __init__(self, 
                 fuzzy_threshold: float = 0.80,
                 semantic_threshold: float = 0.70,
                 jeopardy_format_required: bool = False):
        """
        Initialize the fuzzy matcher.
        
        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matching
            semantic_threshold: Minimum similarity score for semantic matching
            jeopardy_format_required: Whether answers must be in Jeopardy format
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.jeopardy_format_required = jeopardy_format_required
        
        # Initialize NLTK components
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {str(e)}")
            self.lemmatizer = None
            self.stop_words = set()
        
        # Common abbreviations and their expansions
        self.abbreviations = {
            'usa': ['united states', 'united states of america', 'america'],
            'uk': ['united kingdom', 'britain', 'great britain'],
            'nyc': ['new york city', 'new york'],
            'la': ['los angeles'],
            'sf': ['san francisco'],
            'dc': ['washington dc', 'washington d.c.'],
            'jr': ['junior'],
            'sr': ['senior'],
            'dr': ['doctor'],
            'mr': ['mister'],
            'mrs': ['missus'],
            'ms': ['miss'],
            'st': ['saint', 'street'],
            'ave': ['avenue'],
            'blvd': ['boulevard'],
            'rd': ['road'],
            'ft': ['fort'],
            'mt': ['mount', 'mountain'],
        }
        
        # Number words mapping
        self.number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
            'million': '1000000', 'billion': '1000000000'
        }
        
        # Jeopardy question patterns
        self.jeopardy_patterns = [
            r'^what\s+is\s+',
            r'^who\s+is\s+',
            r'^where\s+is\s+',
            r'^when\s+is\s+',
            r'^why\s+is\s+',
            r'^how\s+is\s+',
            r'^what\s+are\s+',
            r'^who\s+are\s+',
            r'^where\s+are\s+',
            r'^when\s+are\s+',
            r'^what\s+was\s+',
            r'^who\s+was\s+',
            r'^where\s+was\s+',
            r'^when\s+was\s+',
            r'^what\s+were\s+',
            r'^who\s+were\s+',
            r'^where\s+were\s+',
            r'^when\s+were\s+'
        ]
    
    def match_answer(self, answer: str, expected: str, 
                    multiple_acceptable: Optional[List[str]] = None) -> MatchResult:
        """
        Match an answer against the expected answer using multiple strategies.
        
        Args:
            answer: The answer to match
            expected: The expected correct answer
            multiple_acceptable: List of alternative acceptable answers
            
        Returns:
            MatchResult with match details
        """
        try:
            # Handle empty inputs
            if not answer or not answer.strip():
                return MatchResult(
                    is_match=False,
                    confidence=0.0,
                    match_type=MatchType.EXACT,
                    details={'error': 'Empty answer'},
                    normalized_answer='',
                    normalized_expected=self._normalize_text(expected)
                )
            
            # Normalize inputs
            norm_answer = self._normalize_text(answer)
            norm_expected = self._normalize_text(expected)
            
            # Collect all acceptable answers
            all_acceptable = [expected]
            if multiple_acceptable:
                all_acceptable.extend(multiple_acceptable)
            
            # Try each matching strategy
            strategies = [
                self._exact_match,
                self._jeopardy_format_match,
                self._number_date_match,
                self._abbreviation_match,
                self._fuzzy_match,
                self._semantic_match
            ]
            
            best_result = None
            for strategy in strategies:
                for acceptable in all_acceptable:
                    result = strategy(norm_answer, self._normalize_text(acceptable), 
                                    answer, acceptable)
                    
                    if result.is_match:
                        return result
                    
                    # Keep track of best result even if not matching
                    if best_result is None or result.confidence > best_result.confidence:
                        best_result = result
            
            # Return best result if no match found
            return best_result or MatchResult(
                is_match=False,
                confidence=0.0,
                match_type=MatchType.EXACT,
                details={'no_match': 'No matching strategy succeeded'},
                normalized_answer=norm_answer,
                normalized_expected=norm_expected
            )
            
        except Exception as e:
            logger.error(f"Answer matching failed: {str(e)}")
            return MatchResult(
                is_match=False,
                confidence=0.0,
                match_type=MatchType.EXACT,
                details={'error': str(e)},
                normalized_answer=answer,
                normalized_expected=expected
            )
    
    def _exact_match(self, norm_answer: str, norm_expected: str,
                    original_answer: str, original_expected: str) -> MatchResult:
        """Perform exact string matching."""
        is_match = norm_answer == norm_expected
        confidence = 1.0 if is_match else 0.0
        
        return MatchResult(
            is_match=is_match,
            confidence=confidence,
            match_type=MatchType.EXACT,
            details={'exact_match': is_match},
            normalized_answer=norm_answer,
            normalized_expected=norm_expected
        )
    
    def _fuzzy_match(self, norm_answer: str, norm_expected: str,
                    original_answer: str, original_expected: str) -> MatchResult:
        """Perform fuzzy string matching using multiple algorithms."""
        # Calculate different fuzzy scores
        ratio = fuzz.ratio(norm_answer, norm_expected) / 100.0
        partial_ratio = fuzz.partial_ratio(norm_answer, norm_expected) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(norm_answer, norm_expected) / 100.0
        token_set_ratio = fuzz.token_set_ratio(norm_answer, norm_expected) / 100.0
        
        # Use the best score
        best_score = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        is_match = best_score >= self.fuzzy_threshold
        
        return MatchResult(
            is_match=is_match,
            confidence=best_score,
            match_type=MatchType.FUZZY,
            details={
                'ratio': ratio,
                'partial_ratio': partial_ratio,
                'token_sort_ratio': token_sort_ratio,
                'token_set_ratio': token_set_ratio,
                'best_score': best_score
            },
            normalized_answer=norm_answer,
            normalized_expected=norm_expected
        )
    
    def _semantic_match(self, norm_answer: str, norm_expected: str,
                       original_answer: str, original_expected: str) -> MatchResult:
        """Perform semantic matching using synonyms and word relationships."""
        if not self.lemmatizer:
            return MatchResult(
                is_match=False,
                confidence=0.0,
                match_type=MatchType.SEMANTIC,
                details={'error': 'NLTK not available'},
                normalized_answer=norm_answer,
                normalized_expected=norm_expected
            )
        
        try:
            # Tokenize and lemmatize
            answer_tokens = self._get_meaningful_tokens(norm_answer)
            expected_tokens = self._get_meaningful_tokens(norm_expected)
            
            if not answer_tokens or not expected_tokens:
                return MatchResult(
                    is_match=False,
                    confidence=0.0,
                    match_type=MatchType.SEMANTIC,
                    details={'no_tokens': True},
                    normalized_answer=norm_answer,
                    normalized_expected=norm_expected
                )
            
            # Calculate semantic similarity
            similarity = self._calculate_semantic_similarity(answer_tokens, expected_tokens)
            is_match = similarity >= self.semantic_threshold
            
            return MatchResult(
                is_match=is_match,
                confidence=similarity,
                match_type=MatchType.SEMANTIC,
                details={
                    'semantic_similarity': similarity,
                    'answer_tokens': answer_tokens,
                    'expected_tokens': expected_tokens
                },
                normalized_answer=norm_answer,
                normalized_expected=norm_expected
            )
            
        except Exception as e:
            logger.warning(f"Semantic matching failed: {str(e)}")
            return MatchResult(
                is_match=False,
                confidence=0.0,
                match_type=MatchType.SEMANTIC,
                details={'error': str(e)},
                normalized_answer=norm_answer,
                normalized_expected=norm_expected
            )
    
    def _jeopardy_format_match(self, norm_answer: str, norm_expected: str,
                              original_answer: str, original_expected: str) -> MatchResult:
        """Match Jeopardy format answers."""
        # Extract content from Jeopardy format
        answer_content = self._extract_jeopardy_content(norm_answer)
        expected_content = self._extract_jeopardy_content(norm_expected)
        
        # If both have Jeopardy format, compare the content
        if answer_content and expected_content:
            is_match = answer_content == expected_content
            confidence = 1.0 if is_match else 0.7  # Partial credit for format
        elif answer_content:  # Answer has format, expected doesn't
            is_match = answer_content == norm_expected
            confidence = 0.9 if is_match else 0.0
        elif expected_content:  # Expected has format, answer doesn't
            is_match = norm_answer == expected_content
            confidence = 0.8 if is_match else 0.0
        else:
            # Neither has Jeopardy format
            is_match = False
            confidence = 0.0
        
        # Check if Jeopardy format is required but missing
        if self.jeopardy_format_required and not self._has_jeopardy_format(original_answer):
            confidence *= 0.5  # Penalize missing format
        
        return MatchResult(
            is_match=is_match,
            confidence=confidence,
            match_type=MatchType.JEOPARDY_FORMAT,
            details={
                'answer_has_format': bool(answer_content),
                'expected_has_format': bool(expected_content),
                'answer_content': answer_content,
                'expected_content': expected_content
            },
            normalized_answer=norm_answer,
            normalized_expected=norm_expected
        )
    
    def _number_date_match(self, norm_answer: str, norm_expected: str,
                          original_answer: str, original_expected: str) -> MatchResult:
        """Match numbers and dates with flexible formatting."""
        # Extract numbers from both strings
        answer_numbers = self._extract_numbers(norm_answer)
        expected_numbers = self._extract_numbers(norm_expected)
        
        # Convert number words to digits
        answer_with_digits = self._convert_number_words(norm_answer)
        expected_with_digits = self._convert_number_words(norm_expected)
        
        answer_numbers.extend(self._extract_numbers(answer_with_digits))
        expected_numbers.extend(self._extract_numbers(expected_with_digits))
        
        # Check for number matches
        number_matches = len(set(answer_numbers) & set(expected_numbers))
        total_numbers = max(len(expected_numbers), 1)
        
        # Extract years (4-digit numbers)
        answer_years = self._extract_years(norm_answer)
        expected_years = self._extract_years(norm_expected)
        year_matches = len(set(answer_years) & set(expected_years))
        
        # Calculate confidence
        number_confidence = number_matches / total_numbers if expected_numbers else 0
        year_confidence = 1.0 if year_matches > 0 else 0.0
        
        overall_confidence = max(number_confidence, year_confidence)
        is_match = overall_confidence >= 0.8
        
        return MatchResult(
            is_match=is_match,
            confidence=overall_confidence,
            match_type=MatchType.NUMBER_DATE,
            details={
                'answer_numbers': answer_numbers,
                'expected_numbers': expected_numbers,
                'answer_years': answer_years,
                'expected_years': expected_years,
                'number_matches': number_matches,
                'year_matches': year_matches
            },
            normalized_answer=norm_answer,
            normalized_expected=norm_expected
        )
    
    def _abbreviation_match(self, norm_answer: str, norm_expected: str,
                           original_answer: str, original_expected: str) -> MatchResult:
        """Match abbreviations with their full forms."""
        # Expand abbreviations in both strings
        expanded_answer = self._expand_abbreviations(norm_answer)
        expanded_expected = self._expand_abbreviations(norm_expected)
        
        # Check various combinations
        matches = [
            norm_answer == norm_expected,
            expanded_answer == norm_expected,
            norm_answer == expanded_expected,
            expanded_answer == expanded_expected
        ]
        
        is_match = any(matches)
        confidence = 1.0 if is_match else 0.0
        
        return MatchResult(
            is_match=is_match,
            confidence=confidence,
            match_type=MatchType.ABBREVIATION,
            details={
                'expanded_answer': expanded_answer,
                'expanded_expected': expanded_expected,
                'match_combinations': matches
            },
            normalized_answer=norm_answer,
            normalized_expected=norm_expected
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove punctuation except apostrophes and hyphens
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove articles at the beginning
        text = re.sub(r'^(the|a|an)\s+', '', text)
        
        return text
    
    def _extract_jeopardy_content(self, text: str) -> Optional[str]:
        """Extract content from Jeopardy format question."""
        for pattern in self.jeopardy_patterns:
            match = re.search(pattern + r'(.+?)(\?|$)', text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _has_jeopardy_format(self, text: str) -> bool:
        """Check if text has Jeopardy question format."""
        text_lower = text.lower().strip()
        return any(re.match(pattern, text_lower) for pattern in self.jeopardy_patterns)
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from text."""
        # Find all numeric patterns
        patterns = [
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas and decimals
            r'\b\d+\b',  # Simple integers
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'  # Dollar amounts
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend([re.sub(r'[,$]', '', match) for match in matches])
        
        return list(set(numbers))
    
    def _extract_years(self, text: str) -> List[str]:
        """Extract 4-digit years from text."""
        return re.findall(r'\b(1\d{3}|20\d{2})\b', text)
    
    def _convert_number_words(self, text: str) -> str:
        """Convert number words to digits."""
        for word, digit in self.number_words.items():
            text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, expansions in self.abbreviations.items():
            for expansion in expansions:
                text = re.sub(r'\b' + abbrev + r'\b', expansion, text, flags=re.IGNORECASE)
        return text
    
    def _get_meaningful_tokens(self, text: str) -> List[str]:
        """Get meaningful tokens (excluding stop words)."""
        if not self.lemmatizer:
            return text.split()
        
        try:
            tokens = word_tokenize(text)
            meaningful_tokens = []
            
            for token in tokens:
                if (token.lower() not in self.stop_words and 
                    token.isalnum() and 
                    len(token) > 2):
                    lemmatized = self.lemmatizer.lemmatize(token.lower())
                    meaningful_tokens.append(lemmatized)
            
            return meaningful_tokens
        except Exception:
            return text.split()
    
    def _calculate_semantic_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate semantic similarity between token sets."""
        if not tokens1 or not tokens2:
            return 0.0
        
        # Direct token overlap
        overlap = len(set(tokens1) & set(tokens2))
        union = len(set(tokens1) | set(tokens2))
        jaccard = overlap / union if union > 0 else 0.0
        
        # WordNet similarity (if available)
        wordnet_sim = 0.0
        if wordnet:
            try:
                similarities = []
                for token1 in tokens1:
                    best_sim = 0.0
                    for token2 in tokens2:
                        synsets1 = wordnet.synsets(token1)
                        synsets2 = wordnet.synsets(token2)
                        
                        for syn1 in synsets1:
                            for syn2 in synsets2:
                                sim = syn1.wup_similarity(syn2)
                                if sim and sim > best_sim:
                                    best_sim = sim
                    
                    similarities.append(best_sim)
                
                wordnet_sim = sum(similarities) / len(similarities) if similarities else 0.0
            except Exception:
                wordnet_sim = 0.0
        
        # Combine similarities
        return max(jaccard, wordnet_sim * 0.8)
    
    def batch_match(self, answers: List[str], expected: List[str],
                   multiple_acceptable: Optional[List[List[str]]] = None) -> List[MatchResult]:
        """Match multiple answers in batch."""
        if len(answers) != len(expected):
            raise ValueError("Answers and expected lists must have the same length")
        
        results = []
        for i, (answer, expect) in enumerate(zip(answers, expected)):
            acceptable = multiple_acceptable[i] if multiple_acceptable else None
            result = self.match_answer(answer, expect, acceptable)
            results.append(result)
        
        return results
    
    def get_match_confidence_distribution(self, results: List[MatchResult]) -> Dict[str, float]:
        """Get distribution of match confidence scores."""
        if not results:
            return {}
        
        confidences = [r.confidence for r in results]
        return {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'median': sorted(confidences)[len(confidences) // 2],
            'std': (sum((x - sum(confidences)/len(confidences))**2 for x in confidences) / len(confidences))**0.5
        }