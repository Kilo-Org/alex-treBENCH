"""
Response Parser

Parses and cleans model responses to extract Jeopardy answers.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ResponseType(str, Enum):
    """Types of responses from models."""
    JEOPARDY_FORMAT = "jeopardy_format"  # "What is...?"
    DIRECT_ANSWER = "direct_answer"      # "Paris"
    EXPLANATION = "explanation"          # Contains reasoning
    REFUSAL = "refusal"                  # Model refused to answer
    ERROR = "error"                      # Invalid/garbled response
    UNCERTAIN = "uncertain"              # Model expressed uncertainty


@dataclass
class ParsedResponse:
    """Parsed response from a language model."""
    original_text: str
    extracted_answer: str
    response_type: ResponseType
    confidence_indicators: List[str]
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseParser:
    """Parses model responses to extract clean Jeopardy answers."""
    
    # Jeopardy question starters
    JEOPARDY_STARTERS = [
        r"what\s+is\s+",
        r"who\s+is\s+", 
        r"where\s+is\s+",
        r"when\s+is\s+",
        r"why\s+is\s+",
        r"how\s+is\s+",
        r"what\s+are\s+",
        r"who\s+are\s+",
        r"where\s+are\s+",
        r"when\s+are\s+",
        r"why\s+are\s+",
        r"how\s+are\s+",
        r"what\s+was\s+",
        r"who\s+was\s+",
        r"where\s+was\s+",
        r"when\s+was\s+",
        r"why\s+was\s+",
        r"how\s+was\s+",
        r"what\s+were\s+",
        r"who\s+were\s+",
        r"where\s+were\s+",
        r"when\s+were\s+",
        r"why\s+were\s+",
        r"how\s+were\s+",
        r"what\s+would\s+",
        r"who\s+would\s+",
        r"where\s+would\s+",
        r"when\s+would\s+",
        r"why\s+would\s+",
        r"how\s+would\s+"
    ]
    
    # Confidence indicators (words that suggest uncertainty)
    UNCERTAINTY_INDICATORS = [
        "i think", "i believe", "possibly", "probably", "likely", 
        "might be", "could be", "seems to be", "appears to be",
        "i'm not sure", "not certain", "uncertain", "maybe",
        "perhaps", "presumably", "supposedly", "allegedly"
    ]
    
    # Refusal indicators
    REFUSAL_INDICATORS = [
        "i don't know", "i'm not sure", "cannot answer",
        "don't have enough information", "insufficient information",
        "unable to determine", "unclear", "ambiguous",
        "sorry", "apologize", "can't help"
    ]
    
    # Common reasoning patterns
    REASONING_PATTERNS = [
        r"because\s+",
        r"since\s+",
        r"let me think\s+",
        r"step by step",
        r"analysis:",
        r"reasoning:",
        r"explanation:",
        r"this is because",
        r"the answer is.*because"
    ]
    
    def parse_response(self, response_text: str) -> ParsedResponse:
        """
        Parse a model response and extract the answer.
        
        Args:
            response_text: Raw response from the model
            
        Returns:
            ParsedResponse with extracted information
        """
        if not response_text or not response_text.strip():
            return ParsedResponse(
                original_text=response_text,
                extracted_answer="",
                response_type=ResponseType.ERROR,
                confidence_indicators=[]
            )
        
        # Clean the response
        cleaned_text = self._clean_response(response_text)
        
        # Detect response type
        response_type = self._detect_response_type(cleaned_text)
        
        # Extract confidence indicators
        confidence_indicators = self._extract_confidence_indicators(cleaned_text)
        
        # Extract reasoning if present
        reasoning = self._extract_reasoning(cleaned_text)
        
        # Extract the actual answer
        extracted_answer = self._extract_answer(cleaned_text, response_type)
        
        return ParsedResponse(
            original_text=response_text,
            extracted_answer=extracted_answer,
            response_type=response_type,
            confidence_indicators=confidence_indicators,
            reasoning=reasoning,
            metadata={
                'cleaned_text': cleaned_text,
                'has_jeopardy_format': self._has_jeopardy_format(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'contains_reasoning': reasoning is not None
            }
        )
    
    def _clean_response(self, text: str) -> str:
        """Clean and normalize the response text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common response prefixes
        prefixes_to_remove = [
            r"^(answer|response|reply):\s*",
            r"^(the answer is):\s*",
            r"^(my answer is):\s*",
            r"^(i think the answer is):\s*",
            r"^(final answer):\s*"
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, "", text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _detect_response_type(self, text: str) -> ResponseType:
        """Detect the type of response."""
        text_lower = text.lower()
        
        # Check for refusal
        for indicator in self.REFUSAL_INDICATORS:
            if indicator in text_lower:
                return ResponseType.REFUSAL
        
        # Check for Jeopardy format
        if self._has_jeopardy_format(text):
            return ResponseType.JEOPARDY_FORMAT
        
        # Check for reasoning/explanation
        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, text_lower):
                return ResponseType.EXPLANATION
        
        # Check for uncertainty
        for indicator in self.UNCERTAINTY_INDICATORS:
            if indicator in text_lower:
                return ResponseType.UNCERTAIN
        
        # Default to direct answer
        return ResponseType.DIRECT_ANSWER
    
    def _has_jeopardy_format(self, text: str) -> bool:
        """Check if text contains Jeopardy-style question format."""
        text_lower = text.lower()
        for starter in self.JEOPARDY_STARTERS:
            if re.search(starter, text_lower):
                return True
        return False
    
    def _extract_confidence_indicators(self, text: str) -> List[str]:
        """Extract phrases that indicate confidence level."""
        indicators = []
        text_lower = text.lower()
        
        for indicator in self.UNCERTAINTY_INDICATORS:
            if indicator in text_lower:
                indicators.append(indicator)
        
        return indicators
    
    def _extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning or explanation from the response."""
        text_lower = text.lower()
        
        # Look for explicit reasoning sections
        reasoning_matches = []
        for pattern in self.REASONING_PATTERNS:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract text after the reasoning indicator
                start = match.end()
                # Find the end (next sentence or paragraph)
                end = len(text)
                for delimiter in ['.', '\n', '?', '!']:
                    next_delim = text.find(delimiter, start)
                    if next_delim != -1 and next_delim < end:
                        end = next_delim + 1
                
                reasoning_text = text[start:end].strip()
                if reasoning_text and len(reasoning_text) > 10:  # Minimum length
                    reasoning_matches.append(reasoning_text)
        
        if reasoning_matches:
            return '. '.join(reasoning_matches)
        
        # If no explicit reasoning found but response is long, assume it contains reasoning
        if len(text.split()) > 20:
            return text
        
        return None
    
    def _extract_answer(self, text: str, response_type: ResponseType) -> str:
        """Extract the actual answer from the response."""
        if response_type == ResponseType.REFUSAL:
            return ""
        
        if response_type == ResponseType.ERROR:
            return ""
        
        # For Jeopardy format, extract the complete question
        if response_type == ResponseType.JEOPARDY_FORMAT:
            return self._extract_jeopardy_answer(text)
        
        # For other types, extract the key answer
        return self._extract_direct_answer(text)
    
    def _extract_jeopardy_answer(self, text: str) -> str:
        """Extract Jeopardy-format answer (What is...?, Who is...?, etc.)."""
        # Find the first Jeopardy-style question in the text
        for starter in self.JEOPARDY_STARTERS:
            pattern = f"({starter}[^?.!]*[?.!]?)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Ensure it ends with a question mark
                if not answer.endswith('?'):
                    answer += '?'
                return answer.capitalize()
        
        # Fallback: return the first sentence that looks like a question
        sentences = re.split(r'[.!]', text)
        for sentence in sentences:
            if '?' in sentence or any(starter.replace(r'\s+', ' ').replace('\\', '') 
                                    in sentence.lower() for starter in self.JEOPARDY_STARTERS):
                return sentence.strip().capitalize()
        
        # Final fallback: return cleaned text
        return text.strip()
    
    def _extract_direct_answer(self, text: str) -> str:
        """Extract direct answer from non-Jeopardy format responses."""
        # Remove reasoning and extract core answer
        
        # Split by common delimiters and take the first substantial part
        for delimiter in ['.', '\n', '!', '?']:
            parts = text.split(delimiter)
            if parts and len(parts[0].strip()) > 2:
                first_part = parts[0].strip()
                
                # Remove common prefixes from the first part
                answer = re.sub(r'^(the answer is|it is|this is)\s+', '', 
                              first_part, flags=re.IGNORECASE)
                
                # Clean up and return
                answer = answer.strip()
                if answer:
                    return answer
        
        # Fallback: return the whole text cleaned up
        return text.strip()
    
    def parse_batch_responses(self, responses: List[str]) -> List[ParsedResponse]:
        """Parse multiple responses in batch."""
        return [self.parse_response(response) for response in responses]
    
    def extract_answers_only(self, responses: List[str]) -> List[str]:
        """Extract just the answers from multiple responses."""
        parsed = self.parse_batch_responses(responses)
        return [p.extracted_answer for p in parsed]
    
    def get_response_quality_score(self, parsed: ParsedResponse) -> float:
        """
        Calculate a quality score for the response (0-1).
        
        Args:
            parsed: ParsedResponse object
            
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Penalize refusals and errors heavily
        if parsed.response_type in [ResponseType.REFUSAL, ResponseType.ERROR]:
            return 0.0
        
        # Reward Jeopardy format
        if parsed.response_type == ResponseType.JEOPARDY_FORMAT:
            score += 0.2
        
        # Penalize uncertainty
        if parsed.confidence_indicators:
            score -= len(parsed.confidence_indicators) * 0.1
        
        # Penalize very short answers (likely incomplete)
        if len(parsed.extracted_answer) < 5:
            score -= 0.3
        
        # Penalize very long answers (likely verbose/unfocused)
        if len(parsed.extracted_answer.split()) > 50:
            score -= 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))