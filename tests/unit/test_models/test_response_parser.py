"""
Tests for Response Parser
"""

import pytest
from src.models.response_parser import (
    ResponseParser, ResponseType, ParsedResponse
)


class TestResponseParser:
    """Test cases for ResponseParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ResponseParser()
    
    def test_parse_empty_response(self):
        """Test parsing empty or None response."""
        # Test empty string
        parsed = self.parser.parse_response("")
        assert parsed.original_text == ""
        assert parsed.extracted_answer == ""
        assert parsed.response_type == ResponseType.ERROR
        assert parsed.confidence_indicators == []
        
        # Test None
        parsed = self.parser.parse_response(None)
        assert parsed.original_text is None
        assert parsed.extracted_answer == ""
        assert parsed.response_type == ResponseType.ERROR
    
    def test_parse_jeopardy_format_response(self):
        """Test parsing Jeopardy-format responses."""
        test_cases = [
            "What is gold?",
            "Who is Shakespeare?",
            "Where is Paris?",
            "When is 1969?",
            "What are the Beatles?",
            "Who was Napoleon?",
        ]
        
        for response_text in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            assert parsed.original_text == response_text
            assert parsed.response_type == ResponseType.JEOPARDY_FORMAT
            assert response_text in parsed.extracted_answer or response_text.lower() in parsed.extracted_answer.lower()
            assert len(parsed.confidence_indicators) == 0  # No uncertainty indicators
    
    def test_parse_direct_answer_response(self):
        """Test parsing direct answer responses."""
        test_cases = [
            "Gold",
            "William Shakespeare",
            "Paris, France",
            "The Beatles",
        ]
        
        for response_text in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            assert parsed.original_text == response_text
            assert parsed.response_type == ResponseType.DIRECT_ANSWER
            assert parsed.extracted_answer == response_text
    
    def test_parse_explanation_response(self):
        """Test parsing responses with explanations."""
        test_cases = [
            "Let me think about this step by step. The answer is gold because it has the chemical symbol Au.",
            "This is because Paris is the capital of France, so the answer is Paris.",
            "Since the question asks about Shakespeare, the answer is William Shakespeare.",
        ]
        
        for response_text in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            assert parsed.original_text == response_text
            assert parsed.response_type == ResponseType.EXPLANATION
            assert parsed.reasoning is not None
            assert len(parsed.reasoning) > 0
    
    def test_parse_uncertain_response(self):
        """Test parsing responses with uncertainty indicators."""
        test_cases = [
            "I think it's gold.",
            "The answer might be Shakespeare.",
            "I'm not sure, but possibly Paris?",
            "It could be the Beatles.",
            "I believe the answer is gold.",
        ]
        
        for response_text in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            assert parsed.original_text == response_text
            assert parsed.response_type == ResponseType.UNCERTAIN
            assert len(parsed.confidence_indicators) > 0
    
    def test_parse_refusal_response(self):
        """Test parsing refusal responses."""
        test_cases = [
            "I don't know the answer.",
            "Sorry, I can't help with this question.",
            "I don't have enough information to answer.",
            "Unable to determine the answer.",
            "I'm not sure about this one.",
        ]
        
        for response_text in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            assert parsed.original_text == response_text
            assert parsed.response_type == ResponseType.REFUSAL
            assert parsed.extracted_answer == ""
    
    def test_extract_jeopardy_answer(self):
        """Test extracting Jeopardy-format answers."""
        test_cases = [
            ("What is gold?", "What is gold?"),
            ("The answer is: What is gold?", "What is gold?"),
            ("I believe the answer is What is gold", "What is gold?"),  # Should add ?
            ("What is gold. This is correct.", "What is gold?"),
        ]
        
        for response_text, expected in test_cases:
            parsed = self.parser.parse_response(response_text)
            
            if "what is" in response_text.lower():
                assert parsed.response_type == ResponseType.JEOPARDY_FORMAT
                assert expected.lower() in parsed.extracted_answer.lower()
    
    def test_extract_direct_answer(self):
        """Test extracting direct answers."""
        test_cases = [
            ("Gold", "Gold"),
            ("The answer is gold.", "gold"),
            ("It is William Shakespeare.", "William Shakespeare"),
            ("This is Paris, France.", "Paris, France"),
        ]
        
        for response_text, expected in test_cases:
            config_direct = response_text
            if "answer is" in response_text.lower() or "it is" in response_text.lower():
                # Parser should extract the core answer
                parsed = self.parser.parse_response(response_text)
                # Should extract after "answer is" or "it is"
                assert expected.lower() in parsed.extracted_answer.lower()
    
    def test_clean_response(self):
        """Test response cleaning functionality."""
        test_cases = [
            ("  Answer:  Gold  ", "Gold"),  # Should remove prefix and whitespace
            ("My answer is: Paris", "Paris"),
            ("Final answer: What is gold?", "What is gold?"),
            ("Response: The Beatles", "The Beatles"),
        ]
        
        for response_text, expected_clean in test_cases:
            cleaned = self.parser._clean_response(response_text)
            assert expected_clean.lower() in cleaned.lower()
    
    def test_has_jeopardy_format(self):
        """Test Jeopardy format detection."""
        jeopardy_responses = [
            "What is gold?",
            "Who is Shakespeare?",
            "Where are the Alps?",
            "When was 1969?",
            "What were the Beatles?",
        ]
        
        non_jeopardy_responses = [
            "Gold",
            "Shakespeare wrote plays",
            "The Alps are in Europe",
            "1969 was the year",
        ]
        
        for response in jeopardy_responses:
            assert self.parser._has_jeopardy_format(response) == True
        
        for response in non_jeopardy_responses:
            assert self.parser._has_jeopardy_format(response) == False
    
    def test_extract_confidence_indicators(self):
        """Test extracting confidence indicators."""
        test_cases = [
            ("I think it's gold", ["i think"]),
            ("I believe the answer is gold", ["i believe"]),
            ("It might be gold, but I'm not sure", ["might be", "i'm not sure"]),
            ("Probably gold", ["probably"]),
            ("Gold", []),  # No indicators
        ]
        
        for response_text, expected_indicators in test_cases:
            indicators = self.parser._extract_confidence_indicators(response_text)
            
            for expected in expected_indicators:
                assert any(expected in indicator for indicator in indicators)
    
    def test_extract_reasoning(self):
        """Test extracting reasoning from responses."""
        test_cases = [
            ("Because gold has the symbol Au, the answer is gold.", True),
            ("Let me think step by step. First, I need to consider...", True),
            ("This is because Paris is the capital.", True),
            ("Gold", False),  # Too short, no reasoning
            ("What is gold?", False),  # Short, no reasoning
        ]
        
        for response_text, should_have_reasoning in test_cases:
            reasoning = self.parser._extract_reasoning(response_text)
            
            if should_have_reasoning:
                assert reasoning is not None
                assert len(reasoning) > 0
            else:
                assert reasoning is None
    
    def test_parse_batch_responses(self):
        """Test parsing multiple responses in batch."""
        responses = [
            "What is gold?",
            "I don't know",
            "Paris",
            "I think it's Shakespeare",
        ]
        
        parsed_list = self.parser.parse_batch_responses(responses)
        
        assert len(parsed_list) == 4
        assert all(isinstance(p, ParsedResponse) for p in parsed_list)
        
        # Check specific response types
        assert parsed_list[0].response_type == ResponseType.JEOPARDY_FORMAT
        assert parsed_list[1].response_type == ResponseType.REFUSAL
        assert parsed_list[2].response_type == ResponseType.DIRECT_ANSWER
        assert parsed_list[3].response_type == ResponseType.UNCERTAIN
    
    def test_extract_answers_only(self):
        """Test extracting just the answers from responses."""
        responses = [
            "What is gold?",
            "Paris",
            "I believe it's Shakespeare",
        ]
        
        answers = self.parser.extract_answers_only(responses)
        
        assert len(answers) == 3
        assert all(isinstance(answer, str) for answer in answers)
        assert "gold" in answers[0].lower()
        assert "Paris" in answers[1]
        assert "shakespeare" in answers[2].lower()
    
    def test_get_response_quality_score(self):
        """Test response quality scoring."""
        # High quality Jeopardy response
        jeopardy_response = ParsedResponse(
            original_text="What is gold?",
            extracted_answer="What is gold?",
            response_type=ResponseType.JEOPARDY_FORMAT,
            confidence_indicators=[]
        )
        score = self.parser.get_response_quality_score(jeopardy_response)
        assert score > 0.8  # Should be high quality
        
        # Refusal response
        refusal_response = ParsedResponse(
            original_text="I don't know",
            extracted_answer="",
            response_type=ResponseType.REFUSAL,
            confidence_indicators=[]
        )
        score = self.parser.get_response_quality_score(refusal_response)
        assert score == 0.0  # Should be zero for refusals
        
        # Uncertain response
        uncertain_response = ParsedResponse(
            original_text="I think it's gold",
            extracted_answer="gold",
            response_type=ResponseType.UNCERTAIN,
            confidence_indicators=["i think"]
        )
        score = self.parser.get_response_quality_score(uncertain_response)
        assert 0.0 < score < 1.0  # Should be penalized but not zero
        
        # Very short response
        short_response = ParsedResponse(
            original_text="Au",
            extracted_answer="Au",
            response_type=ResponseType.DIRECT_ANSWER,
            confidence_indicators=[]
        )
        score = self.parser.get_response_quality_score(short_response)
        assert score < 0.8  # Should be penalized for being too short


class TestParsedResponse:
    """Test cases for ParsedResponse class."""
    
    def test_parsed_response_creation(self):
        """Test creating a ParsedResponse."""
        response = ParsedResponse(
            original_text="What is gold?",
            extracted_answer="What is gold?",
            response_type=ResponseType.JEOPARDY_FORMAT,
            confidence_indicators=[],
            reasoning=None,
            metadata={"test": "value"}
        )
        
        assert response.original_text == "What is gold?"
        assert response.extracted_answer == "What is gold?"
        assert response.response_type == ResponseType.JEOPARDY_FORMAT
        assert response.confidence_indicators == []
        assert response.reasoning is None
        assert response.metadata == {"test": "value"}
    
    def test_parsed_response_default_metadata(self):
        """Test ParsedResponse with default metadata."""
        response = ParsedResponse(
            original_text="Gold",
            extracted_answer="Gold",
            response_type=ResponseType.DIRECT_ANSWER,
            confidence_indicators=[]
        )
        
        assert response.metadata == {}  # Should default to empty dict


class TestResponseType:
    """Test cases for ResponseType enum."""
    
    def test_response_type_values(self):
        """Test that all response type values are strings."""
        assert ResponseType.JEOPARDY_FORMAT == "jeopardy_format"
        assert ResponseType.DIRECT_ANSWER == "direct_answer"
        assert ResponseType.EXPLANATION == "explanation"
        assert ResponseType.REFUSAL == "refusal"
        assert ResponseType.ERROR == "error"
        assert ResponseType.UNCERTAIN == "uncertain"
    
    def test_response_type_membership(self):
        """Test response type enum membership."""
        assert "jeopardy_format" in ResponseType
        assert "direct_answer" in ResponseType
        assert "invalid_type" not in ResponseType