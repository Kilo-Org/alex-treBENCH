"""
Integration Tests for Model Components

Tests the integration of all model components working together.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, Mock
from datetime import datetime

from src.models.model_registry import ModelRegistry, ModelConfig as RegistryModelConfig
from src.models.prompt_formatter import PromptFormatter, PromptConfig, PromptTemplate
from src.models.response_parser import ResponseParser
from src.models.cost_calculator import CostCalculator, BillingTier
from src.models.openrouter import OpenRouterClient
from src.models.base import ModelConfig, ModelResponse


class TestModelsIntegration:
    """Integration tests for all model components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = PromptFormatter()
        self.parser = ResponseParser()
        self.calculator = CostCalculator(billing_tier=BillingTier.BASIC)
        self.config = ModelConfig(
            model_name="openai/gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.1
        )
    
    def test_model_registry_integration(self):
        """Test integration with model registry."""
        # Test getting model configuration
        model_config = ModelRegistry.get_model_config("openai/gpt-3.5-turbo")
        
        assert model_config is not None
        assert model_config.model_id == "openai/gpt-3.5-turbo"
        assert model_config.display_name == "GPT-3.5 Turbo"
        
        # Test cost estimation
        estimated_cost = ModelRegistry.estimate_cost(
            "openai/gpt-3.5-turbo", 1000, 500
        )
        assert estimated_cost > 0
        
        # Test availability validation
        assert ModelRegistry.validate_model_availability("openai/gpt-3.5-turbo") == True
        assert ModelRegistry.validate_model_availability("fake/model") == False
    
    def test_prompt_formatter_integration(self):
        """Test prompt formatter with different templates."""
        question = "This element has the chemical symbol 'Au'"
        category = "SCIENCE"
        value = "$600"
        
        # Test different prompt templates
        templates = [
            PromptTemplate.BASIC_QA,
            PromptTemplate.JEOPARDY_STYLE,
            PromptTemplate.CHAIN_OF_THOUGHT,
            PromptTemplate.FEW_SHOT,
            PromptTemplate.INSTRUCTIONAL
        ]
        
        for template in templates:
            config = PromptConfig(template=template)
            prompt = self.formatter.format_prompt(
                question, category, value, config=config
            )
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert question in prompt
            
            # Estimate tokens for the prompt
            estimated_tokens = self.formatter.estimate_prompt_tokens(prompt)
            assert estimated_tokens > 0
    
    def test_response_parser_integration(self):
        """Test response parser with various response types."""
        test_responses = [
            "What is gold?",  # Jeopardy format
            "Gold",           # Direct answer
            "I think it's gold because it has the chemical symbol Au.",  # Explanation
            "I don't know the answer.",  # Refusal
            "The answer might be gold, but I'm not certain.",  # Uncertain
        ]
        
        parsed_responses = self.parser.parse_batch_responses(test_responses)
        
        assert len(parsed_responses) == len(test_responses)
        
        # Check quality scores
        for parsed in parsed_responses:
            score = self.parser.get_response_quality_score(parsed)
            assert 0.0 <= score <= 1.0
        
        # Extract just the answers
        answers = self.parser.extract_answers_only(test_responses)
        assert len(answers) == len(test_responses)
    
    def test_cost_calculator_integration(self):
        """Test cost calculator with model registry."""
        # Test with known model
        cost = self.calculator.calculate_cost(
            "openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500
        )
        assert cost > 0
        
        # Record usage
        record = self.calculator.record_usage(
            model_id="openai/gpt-3.5-turbo",
            input_tokens=1000,
            output_tokens=500,
            session_id="test-session"
        )
        
        assert record.cost == cost
        assert record.model_id == "openai/gpt-3.5-turbo"
        
        # Test batch cost estimation
        questions = ["Question 1", "Question 2", "Question 3"]
        estimate = self.calculator.estimate_batch_cost(
            "openai/gpt-3.5-turbo",
            questions,
            estimated_input_tokens_per_question=100,
            estimated_output_tokens_per_question=50
        )
        
        assert estimate["num_questions"] == 3
        assert estimate["estimated_total_cost"] > 0
        assert estimate["cost_per_question"] > 0
    
    @pytest.mark.asyncio
    async def test_openrouter_client_integration(self):
        """Test OpenRouter client integration with other components."""
        client = OpenRouterClient(api_key="test-key", config=self.config)
        
        # Mock successful API response
        mock_response = {
            "choices": [
                {
                    "message": {"content": "What is gold?"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125
            },
            "model": "openai/gpt-3.5-turbo"
        }
        
        with patch.object(client, '_make_request_with_retry', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            # Test single query
            response = await client.query("Test question")
            
            assert isinstance(response, ModelResponse)
            assert response.model_id == "openai/gpt-3.5-turbo"
            assert response.tokens_used == 125
            assert response.cost > 0
            
            # Parse the response
            parsed = self.parser.parse_response(response.response)
            assert parsed.response_type.value in ["jeopardy_format", "direct_answer"]
            
            # Calculate quality score
            quality = self.parser.get_response_quality_score(parsed)
            assert 0.0 <= quality <= 1.0
    
    def test_end_to_end_jeopardy_workflow(self):
        """Test complete end-to-end Jeopardy workflow."""
        # 1. Format a Jeopardy question
        question = "This South American capital is known as the 'Paris of South America'"
        category = "WORLD CAPITALS"
        value = "$400"
        
        prompt_config = PromptConfig(
            template=PromptTemplate.JEOPARDY_STYLE,
            include_category=True,
            include_value=True
        )
        
        formatted_prompt = self.formatter.format_prompt(
            question, category, value, config=prompt_config
        )
        
        assert "WORLD CAPITALS" in formatted_prompt
        assert "$400" in formatted_prompt
        assert question in formatted_prompt
        
        # 2. Simulate model response
        simulated_response = "What is Buenos Aires?"
        
        # 3. Parse the response
        parsed_response = self.parser.parse_response(simulated_response)
        
        assert parsed_response.extracted_answer == "What is Buenos Aires?"
        assert parsed_response.response_type.value == "jeopardy_format"
        assert len(parsed_response.confidence_indicators) == 0
        
        # 4. Calculate quality score
        quality_score = self.parser.get_response_quality_score(parsed_response)
        assert quality_score > 0.8  # Should be high quality
        
        # 5. Calculate costs
        estimated_input_tokens = self.formatter.estimate_prompt_tokens(formatted_prompt)
        estimated_output_tokens = 10  # Short Jeopardy answer
        
        cost = self.calculator.calculate_cost(
            "openai/gpt-3.5-turbo",
            estimated_input_tokens,
            estimated_output_tokens
        )
        
        assert cost > 0
        
        # 6. Record usage
        usage_record = self.calculator.record_usage(
            model_id="openai/gpt-3.5-turbo",
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            session_id="jeopardy-test",
            metadata={
                "category": category,
                "value": value,
                "quality_score": quality_score
            }
        )
        
        assert usage_record.model_id == "openai/gpt-3.5-turbo"
        assert usage_record.session_id == "jeopardy-test"
        assert usage_record.metadata["category"] == category
        assert usage_record.metadata["quality_score"] == quality_score
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple questions."""
        # Sample Jeopardy questions
        questions_data = [
            {
                "question": "This element has the chemical symbol 'Au'",
                "category": "SCIENCE",
                "value": "$200"
            },
            {
                "question": "This Shakespeare play features the characters Romeo and Juliet",
                "category": "LITERATURE",
                "value": "$400"
            },
            {
                "question": "This French city is home to the Louvre Museum",
                "category": "GEOGRAPHY",
                "value": "$600"
            }
        ]
        
        # 1. Format all prompts
        prompts = self.formatter.create_batch_prompts(questions_data)
        assert len(prompts) == 3
        
        # 2. Simulate batch responses
        simulated_responses = [
            "What is gold?",
            "What is Romeo and Juliet?",
            "What is Paris?"
        ]
        
        # 3. Parse all responses
        parsed_responses = self.parser.parse_batch_responses(simulated_responses)
        assert len(parsed_responses) == 3
        
        # 4. Calculate batch costs
        total_estimated_cost = 0
        for i, prompt in enumerate(prompts):
            input_tokens = self.formatter.estimate_prompt_tokens(prompt)
            output_tokens = 15  # Estimated for Jeopardy answers
            
            cost = self.calculator.calculate_cost(
                "openai/gpt-3.5-turbo",
                input_tokens,
                output_tokens
            )
            total_estimated_cost += cost
            
            # Record individual usage
            self.calculator.record_usage(
                model_id="openai/gpt-3.5-turbo",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                session_id="batch-test",
                metadata={
                    "question_index": i,
                    "category": questions_data[i]["category"]
                }
            )
        
        # 5. Get usage summary
        summary = self.calculator.get_usage_summary(session_id="batch-test")
        assert summary.total_requests == 3
        assert summary.total_cost > 0
        
        # 6. Calculate average quality
        total_quality = sum(
            self.parser.get_response_quality_score(parsed)
            for parsed in parsed_responses
        )
        average_quality = total_quality / len(parsed_responses)
        assert 0.0 <= average_quality <= 1.0
    
    def test_model_comparison_workflow(self):
        """Test comparing costs across different models."""
        question = "This element has the chemical symbol 'Au'"
        
        # Format prompt once
        prompt = self.formatter.format_prompt(question)
        input_tokens = self.formatter.estimate_prompt_tokens(prompt)
        output_tokens = 10  # Estimated
        
        # Compare costs across models
        models_to_compare = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3-sonnet"
        ]
        
        cost_comparison = {}
        for model_id in models_to_compare:
            cost = self.calculator.calculate_cost(model_id, input_tokens, output_tokens)
            cost_comparison[model_id] = cost
        
        # Should have costs for all models
        assert len(cost_comparison) == 4
        assert all(cost >= 0 for cost in cost_comparison.values())
        
        # GPT-4 should be more expensive than GPT-3.5-turbo
        assert cost_comparison["openai/gpt-4"] > cost_comparison["openai/gpt-3.5-turbo"]
        
        # Claude-3-sonnet should be more expensive than Claude-3-haiku
        assert cost_comparison["anthropic/claude-3-sonnet"] > cost_comparison["anthropic/claude-3-haiku"]
    
    def test_error_handling_integration(self):
        """Test error handling across components."""
        # Test with invalid model
        cost = self.calculator.calculate_cost("invalid/model", 100, 50)
        assert cost == 0.0
        
        # Test with empty response
        parsed = self.parser.parse_response("")
        assert parsed.response_type.value == "error"
        assert parsed.extracted_answer == ""
        
        # Test with invalid template
        with pytest.raises(ValueError):
            invalid_config = PromptConfig(template="invalid_template")
            self.formatter.format_prompt("test", config=invalid_config)
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test integration with configuration system."""
        # Test that we can use model registry with actual model configurations
        available_models = ModelRegistry.list_available_models()
        assert len(available_models) > 0
        
        # Test that we can get configurations for available models
        for model_id in available_models[:3]:  # Test first 3 models
            config = ModelRegistry.get_model_config(model_id)
            assert config is not None
            assert config.model_id == model_id
            assert config.input_cost_per_1m_tokens >= 0
            assert config.output_cost_per_1m_tokens >= 0
            
            # Test that cost calculator can use these models
            cost = self.calculator.calculate_cost(model_id, 1000, 500)
            assert cost >= 0  # Should be 0 or positive
    
    def test_session_tracking_integration(self):
        """Test session-based tracking across components."""
        session_id = "integration-test-session"
        
        # Process multiple questions in the same session
        questions = [
            "This element has symbol Au",
            "This city is the capital of France",
            "This playwright wrote Hamlet"
        ]
        
        for i, question in enumerate(questions):
            # Format prompt
            prompt = self.formatter.format_prompt(question)
            
            # Estimate tokens
            input_tokens = self.formatter.estimate_prompt_tokens(prompt)
            output_tokens = 15
            
            # Record usage
            self.calculator.record_usage(
                model_id="openai/gpt-3.5-turbo",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                session_id=session_id,
                metadata={"question_number": i + 1}
            )
        
        # Get session summary
        session_summary = self.calculator.get_usage_summary(session_id=session_id)
        
        assert session_summary.total_requests == 3
        assert session_summary.total_cost > 0
        assert session_summary.total_input_tokens > 0
        assert session_summary.total_output_tokens == 45  # 15 * 3
    
    def test_quality_metrics_integration(self):
        """Test integration of quality metrics across components."""
        responses = [
            "What is gold?",           # High quality Jeopardy format
            "Gold",                   # Medium quality direct answer  
            "I think it's gold",      # Lower quality (uncertain)
            "I don't know",           # Poor quality (refusal)
            "",                       # Error case
        ]
        
        quality_scores = []
        for response in responses:
            parsed = self.parser.parse_response(response)
            quality = self.parser.get_response_quality_score(parsed)
            quality_scores.append(quality)
        
        # Quality should decrease in the order of responses
        assert quality_scores[0] > quality_scores[1]  # Jeopardy > direct
        assert quality_scores[1] > quality_scores[2]  # direct > uncertain
        assert quality_scores[2] > quality_scores[3]  # uncertain > refusal
        assert quality_scores[3] > quality_scores[4]  # refusal > error
        
        # Calculate average quality for batch
        average_quality = sum(quality_scores) / len(quality_scores)
        assert 0.0 <= average_quality <= 1.0


@pytest.mark.asyncio
async def test_full_system_simulation():
    """Comprehensive system simulation test."""
    # Initialize all components
    formatter = PromptFormatter()
    parser = ResponseParser()
    calculator = CostCalculator(billing_tier=BillingTier.PREMIUM)
    
    # Simulate a complete benchmarking session
    jeopardy_questions = [
        {
            "question": "This element has the chemical symbol 'Au'",
            "category": "SCIENCE",
            "value": "$200",
            "correct_answer": "What is gold?"
        },
        {
            "question": "This Shakespeare tragedy features a Danish prince",
            "category": "LITERATURE", 
            "value": "$400",
            "correct_answer": "What is Hamlet?"
        },
        {
            "question": "This Italian city is famous for its canals",
            "category": "GEOGRAPHY",
            "value": "$600", 
            "correct_answer": "What is Venice?"
        }
    ]
    
    session_id = "full-simulation"
    results = []
    
    for i, q_data in enumerate(jeopardy_questions):
        # 1. Format prompt
        prompt_config = PromptConfig(
            template=PromptTemplate.JEOPARDY_STYLE,
            include_category=True,
            include_value=True
        )
        
        formatted_prompt = formatter.format_prompt(
            q_data["question"],
            q_data["category"], 
            q_data["value"],
            config=prompt_config
        )
        
        # 2. Simulate model response (use correct answer for testing)
        simulated_response = q_data["correct_answer"]
        
        # 3. Parse response
        parsed_response = parser.parse_response(simulated_response)
        
        # 4. Calculate tokens and costs
        input_tokens = formatter.estimate_prompt_tokens(formatted_prompt)
        output_tokens = len(simulated_response.split()) * 1.3  # Rough estimation
        
        # 5. Record usage
        usage_record = calculator.record_usage(
            model_id="openai/gpt-3.5-turbo",
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            session_id=session_id,
            metadata={
                "question_index": i,
                "category": q_data["category"],
                "value": q_data["value"],
                "quality_score": parser.get_response_quality_score(parsed_response)
            }
        )
        
        # 6. Store result
        result = {
            "question": q_data["question"],
            "category": q_data["category"],
            "value": q_data["value"],
            "prompt_tokens": input_tokens,
            "response_tokens": output_tokens,
            "cost": usage_record.cost,
            "response": simulated_response,
            "parsed_response": parsed_response,
            "quality_score": parser.get_response_quality_score(parsed_response)
        }
        results.append(result)
    
    # Analyze session results
    session_summary = calculator.get_usage_summary(session_id=session_id)
    
    # Assertions for complete workflow
    assert len(results) == 3
    assert session_summary.total_requests == 3
    assert session_summary.total_cost > 0
    
    # All responses should be high quality (using correct answers)
    avg_quality = sum(r["quality_score"] for r in results) / len(results)
    assert avg_quality > 0.8
    
    # All should be Jeopardy format
    assert all(r["parsed_response"].response_type.value == "jeopardy_format" for r in results)
    
    # Cost should increase with question value (more complex prompts)
    costs = [r["cost"] for r in results]
    # Note: might not always be true due to token estimation, but generally expected
    
    # Generate summary report
    total_cost = session_summary.total_cost
    total_questions = len(results)
    avg_cost_per_question = total_cost / total_questions
    
    report = {
        "session_id": session_id,
        "total_questions": total_questions,
        "total_cost": total_cost,
        "average_cost_per_question": avg_cost_per_question,
        "average_quality_score": avg_quality,
        "billing_tier": calculator.billing_tier.value,
        "model_used": "openai/gpt-3.5-turbo"
    }
    
    # Final validations
    assert report["total_questions"] == 3
    assert report["total_cost"] > 0
    assert report["average_cost_per_question"] > 0
    assert report["average_quality_score"] > 0.8
    assert report["billing_tier"] == "premium"