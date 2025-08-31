"""
Tests for Prompt Formatter
"""

import pytest
from src.models.prompt_formatter import (
    PromptFormatter, PromptTemplate, PromptConfig
)


class TestPromptFormatter:
    """Test cases for PromptFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = PromptFormatter()
        self.test_question = "This element has the chemical symbol 'Au'"
        self.test_category = "SCIENCE"
        self.test_value = "$600"
        self.test_difficulty = "Medium"
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        formatter = PromptFormatter()
        
        assert formatter.default_config.template == PromptTemplate.JEOPARDY_STYLE
        assert formatter.default_config.include_category == True
        assert formatter.default_config.include_value == True
        assert formatter.default_config.include_difficulty == False
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = PromptConfig(
            template=PromptTemplate.BASIC_QA,
            include_category=False,
            include_value=False,
            include_difficulty=True
        )
        formatter = PromptFormatter(config)
        
        assert formatter.default_config.template == PromptTemplate.BASIC_QA
        assert formatter.default_config.include_category == False
        assert formatter.default_config.include_value == False
        assert formatter.default_config.include_difficulty == True
    
    def test_format_basic_qa(self):
        """Test basic Q&A prompt formatting."""
        config = PromptConfig(template=PromptTemplate.BASIC_QA)
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            value=self.test_value,
            config=config
        )
        
        assert isinstance(prompt, str)
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert self.test_question in prompt
        assert self.test_category in prompt
        assert self.test_value in prompt
        assert "expert quiz contestant" in prompt.lower()
    
    def test_format_jeopardy_style(self):
        """Test Jeopardy-style prompt formatting."""
        config = PromptConfig(template=PromptTemplate.JEOPARDY_STYLE)
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            value=self.test_value,
            config=config
        )
        
        assert isinstance(prompt, str)
        assert "Clue:" in prompt
        assert "Response:" in prompt
        assert self.test_question in prompt
        assert self.test_category in prompt
        assert self.test_value in prompt
        assert "jeopardy" in prompt.lower()
        assert "form of a question" in prompt.lower()
    
    def test_format_chain_of_thought(self):
        """Test chain-of-thought prompt formatting."""
        config = PromptConfig(template=PromptTemplate.CHAIN_OF_THOUGHT)
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            config=config
        )
        
        assert isinstance(prompt, str)
        assert "step by step" in prompt.lower()
        assert "Analysis:" in prompt
        assert "Reasoning:" in prompt
        assert "Final answer:" in prompt
        assert self.test_question in prompt
    
    def test_format_few_shot(self):
        """Test few-shot prompt formatting."""
        config = PromptConfig(template=PromptTemplate.FEW_SHOT)
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            config=config
        )
        
        assert isinstance(prompt, str)
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Now answer this new clue:" in prompt
        assert self.test_question in prompt
        assert "What is Buenos Aires?" in prompt  # From default examples
    
    def test_format_instructional(self):
        """Test instructional prompt formatting."""
        config = PromptConfig(template=PromptTemplate.INSTRUCTIONAL)
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            config=config
        )
        
        assert isinstance(prompt, str)
        assert "INSTRUCTIONS:" in prompt
        assert "CLUE:" in prompt
        assert "YOUR RESPONSE:" in prompt
        assert "Read the clue carefully" in prompt
        assert self.test_question in prompt
    
    def test_include_exclude_metadata(self):
        """Test including/excluding metadata in prompts."""
        # Test with all metadata included
        config = PromptConfig(
            template=PromptTemplate.JEOPARDY_STYLE,
            include_category=True,
            include_value=True,
            include_difficulty=True
        )
        
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            value=self.test_value,
            difficulty=self.test_difficulty,
            config=config
        )
        
        assert self.test_category in prompt
        assert self.test_value in prompt
        assert self.test_difficulty in prompt
        
        # Test with all metadata excluded
        config = PromptConfig(
            template=PromptTemplate.JEOPARDY_STYLE,
            include_category=False,
            include_value=False,
            include_difficulty=False
        )
        
        prompt = self.formatter.format_prompt(
            self.test_question,
            category=self.test_category,
            value=self.test_value,
            difficulty=self.test_difficulty,
            config=config
        )
        
        assert self.test_category not in prompt
        assert self.test_value not in prompt
        assert self.test_difficulty not in prompt
    
    def test_custom_system_prompt(self):
        """Test using custom system prompt."""
        custom_system = "You are a trivia expert specializing in science."
        config = PromptConfig(
            template=PromptTemplate.BASIC_QA,
            system_prompt=custom_system
        )
        
        prompt = self.formatter.format_prompt(
            self.test_question,
            config=config
        )
        
        assert custom_system in prompt
    
    def test_custom_few_shot_examples(self):
        """Test using custom few-shot examples."""
        custom_examples = [
            {
                "category": "CUSTOM",
                "value": "$100",
                "question": "Test question",
                "correct_answer": "What is test answer?"
            }
        ]
        
        config = PromptConfig(
            template=PromptTemplate.FEW_SHOT,
            few_shot_examples=custom_examples
        )
        
        prompt = self.formatter.format_prompt(
            self.test_question,
            config=config
        )
        
        assert "Test question" in prompt
        assert "What is test answer?" in prompt
        assert "CUSTOM" in prompt
    
    def test_invalid_template_raises_error(self):
        """Test that invalid template raises ValueError."""
        # Create a config with invalid template by bypassing enum validation
        config = PromptConfig(template="invalid_template")
        
        with pytest.raises(ValueError, match="Unknown prompt template"):
            self.formatter.format_prompt(self.test_question, config=config)
    
    def test_get_available_templates(self):
        """Test getting available templates."""
        templates = self.formatter.get_available_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert PromptTemplate.BASIC_QA in templates
        assert PromptTemplate.JEOPARDY_STYLE in templates
        assert PromptTemplate.CHAIN_OF_THOUGHT in templates
        assert PromptTemplate.FEW_SHOT in templates
        assert PromptTemplate.INSTRUCTIONAL in templates
    
    def test_create_batch_prompts(self):
        """Test creating batch prompts."""
        questions = [
            {
                "question": "Question 1",
                "category": "CATEGORY1",
                "value": "$100",
                "difficulty": "Easy"
            },
            {
                "question": "Question 2",
                "category": "CATEGORY2",
                "value": "$200",
                "difficulty": "Medium"
            }
        ]
        
        prompts = self.formatter.create_batch_prompts(questions)
        
        assert isinstance(prompts, list)
        assert len(prompts) == 2
        
        # Check that each question appears in its corresponding prompt
        assert "Question 1" in prompts[0]
        assert "Question 2" in prompts[1]
        assert "CATEGORY1" in prompts[0]
        assert "CATEGORY2" in prompts[1]
    
    def test_estimate_prompt_tokens(self):
        """Test token estimation."""
        prompt = self.formatter.format_prompt(self.test_question)
        tokens = self.formatter.estimate_prompt_tokens(prompt)
        
        assert isinstance(tokens, int)
        assert tokens > 0
        
        # Rough check - should be approximately len(prompt) / 4
        expected_range = len(prompt) // 4
        assert tokens >= expected_range // 2  # Allow some variation
        assert tokens <= expected_range * 2


class TestPromptConfig:
    """Test cases for PromptConfig class."""
    
    def test_prompt_config_creation(self):
        """Test creating a PromptConfig."""
        config = PromptConfig(
            template=PromptTemplate.BASIC_QA,
            include_category=False,
            include_value=True,
            include_difficulty=False,
            system_prompt="Custom system prompt",
            max_length=500
        )
        
        assert config.template == PromptTemplate.BASIC_QA
        assert config.include_category == False
        assert config.include_value == True
        assert config.include_difficulty == False
        assert config.system_prompt == "Custom system prompt"
        assert config.max_length == 500
        assert config.few_shot_examples is None  # Default
    
    def test_prompt_config_defaults(self):
        """Test PromptConfig defaults."""
        config = PromptConfig(template=PromptTemplate.JEOPARDY_STYLE)
        
        assert config.template == PromptTemplate.JEOPARDY_STYLE
        assert config.include_category == True  # Default
        assert config.include_value == True    # Default
        assert config.include_difficulty == False  # Default
        assert config.system_prompt is None   # Default
        assert config.few_shot_examples is None  # Default
        assert config.max_length is None      # Default


class TestPromptTemplate:
    """Test cases for PromptTemplate enum."""
    
    def test_template_values(self):
        """Test that all template values are strings."""
        assert PromptTemplate.BASIC_QA == "basic_qa"
        assert PromptTemplate.JEOPARDY_STYLE == "jeopardy_style"
        assert PromptTemplate.CHAIN_OF_THOUGHT == "chain_of_thought"
        assert PromptTemplate.FEW_SHOT == "few_shot"
        assert PromptTemplate.INSTRUCTIONAL == "instructional"
    
    def test_template_membership(self):
        """Test template enum membership."""
        assert "basic_qa" in PromptTemplate
        assert "jeopardy_style" in PromptTemplate
        assert "invalid_template" not in PromptTemplate