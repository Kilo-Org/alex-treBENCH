"""
Prompt Formatter

Formats Jeopardy questions into prompts optimized for different models and strategies.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class PromptTemplate(str, Enum):
    """Available prompt templates."""
    BASIC_QA = "basic_qa"
    JEOPARDY_STYLE = "jeopardy_style"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    INSTRUCTIONAL = "instructional"


@dataclass
class PromptConfig:
    """Configuration for prompt formatting."""
    template: PromptTemplate
    include_category: bool = True
    include_value: bool = True
    include_difficulty: bool = False
    system_prompt: Optional[str] = None
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    max_length: Optional[int] = None


class PromptFormatter:
    """Formats Jeopardy questions into optimized prompts for language models."""
    
    # Default few-shot examples for Jeopardy
    DEFAULT_FEW_SHOT_EXAMPLES = [
        {
            "category": "WORLD CAPITALS",
            "value": "$400",
            "question": "This South American capital is known as the 'Paris of South America'",
            "correct_answer": "What is Buenos Aires?"
        },
        {
            "category": "SCIENCE",
            "value": "$600",
            "question": "This element has the chemical symbol 'Au'",
            "correct_answer": "What is gold?"
        },
        {
            "category": "LITERATURE",
            "value": "$800",
            "question": "This author wrote 'Pride and Prejudice'",
            "correct_answer": "Who is Jane Austen?"
        }
    ]
    
    # System prompts for different approaches
    SYSTEM_PROMPTS = {
        PromptTemplate.BASIC_QA: (
            "You are an expert quiz contestant. Answer the question accurately and concisely."
        ),
        PromptTemplate.JEOPARDY_STYLE: (
            "You are a Jeopardy! contestant. Respond to each clue in the form of a question, "
            "starting with 'What is', 'Who is', 'Where is', etc. as appropriate."
        ),
        PromptTemplate.CHAIN_OF_THOUGHT: (
            "You are a Jeopardy! contestant. Think through the clue step by step, "
            "then provide your final answer in the form of a question."
        ),
        PromptTemplate.FEW_SHOT: (
            "You are a Jeopardy! contestant. Study the examples below, then answer "
            "the new question in the same format."
        ),
        PromptTemplate.INSTRUCTIONAL: (
            "You are participating in Jeopardy!, a quiz show where contestants are given "
            "answers and must respond with the corresponding question. Your response must "
            "be in the form of a question (e.g., 'What is...?', 'Who is...?', 'Where is...?')."
        )
    }
    
    def __init__(self, default_config: Optional[PromptConfig] = None):
        """Initialize the prompt formatter with default configuration."""
        self.default_config = default_config or PromptConfig(
            template=PromptTemplate.JEOPARDY_STYLE,
            include_category=True,
            include_value=True
        )
    
    def format_prompt(
        self, 
        question: str,
        category: Optional[str] = None,
        value: Optional[str] = None,
        difficulty: Optional[str] = None,
        config: Optional[PromptConfig] = None
    ) -> str:
        """
        Format a Jeopardy question into a prompt.
        
        Args:
            question: The Jeopardy clue/question
            category: Category of the question
            value: Dollar value of the question
            difficulty: Difficulty level
            config: Prompt configuration (uses default if None)
            
        Returns:
            Formatted prompt string
        """
        config = config or self.default_config
        
        if config.template == PromptTemplate.BASIC_QA:
            return self._format_basic_qa(question, category, value, difficulty, config)
        elif config.template == PromptTemplate.JEOPARDY_STYLE:
            return self._format_jeopardy_style(question, category, value, difficulty, config)
        elif config.template == PromptTemplate.CHAIN_OF_THOUGHT:
            return self._format_chain_of_thought(question, category, value, difficulty, config)
        elif config.template == PromptTemplate.FEW_SHOT:
            return self._format_few_shot(question, category, value, difficulty, config)
        elif config.template == PromptTemplate.INSTRUCTIONAL:
            return self._format_instructional(question, category, value, difficulty, config)
        else:
            raise ValueError(f"Unknown prompt template: {config.template}")
    
    def _format_basic_qa(
        self, question: str, category: Optional[str], value: Optional[str], 
        difficulty: Optional[str], config: PromptConfig
    ) -> str:
        """Format as basic Q&A."""
        parts = []
        
        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append(self.SYSTEM_PROMPTS[PromptTemplate.BASIC_QA])
        
        parts.append("")  # Empty line
        
        # Add context if requested
        if config.include_category and category:
            parts.append(f"Category: {category}")
        if config.include_value and value:
            parts.append(f"Value: {value}")
        if config.include_difficulty and difficulty:
            parts.append(f"Difficulty: {difficulty}")
        
        if any([config.include_category and category, 
                config.include_value and value, 
                config.include_difficulty and difficulty]):
            parts.append("")  # Empty line after context
        
        parts.append(f"Question: {question}")
        parts.append("")
        parts.append("Answer:")
        
        return "\n".join(parts)
    
    def _format_jeopardy_style(
        self, question: str, category: Optional[str], value: Optional[str], 
        difficulty: Optional[str], config: PromptConfig
    ) -> str:
        """Format in Jeopardy style (answer in form of a question)."""
        parts = []
        
        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append(self.SYSTEM_PROMPTS[PromptTemplate.JEOPARDY_STYLE])
        
        parts.append("")  # Empty line
        
        # Add context if requested
        context_parts = []
        if config.include_category and category:
            context_parts.append(f"Category: {category}")
        if config.include_value and value:
            context_parts.append(f"Value: {value}")
        if config.include_difficulty and difficulty:
            context_parts.append(f"Difficulty: {difficulty}")
        
        if context_parts:
            parts.extend(context_parts)
            parts.append("")  # Empty line after context
        
        parts.append(f"Clue: {question}")
        parts.append("")
        parts.append("Response:")
        
        return "\n".join(parts)
    
    def _format_chain_of_thought(
        self, question: str, category: Optional[str], value: Optional[str], 
        difficulty: Optional[str], config: PromptConfig
    ) -> str:
        """Format with chain-of-thought reasoning."""
        parts = []
        
        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append(self.SYSTEM_PROMPTS[PromptTemplate.CHAIN_OF_THOUGHT])
        
        parts.append("")  # Empty line
        
        # Add context if requested
        if config.include_category and category:
            parts.append(f"Category: {category}")
        if config.include_value and value:
            parts.append(f"Value: {value}")
        if config.include_difficulty and difficulty:
            parts.append(f"Difficulty: {difficulty}")
        
        if any([config.include_category and category, 
                config.include_value and value, 
                config.include_difficulty and difficulty]):
            parts.append("")  # Empty line after context
        
        parts.append(f"Clue: {question}")
        parts.append("")
        parts.append("Let me think through this step by step:")
        parts.append("1. Analysis:")
        parts.append("2. Key information:")
        parts.append("3. Reasoning:")
        parts.append("4. Final answer:")
        
        return "\n".join(parts)
    
    def _format_few_shot(
        self, question: str, category: Optional[str], value: Optional[str], 
        difficulty: Optional[str], config: PromptConfig
    ) -> str:
        """Format with few-shot examples."""
        parts = []
        
        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append(self.SYSTEM_PROMPTS[PromptTemplate.FEW_SHOT])
        
        parts.append("")  # Empty line
        
        # Add examples
        examples = config.few_shot_examples or self.DEFAULT_FEW_SHOT_EXAMPLES
        for i, example in enumerate(examples, 1):
            parts.append(f"Example {i}:")
            if config.include_category:
                parts.append(f"Category: {example['category']}")
            if config.include_value:
                parts.append(f"Value: {example['value']}")
            parts.append(f"Clue: {example['question']}")
            parts.append(f"Response: {example['correct_answer']}")
            parts.append("")  # Empty line after each example
        
        # Add the actual question
        parts.append("Now answer this new clue:")
        if config.include_category and category:
            parts.append(f"Category: {category}")
        if config.include_value and value:
            parts.append(f"Value: {value}")
        if config.include_difficulty and difficulty:
            parts.append(f"Difficulty: {difficulty}")
        
        parts.append(f"Clue: {question}")
        parts.append("Response:")
        
        return "\n".join(parts)
    
    def _format_instructional(
        self, question: str, category: Optional[str], value: Optional[str], 
        difficulty: Optional[str], config: PromptConfig
    ) -> str:
        """Format with detailed instructions."""
        parts = []
        
        if config.system_prompt:
            parts.append(config.system_prompt)
        else:
            parts.append(self.SYSTEM_PROMPTS[PromptTemplate.INSTRUCTIONAL])
        
        parts.append("")
        parts.append("INSTRUCTIONS:")
        parts.append("- Read the clue carefully")
        parts.append("- Identify what type of answer is expected")
        parts.append("- Respond in question format (What is...? Who is...? etc.)")
        parts.append("- Be specific and accurate")
        parts.append("- If unsure, make your best educated guess")
        parts.append("")
        
        # Add context if requested
        if config.include_category and category:
            parts.append(f"CATEGORY: {category}")
        if config.include_value and value:
            parts.append(f"VALUE: {value}")
        if config.include_difficulty and difficulty:
            parts.append(f"DIFFICULTY: {difficulty}")
        
        if any([config.include_category and category, 
                config.include_value and value, 
                config.include_difficulty and difficulty]):
            parts.append("")  # Empty line after context
        
        parts.append(f"CLUE: {question}")
        parts.append("")
        parts.append("YOUR RESPONSE:")
        
        return "\n".join(parts)
    
    def get_available_templates(self) -> List[PromptTemplate]:
        """Get list of available prompt templates."""
        return list(PromptTemplate)
    
    def create_batch_prompts(
        self, 
        questions: List[Dict[str, Any]], 
        config: Optional[PromptConfig] = None
    ) -> List[str]:
        """
        Create prompts for a batch of questions.
        
        Args:
            questions: List of question dictionaries with keys: 
                      question, category, value, difficulty
            config: Prompt configuration
            
        Returns:
            List of formatted prompt strings
        """
        prompts = []
        for q in questions:
            prompt = self.format_prompt(
                question=q.get('question', ''),
                category=q.get('category'),
                value=q.get('value'),
                difficulty=q.get('difficulty'),
                config=config
            )
            prompts.append(prompt)
        return prompts
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """
        Rough estimation of token count for a prompt.
        Uses approximate 4 characters per token rule.
        
        Args:
            prompt: The prompt string
            
        Returns:
            Estimated token count
        """
        return len(prompt) // 4