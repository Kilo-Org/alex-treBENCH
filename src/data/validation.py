"""
Data Validation Utilities

Validation functions for Jeopardy question data quality and format checking.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime

from ..core.exceptions import ValidationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Comprehensive data validation for Jeopardy questions."""
    
    # Valid Jeopardy categories (common ones - can be extended)
    VALID_CATEGORIES = {
        'SCIENCE', 'HISTORY', 'LITERATURE', 'GEOGRAPHY', 'SPORTS', 'MOVIES', 
        'MUSIC', 'TELEVISION', 'POLITICS', 'FOOD & DRINK', 'POTPOURRI',
        'BEFORE & AFTER', 'RHYME TIME', 'WORDPLAY', 'WORLD CAPITALS',
        'AMERICAN LITERATURE', 'WORLD HISTORY', 'THE MOVIES', 'SCIENCE & NATURE'
    }
    
    # Valid dollar values for different eras
    VALID_VALUES = {
        'classic': [100, 200, 300, 400, 500],  # Classic Jeopardy
        'modern': [200, 400, 600, 800, 1000],  # Modern Jeopardy
        'double': [400, 800, 1200, 1600, 2000], # Double Jeopardy
        'all': [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000]
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, apply stricter validation rules
        """
        self.strict_mode = strict_mode
    
    def validate_question_format(self, question: str) -> Tuple[bool, List[str]]:
        """
        Validate question format and content.
        
        Args:
            question: Question text to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not question or not isinstance(question, str):
            issues.append("Question is empty or not a string")
            return False, issues
        
        question = question.strip()
        
        # Check minimum length
        if len(question) < 10:
            issues.append(f"Question too short ({len(question)} chars, minimum 10)")
        
        # Check maximum length (reasonable limit)
        if len(question) > 500:
            issues.append(f"Question too long ({len(question)} chars, maximum 500)")
        
        # Check for proper sentence structure
        if not question.endswith(('.', '?', '!')):
            issues.append("Question should end with proper punctuation")
        
        # Check for HTML tags (should be cleaned)
        if re.search(r'<[^>]+>', question):
            issues.append("Question contains HTML tags")
        
        # Check for excessive whitespace
        if re.search(r'\s{3,}', question):
            issues.append("Question contains excessive whitespace")
        
        # In strict mode, check for common question patterns
        if self.strict_mode:
            # Questions should typically be statements, not direct questions
            if question.count('?') > 1:
                issues.append("Question contains multiple question marks")
            
            # Check for common Jeopardy patterns
            jeopardy_patterns = [
                r'\bthis\b', r'\bthese\b', r'\bit\b', r'\bhe\b', r'\bshe\b'
            ]
            if not any(re.search(pattern, question, re.IGNORECASE) for pattern in jeopardy_patterns):
                issues.append("Question may not follow typical Jeopardy format")
        
        return len(issues) == 0, issues
    
    def validate_answer_format(self, answer: str) -> Tuple[bool, List[str]]:
        """
        Validate answer format and content.
        
        Args:
            answer: Answer text to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not answer or not isinstance(answer, str):
            issues.append("Answer is empty or not a string")
            return False, issues
        
        answer = answer.strip()
        
        # Check minimum length
        if len(answer) < 1:
            issues.append("Answer is empty after stripping whitespace")
        
        # Check maximum length (reasonable limit)
        if len(answer) > 200:
            issues.append(f"Answer too long ({len(answer)} chars, maximum 200)")
        
        # Check for HTML tags
        if re.search(r'<[^>]+>', answer):
            issues.append("Answer contains HTML tags")
        
        # Check for excessive whitespace
        if re.search(r'\s{3,}', answer):
            issues.append("Answer contains excessive whitespace")
        
        # In strict mode, validate proper Jeopardy answer format
        if self.strict_mode:
            # Check for proper "What/Who is" format (if present)
            answer_lower = answer.lower()
            if any(phrase in answer_lower for phrase in ['what is', 'who is', 'where is', 'when is']):
                if not re.match(r'^(what|who|where|when|how) (is|are|was|were)', answer_lower):
                    issues.append("Answer format may be incorrect for Jeopardy style")
        
        return len(issues) == 0, issues
    
    def validate_category(self, category: str, allow_custom: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate category name and format.
        
        Args:
            category: Category name to validate
            allow_custom: Whether to allow categories not in the predefined list
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not category or not isinstance(category, str):
            issues.append("Category is empty or not a string")
            return False, issues
        
        category = category.strip()
        
        # Check length
        if len(category) < 2:
            issues.append(f"Category too short ({len(category)} chars, minimum 2)")
        
        if len(category) > 50:
            issues.append(f"Category too long ({len(category)} chars, maximum 50)")
        
        # Check for valid characters (letters, numbers, spaces, punctuation)
        if not re.match(r'^[A-Za-z0-9\s\&\-\.\!\?\'\"]+$', category):
            issues.append("Category contains invalid characters")
        
        # In strict mode, check against known categories
        if self.strict_mode and not allow_custom:
            if category.upper() not in self.VALID_CATEGORIES:
                issues.append(f"Category '{category}' not in predefined valid categories")
        
        return len(issues) == 0, issues
    
    def validate_dollar_value(self, value: Any, era: str = 'all') -> Tuple[bool, List[str]]:
        """
        Validate dollar value for Jeopardy questions.
        
        Args:
            value: Dollar value to validate
            era: Era to validate against ('classic', 'modern', 'double', 'all')
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Allow None values (some questions might not have values)
        if value is None or pd.isna(value):
            return True, []
        
        # Convert to int if possible
        try:
            value = int(value)
        except (ValueError, TypeError):
            issues.append(f"Value '{value}' cannot be converted to integer")
            return False, issues
        
        # Check for reasonable range
        if value <= 0:
            issues.append(f"Value must be positive (got {value})")
        
        if value > 10000:
            issues.append(f"Value unusually high ({value}), maximum expected is 10000")
        
        # Check against valid values for era
        valid_values = self.VALID_VALUES.get(era, self.VALID_VALUES['all'])
        
        if self.strict_mode and value not in valid_values:
            issues.append(f"Value {value} not in valid set for era '{era}': {valid_values}")
        
        return len(issues) == 0, issues
    
    def validate_date(self, date_value: Any) -> Tuple[bool, List[str]]:
        """
        Validate air date.
        
        Args:
            date_value: Date value to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Allow None values
        if date_value is None or pd.isna(date_value):
            return True, []
        
        # Try to parse the date
        try:
            if isinstance(date_value, str):
                parsed_date = pd.to_datetime(date_value)
            else:
                parsed_date = pd.to_datetime(date_value)
            
            # Check reasonable date range (Jeopardy started in 1964)
            min_date = datetime(1964, 1, 1)
            max_date = datetime.now()
            
            if parsed_date.date() < min_date.date():
                issues.append(f"Date {parsed_date.date()} is before Jeopardy started (1964)")
            
            if parsed_date.date() > max_date.date():
                issues.append(f"Date {parsed_date.date()} is in the future")
                
        except (ValueError, TypeError) as e:
            issues.append(f"Invalid date format: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_difficulty_level(self, difficulty: str) -> Tuple[bool, List[str]]:
        """
        Validate difficulty level.
        
        Args:
            difficulty: Difficulty level to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        valid_levels = ['Easy', 'Medium', 'Hard', 'Unknown']
        
        if difficulty is None or pd.isna(difficulty):
            return True, []
        
        if not isinstance(difficulty, str):
            issues.append("Difficulty must be a string")
            return False, issues
        
        if difficulty not in valid_levels:
            issues.append(f"Difficulty '{difficulty}' not in valid levels: {valid_levels}")
        
        return len(issues) == 0, issues
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate entire DataFrame of questions.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating DataFrame with {len(df)} questions")
        
        results = {
            'total_questions': len(df),
            'valid_questions': 0,
            'validation_errors': [],
            'field_validation': {
                'question': {'valid': 0, 'invalid': 0, 'issues': []},
                'answer': {'valid': 0, 'invalid': 0, 'issues': []},
                'category': {'valid': 0, 'invalid': 0, 'issues': []},
                'value': {'valid': 0, 'invalid': 0, 'issues': []},
                'air_date': {'valid': 0, 'invalid': 0, 'issues': []},
                'difficulty_level': {'valid': 0, 'invalid': 0, 'issues': []}
            }
        }
        
        for idx, row in df.iterrows():
            question_valid = True
            question_issues = []
            
            # Validate question
            if 'question' in df.columns:
                valid, issues = self.validate_question_format(row.get('question'))
                results['field_validation']['question']['valid'] += 1 if valid else 0
                results['field_validation']['question']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['question']['issues'].extend(issues)
                    question_valid = False
                    question_issues.extend([f"Question: {issue}" for issue in issues])
            
            # Validate answer
            if 'answer' in df.columns:
                valid, issues = self.validate_answer_format(row.get('answer'))
                results['field_validation']['answer']['valid'] += 1 if valid else 0
                results['field_validation']['answer']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['answer']['issues'].extend(issues)
                    question_valid = False
                    question_issues.extend([f"Answer: {issue}" for issue in issues])
            
            # Validate category
            if 'category' in df.columns:
                valid, issues = self.validate_category(row.get('category'))
                results['field_validation']['category']['valid'] += 1 if valid else 0
                results['field_validation']['category']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['category']['issues'].extend(issues)
                    if self.strict_mode:
                        question_valid = False
                        question_issues.extend([f"Category: {issue}" for issue in issues])
            
            # Validate value
            if 'value' in df.columns:
                valid, issues = self.validate_dollar_value(row.get('value'))
                results['field_validation']['value']['valid'] += 1 if valid else 0
                results['field_validation']['value']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['value']['issues'].extend(issues)
                    if self.strict_mode:
                        question_valid = False
                        question_issues.extend([f"Value: {issue}" for issue in issues])
            
            # Validate air_date
            if 'air_date' in df.columns:
                valid, issues = self.validate_date(row.get('air_date'))
                results['field_validation']['air_date']['valid'] += 1 if valid else 0
                results['field_validation']['air_date']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['air_date']['issues'].extend(issues)
            
            # Validate difficulty_level
            if 'difficulty_level' in df.columns:
                valid, issues = self.validate_difficulty_level(row.get('difficulty_level'))
                results['field_validation']['difficulty_level']['valid'] += 1 if valid else 0
                results['field_validation']['difficulty_level']['invalid'] += 0 if valid else 1
                if issues:
                    results['field_validation']['difficulty_level']['issues'].extend(issues)
            
            if question_valid:
                results['valid_questions'] += 1
            else:
                results['validation_errors'].append({
                    'row_index': idx,
                    'issues': question_issues
                })
        
        validation_rate = (results['valid_questions'] / results['total_questions']) * 100 if results['total_questions'] > 0 else 0
        logger.info(f"Validation complete: {results['valid_questions']}/{results['total_questions']} questions valid ({validation_rate:.1f}%)")
        
        return results


def validate_question_format(question: str) -> bool:
    """Quick validation function for question format."""
    validator = DataValidator(strict_mode=False)
    valid, _ = validator.validate_question_format(question)
    return valid


def validate_answer_format(answer: str) -> bool:
    """Quick validation function for answer format."""
    validator = DataValidator(strict_mode=False)
    valid, _ = validator.validate_answer_format(answer)
    return valid


def validate_category_name(category: str) -> bool:
    """Quick validation function for category name."""
    validator = DataValidator(strict_mode=False)
    valid, _ = validator.validate_category(category, allow_custom=True)
    return valid


def validate_dollar_value(value: Any) -> bool:
    """Quick validation function for dollar value."""
    validator = DataValidator(strict_mode=False)
    valid, _ = validator.validate_dollar_value(value)
    return valid