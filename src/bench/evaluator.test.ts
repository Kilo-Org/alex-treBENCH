import { describe, it, expect } from 'bun:test';
import { JeopardyEvaluator } from './evaluator.js';

describe('JeopardyEvaluator', () => {
  const evaluator = new JeopardyEvaluator();

  describe('exact matching', () => {
    it('should match exact answers', () => {
      const result = evaluator.evaluateAnswer('Mars', 'Mars');
      expect(result.correct).toBe(true);
      expect(result.method).toBe('exact');
    });

    it('should handle case differences', () => {
      const result = evaluator.evaluateAnswer('Mars', 'mars');
      expect(result.correct).toBe(true);
    });

    it('should handle extra whitespace', () => {
      const result = evaluator.evaluateAnswer('Mars', '  Mars  ');
      expect(result.correct).toBe(true);
    });
  });

  describe('jeopardy format matching', () => {
    it('should handle "What is" format', () => {
      const result = evaluator.evaluateAnswer('Mars', 'What is Mars?');
      expect(result.correct).toBe(true);
      expect(result.method).toBe('jeopardy_format');
    });

    it('should handle "Who is" format', () => {
      const result = evaluator.evaluateAnswer('Shakespeare', 'Who is William Shakespeare?');
      expect(result.correct).toBe(true);
      expect(result.method).toBe('jeopardy_format');
    });

    it('should handle various question words', () => {
      const testCases = [
        ['Paris', 'Where is Paris?'],
        ['1912', 'When was 1912?'],
        ['Tennis', 'What are tennis?']
      ];

      testCases.forEach(([answer, response]) => {
        const result = evaluator.evaluateAnswer(answer, response);
        expect(result.correct).toBe(true);
      });
    });
  });

  describe('substring matching', () => {
    it('should match when answer is contained in response', () => {
      const result = evaluator.evaluateAnswer('Mars', 'The answer is Mars, the red planet');
      expect(result.correct).toBe(true);
      expect(result.method).toBe('substring');
    });

    it('should match when response is contained in answer', () => {
      const result = evaluator.evaluateAnswer('William Shakespeare', 'Shakespeare');
      expect(result.correct).toBe(true);
    });
  });

  describe('word matching', () => {
    it('should match significant words', () => {
      const result = evaluator.evaluateAnswer('William Shakespeare', 'The playwright William was Shakespeare');
      expect(result.correct).toBe(true);
      expect(result.method).toBe('word_match');
    });

    it('should not match with insufficient word overlap', () => {
      const result = evaluator.evaluateAnswer('Mars Venus Jupiter', 'Saturn');
      expect(result.correct).toBe(false);
    });
  });

  describe('negative answers', () => {
    it('should reject responses containing negative answers', () => {
      const result = evaluator.evaluateAnswer(
        'Mars', 
        'What is Mars?', 
        ['Venus', 'Jupiter']
      );
      expect(result.correct).toBe(true);
    });

    it('should fail when negative answer is present', () => {
      const result = evaluator.evaluateAnswer(
        'Mars', 
        'Venus is the answer', 
        ['Venus', 'Jupiter']
      );
      expect(result.correct).toBe(false);
      expect(result.method).toBe('negative_match');
    });
  });

  describe('text normalization', () => {
    it('should remove common articles and prepositions', () => {
      const result = evaluator.evaluateAnswer('The Great Wall', 'Great Wall');
      expect(result.correct).toBe(true);
    });

    it('should handle punctuation', () => {
      const result = evaluator.evaluateAnswer('Dr. Watson', 'Dr Watson');
      expect(result.correct).toBe(true);
    });
  });

  describe('confidence scoring', () => {
    it('should return high confidence for exact matches', () => {
      const result = evaluator.evaluateAnswer('Mars', 'Mars');
      expect(result.confidence).toBe(1.0);
    });

    it('should return lower confidence for fuzzy matches', () => {
      const result = evaluator.evaluateAnswer('Mars', 'The planet Mars is red');
      expect(result.confidence).toBeLessThan(1.0);
      expect(result.confidence).toBeGreaterThan(0.0);
    });

    it('should return zero confidence for no matches', () => {
      const result = evaluator.evaluateAnswer('Mars', 'Jupiter');
      expect(result.confidence).toBe(0.0);
    });
  });
});