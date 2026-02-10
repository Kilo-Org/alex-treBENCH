/**
 * Answer evaluation logic for Jeopardy questions
 * Combines simple string matching with Jeopardy-specific patterns
 */

export interface EvaluationResult {
  correct: boolean;
  confidence: number;
  method: string;
  normalizedAnswer: string;
  normalizedResponse: string;
}

export class JeopardyEvaluator {
  /**
   * Evaluate if a model's response correctly answers a Jeopardy question
   */
  evaluateAnswer(
    expectedAnswer: string,
    modelResponse: string,
    negativeAnswers: string[] = []
  ): EvaluationResult {
    const normalizedResponse = this.normalizeText(modelResponse);
    const normalizedExpected = this.normalizeText(expectedAnswer);

    // First check for negative matches (immediate fail)
    for (const negAnswer of negativeAnswers) {
      const normalizedNeg = this.normalizeText(negAnswer);
      if (normalizedResponse.includes(normalizedNeg)) {
        return {
          correct: false,
          confidence: 1.0,
          method: 'negative_match',
          normalizedAnswer: normalizedExpected,
          normalizedResponse
        };
      }
    }

    // Try different matching strategies in order of specificity
    const strategies = [
      () => this.exactMatch(normalizedExpected, normalizedResponse),
      () => this.jeopardyFormatMatch(expectedAnswer, modelResponse),
      () => this.substringMatch(normalizedExpected, normalizedResponse),
      () => this.wordMatch(normalizedExpected, normalizedResponse),
      () => this.fuzzyMatch(normalizedExpected, normalizedResponse)
    ];

    for (const strategy of strategies) {
      const result = strategy();
      if (result.correct) {
        return {
          ...result,
          normalizedAnswer: normalizedExpected,
          normalizedResponse
        };
      }
    }

    return {
      correct: false,
      confidence: 0.0,
      method: 'no_match',
      normalizedAnswer: normalizedExpected,
      normalizedResponse
    };
  }

  /**
   * Normalize text for comparison - removes common noise
   */
  private normalizeText(text: string): string {
    return text
      .toLowerCase()
      .trim()
      // Remove common articles and prepositions
      .replace(/\b(the|a|an|and|or|of|in|on|at|to|for|with|by)\b/g, '')
      // Remove punctuation except apostrophes
      .replace(/[^\w\s']/g, '')
      // Collapse multiple spaces
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * Exact match after normalization
   */
  private exactMatch(expected: string, response: string): Partial<EvaluationResult> {
    return {
      correct: expected === response,
      confidence: expected === response ? 1.0 : 0.0,
      method: 'exact'
    };
  }

  /**
   * Handle Jeopardy-style responses like "What is..." or "Who is..."
   */
  private jeopardyFormatMatch(expected: string, response: string): Partial<EvaluationResult> {
    const jeopardyPattern = /^(what|who|where|when|why|how)\s+(is|are|was|were)\s+(.+)$/i;
    const match = response.match(jeopardyPattern);
    
    if (match) {
      const extractedAnswer = match[3];
      const normalizedExtracted = this.normalizeText(extractedAnswer);
      const normalizedExpected = this.normalizeText(expected);
      
      if (normalizedExtracted.includes(normalizedExpected) || 
          normalizedExpected.includes(normalizedExtracted)) {
        return {
          correct: true,
          confidence: 0.9,
          method: 'jeopardy_format'
        };
      }
    }

    return { correct: false, confidence: 0.0, method: 'jeopardy_format' };
  }

  /**
   * Check if expected answer is contained in response or vice versa
   */
  private substringMatch(expected: string, response: string): Partial<EvaluationResult> {
    const isSubstring = response.includes(expected) || expected.includes(response);
    
    return {
      correct: isSubstring,
      confidence: isSubstring ? 0.8 : 0.0,
      method: 'substring'
    };
  }

  /**
   * Check if key words from expected answer appear in response
   */
  private wordMatch(expected: string, response: string): Partial<EvaluationResult> {
    const expectedWords = expected.split(/\s+/).filter(word => word.length > 2);
    const responseWords = new Set(response.split(/\s+/));
    
    const matchingWords = expectedWords.filter(word => responseWords.has(word));
    const matchRatio = matchingWords.length / expectedWords.length;
    
    // Consider it a match if 70% or more of significant words are present
    const correct = matchRatio >= 0.7 && expectedWords.length > 0;
    
    return {
      correct,
      confidence: correct ? matchRatio : 0.0,
      method: 'word_match'
    };
  }

  /**
   * Simple fuzzy matching based on character overlap
   */
  private fuzzyMatch(expected: string, response: string): Partial<EvaluationResult> {
    if (expected.length === 0 || response.length === 0) {
      return { correct: false, confidence: 0.0, method: 'fuzzy' };
    }

    const similarity = this.calculateSimilarity(expected, response);
    const correct = similarity >= 0.8; // 80% similarity threshold
    
    return {
      correct,
      confidence: similarity,
      method: 'fuzzy'
    };
  }

  /**
   * Calculate Jaccard similarity between two strings
   */
  private calculateSimilarity(str1: string, str2: string): number {
    const set1 = new Set(str1.split(''));
    const set2 = new Set(str2.split(''));
    
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  }
}