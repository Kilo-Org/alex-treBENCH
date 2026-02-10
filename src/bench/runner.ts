/**
 * Main benchmarking runner
 * Inspired by skatebench/index.ts but adapted for Jeopardy questions
 */

import { generateText } from 'ai';
import fs from 'fs/promises';
import path from 'path';
import { createHash } from 'crypto';

import { type RunnableModel, BENCHMARK_CONFIG } from '../models/config.js';
import { type JeopardyQuestion } from '../data/downloader.js';
import { JeopardyEvaluator, type EvaluationResult } from './evaluator.js';

export interface BenchmarkResult {
  model: string;
  question: JeopardyQuestion;
  response: string;
  evaluation: EvaluationResult;
  duration: number;
  cost?: number;
  completionTokens?: number;
  error?: string;
}

export interface BenchmarkSummary {
  model: string;
  totalQuestions: number;
  correctAnswers: number;
  accuracy: number;
  averageDuration: number;
  totalCost: number;
  totalTokens: number;
  errors: number;
}

export interface RunnerEvent {
  type: 'start' | 'progress' | 'complete' | 'error';
  model?: string;
  progress?: number;
  total?: number;
  result?: BenchmarkResult;
  summary?: BenchmarkSummary;
  error?: string;
}

export class JeopardyBenchmarkRunner {
  private evaluator: JeopardyEvaluator;
  private cacheDir: string;

  constructor(cacheDir: string = path.join(BENCHMARK_CONFIG.outputDirectory, 'cache')) {
    this.evaluator = new JeopardyEvaluator();
    this.cacheDir = cacheDir;
  }

  /**
   * Run benchmark on a set of models with given questions
   */
  async runBenchmark(
    models: RunnableModel[],
    questions: JeopardyQuestion[],
    options: {
      onEvent?: (event: RunnerEvent) => void;
      useCache?: boolean;
      maxConcurrency?: number;
    } = {}
  ): Promise<Map<string, BenchmarkSummary>> {
    const { onEvent, useCache = true, maxConcurrency = BENCHMARK_CONFIG.maxConcurrency } = options;

    await this.ensureOutputDir();
    
    const results = new Map<string, BenchmarkResult[]>();
    
    for (const model of models) {
      onEvent?.({ type: 'start', model: model.name });
      
      try {
        const modelResults = await this.runModelBenchmark(model, questions, {
          useCache,
          maxConcurrency,
          onProgress: (completed, total) => {
            onEvent?.({ 
              type: 'progress', 
              model: model.name, 
              progress: completed, 
              total 
            });
          }
        });
        
        results.set(model.name, modelResults);
        
        const summary = this.calculateSummary(model.name, modelResults);
        onEvent?.({ type: 'complete', model: model.name, summary });
        
      } catch (error) {
        onEvent?.({ 
          type: 'error', 
          model: model.name, 
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Save results and return summaries
    await this.saveResults(results);
    return this.generateSummaries(results);
  }

  /**
   * Run benchmark for a single model
   */
  private async runModelBenchmark(
    model: RunnableModel,
    questions: JeopardyQuestion[],
    options: {
      useCache: boolean;
      maxConcurrency: number;
      onProgress?: (completed: number, total: number) => void;
    }
  ): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];
    const semaphore = new Semaphore(options.maxConcurrency);
    
    const promises = questions.map(async (question, index) => {
      await semaphore.acquire();
      try {
        const cacheKey = this.getCacheKey(model.name, question);
        let result: BenchmarkResult;

        if (options.useCache) {
          const cached = await this.getCachedResult(cacheKey);
          if (cached) {
            result = cached;
            results.push(result);
            options.onProgress?.(results.length, questions.length);
            return;
          }
        }

        result = await this.runSingleQuestion(model, question);
        
        if (options.useCache) {
          await this.cacheResult(cacheKey, result);
        }
        
        results.push(result);
        options.onProgress?.(results.length, questions.length);
        
      } finally {
        semaphore.release();
      }
    });

    await Promise.all(promises);
    return results.sort((a, b) => questions.findIndex(q => q === a.question) - questions.findIndex(q => q === b.question));
  }

  /**
   * Run a single question against a model
   */
  private async runSingleQuestion(
    model: RunnableModel,
    question: JeopardyQuestion
  ): Promise<BenchmarkResult> {
    const startTime = Date.now();
    
    try {
      const systemPrompt = this.buildSystemPrompt();
      const prompt = this.buildQuestionPrompt(question);
      
      const result = await generateText({
        model: model.llm,
        system: systemPrompt,
        prompt,
        maxTokens: 100, // Keep responses concise
        temperature: 0.1, // Low temperature for consistent answers
      });

      const duration = Date.now() - startTime;
      const response = result.text.trim();
      
      const evaluation = this.evaluator.evaluateAnswer(
        question.answer,
        response
      );

      return {
        model: model.name,
        question,
        response,
        evaluation,
        duration,
        cost: result.usage?.totalCost,
        completionTokens: result.usage?.completionTokens,
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      return {
        model: model.name,
        question,
        response: '',
        evaluation: {
          correct: false,
          confidence: 0,
          method: 'error',
          normalizedAnswer: question.answer,
          normalizedResponse: ''
        },
        duration,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Build system prompt for Jeopardy questions
   */
  private buildSystemPrompt(): string {
    return `You are answering Jeopardy questions. Provide concise, accurate answers.

Guidelines:
- Give direct answers without extra explanation
- For person questions, provide the full name if known
- For place questions, be specific (city, country, etc.)
- For date questions, provide the year or specific date
- You may respond in Jeopardy format (e.g., "What is...") but it's not required

Examples:
Question: "This planet is known as the Red Planet"
Answer: Mars

Question: "He wrote Romeo and Juliet"
Answer: William Shakespeare

Question: "This ship sank on its maiden voyage in 1912"
Answer: Titanic`;
  }

  /**
   * Build prompt for a specific question
   */
  private buildQuestionPrompt(question: JeopardyQuestion): string {
    let prompt = `Category: ${question.category}\n`;
    prompt += `Question: ${question.question}`;
    
    if (question.value) {
      prompt += `\nValue: $${question.value}`;
    }
    
    return prompt;
  }

  /**
   * Calculate summary statistics for a model's results
   */
  private calculateSummary(modelName: string, results: BenchmarkResult[]): BenchmarkSummary {
    const correctAnswers = results.filter(r => r.evaluation.correct).length;
    const totalCost = results.reduce((sum, r) => sum + (r.cost || 0), 0);
    const totalTokens = results.reduce((sum, r) => sum + (r.completionTokens || 0), 0);
    const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
    const errors = results.filter(r => r.error).length;

    return {
      model: modelName,
      totalQuestions: results.length,
      correctAnswers,
      accuracy: results.length > 0 ? correctAnswers / results.length : 0,
      averageDuration: results.length > 0 ? totalDuration / results.length : 0,
      totalCost,
      totalTokens,
      errors
    };
  }

  /**
   * Generate summaries for all model results
   */
  private generateSummaries(results: Map<string, BenchmarkResult[]>): Map<string, BenchmarkSummary> {
    const summaries = new Map<string, BenchmarkSummary>();
    
    for (const [modelName, modelResults] of results) {
      summaries.set(modelName, this.calculateSummary(modelName, modelResults));
    }
    
    return summaries;
  }

  /**
   * Generate cache key for a model/question combination
   */
  private getCacheKey(modelName: string, question: JeopardyQuestion): string {
    const content = JSON.stringify({ modelName, question });
    return createHash('sha256').update(content).digest('hex');
  }

  /**
   * Get cached result if available
   */
  private async getCachedResult(cacheKey: string): Promise<BenchmarkResult | null> {
    try {
      const cachePath = path.join(this.cacheDir, `${cacheKey}.json`);
      const content = await fs.readFile(cachePath, 'utf-8');
      return JSON.parse(content);
    } catch {
      return null;
    }
  }

  /**
   * Cache a result
   */
  private async cacheResult(cacheKey: string, result: BenchmarkResult): Promise<void> {
    try {
      await fs.mkdir(this.cacheDir, { recursive: true });
      const cachePath = path.join(this.cacheDir, `${cacheKey}.json`);
      await fs.writeFile(cachePath, JSON.stringify(result, null, 2));
    } catch (error) {
      // Ignore cache errors - they shouldn't fail the benchmark
      console.warn('Failed to cache result:', error);
    }
  }

  /**
   * Save results to output directory
   */
  private async saveResults(results: Map<string, BenchmarkResult[]>): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const outputDir = path.join(BENCHMARK_CONFIG.outputDirectory, `run-${timestamp}`);
    await fs.mkdir(outputDir, { recursive: true });

    // Save individual model results
    for (const [modelName, modelResults] of results) {
      const filename = `${modelName.replace(/[^a-zA-Z0-9-]/g, '_')}.json`;
      await fs.writeFile(
        path.join(outputDir, filename),
        JSON.stringify(modelResults, null, 2)
      );
    }

    // Save combined summary
    const summaries = this.generateSummaries(results);
    const summaryData = Array.from(summaries.values());
    await fs.writeFile(
      path.join(outputDir, 'summary.json'),
      JSON.stringify(summaryData, null, 2)
    );

    console.log(`Results saved to: ${outputDir}`);
  }

  /**
   * Ensure output directory exists
   */
  private async ensureOutputDir(): Promise<void> {
    await fs.mkdir(BENCHMARK_CONFIG.outputDirectory, { recursive: true });
    await fs.mkdir(this.cacheDir, { recursive: true });
  }
}

/**
 * Simple semaphore for concurrency control
 */
class Semaphore {
  private permits: number;
  private waiting: Array<() => void> = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise<void>((resolve) => {
      this.waiting.push(resolve);
    });
  }

  release(): void {
    if (this.waiting.length > 0) {
      const resolve = this.waiting.shift()!;
      resolve();
    } else {
      this.permits++;
    }
  }
}