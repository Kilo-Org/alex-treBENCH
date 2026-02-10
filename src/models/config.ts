/**
 * Model configuration inspired by skatebench/constants.ts
 * Simplified for Jeopardy benchmarking
 */

import { type LanguageModel } from "ai";
import { openrouter } from "@openrouter/ai-sdk-provider";

export const BENCHMARK_CONFIG = {
  outputDirectory: "./results",
  maxConcurrency: 10, // Reduced for simpler use
  testRunsPerModel: 20, // Reduced for faster iteration
  timeoutSeconds: 120,
  staggerDelayMs: 200,
  defaultSampleSize: 50, // Number of questions to test by default
};

export interface RunnableModel {
  name: string;
  llm: LanguageModel;
  providerOptions?: any;
  reasoning?: boolean;
}

// Include "usage" so we can log cost
const defaultProviderOptions = {
  usage: {
    include: true,
  },
};

// Simplified model list focused on popular/available models
export const availableModels: RunnableModel[] = [
  // OpenAI Models
  {
    name: "gpt-4o",
    llm: openrouter("openai/gpt-4o", defaultProviderOptions),
  },
  {
    name: "gpt-4o-mini",
    llm: openrouter("openai/gpt-4o-mini", defaultProviderOptions),
  },
  {
    name: "gpt-3.5-turbo",
    llm: openrouter("openai/gpt-3.5-turbo", defaultProviderOptions),
  },
  
  // Anthropic Models
  {
    name: "claude-3.5-sonnet",
    llm: openrouter("anthropic/claude-3.5-sonnet", defaultProviderOptions),
  },
  {
    name: "claude-3-haiku",
    llm: openrouter("anthropic/claude-3-haiku", defaultProviderOptions),
  },

  // Google Models
  {
    name: "gemini-2.0-flash",
    llm: openrouter("google/gemini-2.0-flash-exp", defaultProviderOptions),
  },
  {
    name: "gemini-1.5-pro",
    llm: openrouter("google/gemini-pro-1.5", defaultProviderOptions),
  },

  // DeepSeek Models
  {
    name: "deepseek-v3",
    llm: openrouter("deepseek/deepseek-chat", defaultProviderOptions),
  },

  // Meta Models
  {
    name: "llama-3.1-70b",
    llm: openrouter("meta-llama/llama-3.1-70b-instruct", defaultProviderOptions),
  },

  // Reasoning Models (if available)
  {
    name: "o1-preview",
    llm: openrouter("openai/o1-preview", defaultProviderOptions),
    reasoning: true,
  },
  {
    name: "o1-mini",
    llm: openrouter("openai/o1-mini", defaultProviderOptions),
    reasoning: true,
  },
];

// Default models for quick testing
export const defaultTestModels = [
  "gpt-4o-mini",
  "claude-3-haiku", 
  "gemini-2.0-flash"
];

/**
 * Get models by name
 */
export function getModelsByName(modelNames: string[]): RunnableModel[] {
  return availableModels.filter(model => 
    modelNames.includes(model.name)
  );
}

/**
 * Get all available model names
 */
export function getAvailableModelNames(): string[] {
  return availableModels.map(model => model.name);
}