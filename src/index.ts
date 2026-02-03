#!/usr/bin/env node

/**
 * Main CLI entry point for Jeopardy Benchmark Tool
 * Combines question downloading from org with benchmarking design from skatebench
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';

import { JeopardyDownloader } from './data/downloader.js';
import { JeopardyBenchmarkRunner } from './bench/runner.js';
import { 
  availableModels, 
  defaultTestModels, 
  getModelsByName, 
  getAvailableModelNames 
} from './models/config.js';

const program = new Command();

program
  .name('jeopardy-bench')
  .description('Simplified Jeopardy benchmarking tool')
  .version('1.0.0');

program
  .command('download')
  .description('Download Jeopardy questions dataset')
  .option('-f, --force', 'Force re-download even if cached')
  .option('-s, --sample <count>', 'Sample a specific number of questions', '50')
  .option('-c, --category <category>', 'Filter by category')
  .action(async (options) => {
    const spinner = ora('Downloading Jeopardy questions...').start();
    
    try {
      const downloader = new JeopardyDownloader();
      
      if (options.sample) {
        const sampleCount = parseInt(options.sample, 10);
        const questions = await downloader.sampleQuestions(sampleCount, options.category);
        spinner.succeed(`Downloaded ${questions.length} sample questions`);
        
        console.log('\nSample questions:');
        questions.slice(0, 5).forEach((q, i) => {
          console.log(`\n${i + 1}. Category: ${chalk.cyan(q.category)}`);
          console.log(`   Question: ${q.question}`);
          console.log(`   Answer: ${chalk.green(q.answer)}`);
        });
        
        if (questions.length > 5) {
          console.log(`\n... and ${questions.length - 5} more questions`);
        }
      } else {
        const questions = await downloader.downloadQuestions(options.force);
        spinner.succeed(`Downloaded ${questions.length} questions`);
        
        // Show category breakdown
        const categories = questions.reduce((acc, q) => {
          acc[q.category] = (acc[q.category] || 0) + 1;
          return acc;
        }, {} as Record<string, number>);
        
        console.log('\nCategories:');
        Object.entries(categories)
          .sort(([,a], [,b]) => b - a)
          .slice(0, 10)
          .forEach(([cat, count]) => {
            console.log(`  ${chalk.cyan(cat)}: ${count} questions`);
          });
      }
    } catch (error) {
      spinner.fail('Failed to download questions');
      console.error(chalk.red(`Error: ${error instanceof Error ? error.message : error}`));
      process.exit(1);
    }
  });

program
  .command('benchmark')
  .description('Run Jeopardy benchmark on AI models')
  .option('-m, --models <models...>', 'Models to test (space-separated)')
  .option('-s, --sample <count>', 'Number of questions to test', '25')
  .option('-c, --category <category>', 'Filter questions by category')
  .option('--no-cache', 'Disable result caching')
  .option('--concurrency <num>', 'Max concurrent requests', '5')
  .action(async (options) => {
    console.log(chalk.blue('🧠 Jeopardy AI Benchmark\n'));
    
    try {
      // Get questions
      const downloader = new JeopardyDownloader();
      const sampleCount = parseInt(options.sample, 10);
      const questions = await downloader.sampleQuestions(sampleCount, options.category);
      
      console.log(`Testing ${questions.length} questions`);
      if (options.category) {
        console.log(`Category filter: ${chalk.cyan(options.category)}`);
      }
      
      // Get models
      let models = getModelsByName(options.models || defaultTestModels);
      
      if (options.models) {
        const requestedModels = options.models;
        const availableNames = getAvailableModelNames();
        const invalidModels = requestedModels.filter(m => !availableNames.includes(m));
        
        if (invalidModels.length > 0) {
          console.error(chalk.red(`Invalid models: ${invalidModels.join(', ')}`));
          console.log(`Available models: ${availableNames.join(', ')}`);
          process.exit(1);
        }
      }
      
      console.log(`Models: ${models.map(m => chalk.green(m.name)).join(', ')}\n`);
      
      // Run benchmark
      const runner = new JeopardyBenchmarkRunner();
      const modelSpinners = new Map<string, any>();
      
      const results = await runner.runBenchmark(models, questions, {
        useCache: options.cache,
        maxConcurrency: parseInt(options.concurrency, 10),
        onEvent: (event) => {
          switch (event.type) {
            case 'start':
              const spinner = ora(`Testing ${event.model}...`).start();
              modelSpinners.set(event.model!, spinner);
              break;
              
            case 'progress':
              const progressSpinner = modelSpinners.get(event.model!);
              if (progressSpinner) {
                progressSpinner.text = `Testing ${event.model} (${event.progress}/${event.total})`;
              }
              break;
              
            case 'complete':
              const completeSpinner = modelSpinners.get(event.model!);
              if (completeSpinner && event.summary) {
                const accuracy = (event.summary.accuracy * 100).toFixed(1);
                const avgTime = event.summary.averageDuration.toFixed(0);
                completeSpinner.succeed(
                  `${event.model}: ${chalk.green(accuracy + '%')} accuracy (${avgTime}ms avg)`
                );
              }
              break;
              
            case 'error':
              const errorSpinner = modelSpinners.get(event.model!);
              if (errorSpinner) {
                errorSpinner.fail(`${event.model}: ${chalk.red(event.error)}`);
              }
              break;
          }
        }
      });
      
      // Display final results
      console.log('\n' + chalk.blue('📊 Final Results'));
      console.log('=' + '='.repeat(50));
      
      const sortedResults = Array.from(results.values()).sort((a, b) => b.accuracy - a.accuracy);
      
      sortedResults.forEach((summary, index) => {
        const accuracy = (summary.accuracy * 100).toFixed(1);
        const rank = index + 1;
        const medal = rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : '  ';
        
        console.log(`${medal} ${summary.model.padEnd(20)} ${chalk.green(accuracy + '%').padStart(8)} accuracy  ${summary.correctAnswers}/${summary.totalQuestions} correct`);
        
        if (summary.totalCost > 0) {
          console.log(`    Cost: $${summary.totalCost.toFixed(4)}  Tokens: ${summary.totalTokens}  Time: ${summary.averageDuration.toFixed(0)}ms avg`);
        }
        
        if (summary.errors > 0) {
          console.log(`    ${chalk.red('Errors:')} ${summary.errors}`);
        }
        console.log();
      });
      
    } catch (error) {
      console.error(chalk.red(`\nBenchmark failed: ${error instanceof Error ? error.message : error}`));
      process.exit(1);
    }
  });

program
  .command('models')
  .description('List available models')
  .action(() => {
    console.log(chalk.blue('Available Models:\n'));
    
    const grouped = availableModels.reduce((acc, model) => {
      // Group by provider (first part of model name before slash or dash)
      const provider = model.name.split(/[-/]/)[0];
      if (!acc[provider]) acc[provider] = [];
      acc[provider].push(model);
      return acc;
    }, {} as Record<string, typeof availableModels>);
    
    Object.entries(grouped).forEach(([provider, models]) => {
      console.log(chalk.cyan(`${provider.toUpperCase()}:`));
      models.forEach(model => {
        const reasoning = model.reasoning ? chalk.yellow(' (reasoning)') : '';
        console.log(`  • ${model.name}${reasoning}`);
      });
      console.log();
    });
    
    console.log(chalk.dim(`Default test models: ${defaultTestModels.join(', ')}`));
  });

// Handle errors
program.on('command:*', () => {
  console.error(chalk.red(`Invalid command: ${program.args.join(' ')}`));
  console.log('Use --help for available commands');
  process.exit(1);
});

// Parse arguments
program.parse();