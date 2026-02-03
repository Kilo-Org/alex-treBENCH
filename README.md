# Jeopardy Benchmark Tool

A simplified Jeopardy benchmarking tool that combines the question downloading capabilities from the `org` folder with the elegant design patterns from `skatebench`.

> **Note**: This project now uses [Bun](https://bun.sh) as the JavaScript runtime and package manager for improved performance and developer experience.

## Features

- **Question Download**: Automatically downloads Jeopardy questions from public datasets
- **Multi-Model Support**: Test multiple AI models simultaneously via OpenRouter
- **Smart Evaluation**: Multiple answer matching strategies (exact, fuzzy, Jeopardy format, etc.)
- **Caching System**: Intelligent result caching to minimize API costs
- **Real-time Progress**: Live terminal feedback during benchmarking
- **Cost Tracking**: Monitor token usage and API costs

## Prerequisites

- [Bun](https://bun.sh) runtime (latest version recommended)
- OpenRouter API key

## Quick Start

1. **Install dependencies:**
   ```bash
   bun install
   ```

2. **Set up OpenRouter API key:**
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Download sample questions:**
   ```bash
   bun run dev download --sample 50
   ```

4. **Run a quick benchmark:**
   ```bash
   bun run dev benchmark
   ```

## Commands

### Download Questions
```bash
# Download sample questions
bun run dev download --sample 50

# Download questions from specific category
bun run dev download --sample 30 --category "SCIENCE"

# Force re-download (bypass cache)
bun run dev download --force
```

### Run Benchmark
```bash
# Quick test with default models
bun run dev benchmark

# Test specific models
bun run dev benchmark --models gpt-4o-mini claude-3-haiku

# Test more questions
bun run dev benchmark --sample 100

# Filter by category
bun run dev benchmark --category "HISTORY" --sample 25

# Disable caching
bun run dev benchmark --no-cache

# Increase concurrency
bun run dev benchmark --concurrency 10
```

### List Available Models
```bash
bun run dev models
```

## Architecture

### Design Principles

This tool combines the best of both existing implementations:

**From `org` folder (Jeopardy tool):**
- Robust question downloading and caching
- Multiple answer matching strategies
- Professional error handling

**From `skatebench`:**
- Clean, functional TypeScript architecture
- Intelligent result caching
- Real-time progress feedback
- Simple but powerful evaluation

**Simplified improvements:**
- Reduced complexity while maintaining functionality
- Modern TypeScript with AI SDK integration
- Focus on usability over configurability
- Clear separation of concerns

### File Structure

```
app/
├── src/
│   ├── data/
│   │   └── downloader.ts     # Question downloading logic
│   ├── models/
│   │   └── config.ts         # Model configurations
│   ├── bench/
│   │   ├── evaluator.ts      # Answer evaluation logic
│   │   └── runner.ts         # Main benchmarking engine
│   └── index.ts              # CLI interface
├── package.json
├── tsconfig.json
└── README.md
```

## Answer Evaluation

The tool uses multiple evaluation strategies in order of precision:

1. **Exact Match**: Direct string comparison after normalization
2. **Jeopardy Format**: Handles "What is..." style responses
3. **Substring Match**: Checks if answer appears in response
4. **Word Match**: Matches significant words (70% threshold)
5. **Fuzzy Match**: Character similarity (80% threshold)

## Model Support

Currently supports models via OpenRouter including:
- OpenAI (GPT-4, GPT-3.5, O1 series)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
- Google (Gemini models)
- Meta (Llama models)
- DeepSeek and others

## Results

Results are saved to `./results/` with:
- Individual model results as JSON files
- Combined summary with accuracy statistics
- Cached results to avoid duplicate API calls

## Configuration

Key settings in `src/models/config.ts`:
- `maxConcurrency`: Number of simultaneous API calls
- `testRunsPerModel`: Repetitions per model (for consistency)
- `timeoutSeconds`: Request timeout
- `defaultSampleSize`: Default number of questions

## Cost Management

- **Caching**: Results are cached by model/question hash
- **Sampling**: Test on smaller subsets first
- **Concurrency Control**: Limit simultaneous requests
- **Usage Tracking**: Monitor costs per model

## Example Output

```
🧠 Jeopardy AI Benchmark

Testing 25 questions
Models: gpt-4o-mini, claude-3-haiku, gemini-2.0-flash

✓ gpt-4o-mini: 84.0% accuracy (1250ms avg)
✓ claude-3-haiku: 76.0% accuracy (980ms avg)  
✓ gemini-2.0-flash: 80.0% accuracy (1100ms avg)

📊 Final Results
==================================================
🥇 gpt-4o-mini        84.0% accuracy  21/25 correct
    Cost: $0.0023  Tokens: 450  Time: 1250ms avg

🥈 gemini-2.0-flash   80.0% accuracy  20/25 correct
    Cost: $0.0019  Tokens: 380  Time: 1100ms avg

🥉 claude-3-haiku     76.0% accuracy  19/25 correct
    Cost: $0.0015  Tokens: 420  Time: 980ms avg
```

## Development

```bash
# Install dependencies
bun install

# Run in development mode
bun run dev

# Build for production  
bun run build

# Run built version
bun start

# Run tests
bun test
```

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `JEOPARDY_CACHE_DIR`: Custom cache directory (optional)

## Comparison to Original Tools

| Feature | org (Jeopardy) | skatebench | app (This Tool) |
|---------|----------------|------------|-----------------|
| **Complexity** | Very High (15k+ LOC) | Medium (3k LOC) | **Low-Medium (1.5k LOC)** |
| **Setup** | Complex config | Simple | **Very Simple** |
| **Evaluation** | 6 strategies | 1 strategy | **4 strategies** |
| **Caching** | Database | File-based | **File-based** |
| **UI** | CLI + Progress | Terminal UI | **CLI + Spinners** |
| **Models** | Plugin system | Direct SDK | **Direct SDK** |

This tool strikes the right balance: more capable than skatebench, simpler than org.