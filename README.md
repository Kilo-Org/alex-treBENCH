# alex-treBENCH! 🎯

**Jeopardy Language Model Benchmarking System**

🎮 Benchmark LLMs with Jeopardy! questions. Tournament-style testing for large language models. What is... your model's true performance?

[![Bun Ready](https://img.shields.io/badge/bun-ready-orange.svg)](https://bun.sh/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/typescript-powered-blue.svg)](https://www.typescriptlang.org/)

A lightning-fast benchmarking system that evaluates language models using authentic Jeopardy! questions, delivering tournament-style competition results with statistical precision and entertaining flair.

> **Note**: This project now uses [Bun](https://bun.sh) as the JavaScript runtime and package manager for blazing-fast performance and an improved developer experience.

## 🏆 What is... alex-treBENCH?

This system transforms LLM evaluation into an engaging tournament experience:

- ✅ **Tournament Mode**: Pit multiple language models against each other in head-to-head Jeopardy! competition
- ✅ **Smart Evaluation**: Multiple answer-matching strategies handle the quirks of "What is..." format responses
- ✅ **Lightning Fast**: Powered by Bun runtime with intelligent caching to minimize costs and maximize speed
- ✅ **Real-time Drama**: Watch models compete with live progress bars and instant scoring
- ✅ **Cost Conscious**: Smart caching and sampling keep your API budget happy
- ✅ **Professional Results**: Generate comprehensive reports worthy of a game show finale

## 🎯 Key Features

### Tournament Capabilities

- **Multi-Model Showdowns**: Test 2-10 language models simultaneously via OpenRouter
- **Smart Question Selection**: Automatic sampling from authentic Jeopardy! datasets
- **Live Competition Feed**: Real-time terminal updates as models battle for supremacy
- **Podium Rankings**: Clear winner determination with accuracy, speed, and cost metrics
- **Replay System**: Intelligent result caching avoids duplicate API calls

### Evaluation Excellence

The system uses tournament-grade evaluation with multiple strategies:

1. **🎯 Exact Match**: Perfect accuracy for precise responses
2. **🎪 Jeopardy Format**: Handles "What is..." and "Who is..." responses like a pro
3. **🔍 Substring Detection**: Finds correct answers buried in verbose responses  
4. **📝 Word Matching**: Matches significant terms (70% threshold)
5. **🌟 Fuzzy Logic**: Character similarity matching (80% threshold)

### Performance Metrics

Track what matters in the tournament:
- **Accuracy Rate**: Correct responses / total questions
- **Response Speed**: Average time per question
- **Cost Efficiency**: API costs per correct answer
- **Token Usage**: Input/output token consumption
- **Consistency**: Performance variance across question types

## 🚀 Quick Tournament Setup

### Prerequisites

- [Bun](https://bun.sh) runtime (latest version recommended)
- OpenRouter API key ([get one here](https://openrouter.ai))
- Internet connection for the competition

### Installation

```bash
# Clone the tournament system
git clone <repository-url>
cd alex-treBENCH

# Install with lightning speed
bun install
```

### Competition Setup

```bash
# Set your API key for tournament access
export OPENROUTER_API_KEY=your_api_key_here
# Or add to .env file for convenience
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

### Your First Tournament

```bash
# Download sample questions for a quick match
bun run dev download --sample 50

# Start your first tournament!
bun run dev benchmark
```

## 🎮 Tournament Commands

### Question Management

```bash
# Download sample questions (perfect for testing)
bun run dev download --sample 50

# Focus on specific categories (like the real show!)
bun run dev download --sample 30 --category "SCIENCE"

# Force fresh download (bypass cache)
bun run dev download --force
```

### Running Tournaments

```bash
# Quick championship match with default contenders
bun run dev benchmark

# Custom tournament with your favorite models
bun run dev benchmark --models gpt-4o-mini claude-3-haiku gemini-2.0-flash

# Extended tournament with more questions
bun run dev benchmark --sample 100

# Category-specific showdown
bun run dev benchmark --category "HISTORY" --sample 25

# High-speed competition (disable caching for fresh results)
bun run dev benchmark --no-cache

# Tournament on steroids (increase concurrency)
bun run dev benchmark --concurrency 10
```

### Competitor Information

```bash
# View all available tournament contenders
bun run dev models
```

## 🏅 Tournament Results

### Sample Championship Output

```
🎯 alex-treBENCH Tournament Results
══════════════════════════════════════════════

🎮 Tournament: Quick Championship
📊 Questions: 25 | Categories: Mixed | Duration: 2m 15s

🏆 FINAL STANDINGS 🏆

🥇 CHAMPION: gpt-4o-mini
   📈 Accuracy: 84.0% (21/25 correct)
   ⚡ Speed: 1250ms average response
   💰 Cost: $0.0023 (most efficient!)
   
🥈 RUNNER-UP: gemini-2.0-flash  
   📈 Accuracy: 80.0% (20/25 correct)
   ⚡ Speed: 1100ms average response
   💰 Cost: $0.0019
   
🥉 THIRD PLACE: claude-3-haiku
   📈 Accuracy: 76.0% (19/25 correct)
   ⚡ Speed: 980ms average response (fastest!)
   💰 Cost: $0.0015 (most economical!)

🎪 Tournament Highlights:
• Most challenging category: SCIENCE (62% avg accuracy)
• Easiest category: POTPOURRI (88% avg accuracy)
• Closest match: Questions 12-15 (all models within 5%)
• Speed demon: claude-3-haiku dominated response times

Results saved to: ./results/tournament_2024_01_15_143022.json
```

## 🏗️ Tournament Architecture

### Design Philosophy

alex-treBENCH combines the best elements from multiple benchmarking approaches:

**🎯 From Professional Jeopardy! Systems:**
- Robust question downloading and intelligent caching
- Multiple answer evaluation strategies for real-world accuracy
- Professional error handling and recovery

**⚡ From Modern Benchmarking Tools:**
- Clean, functional TypeScript architecture with AI SDK integration
- Real-time progress feedback and tournament atmosphere
- Smart result caching to minimize API costs

**🚀 Simplified Excellence:**
- Reduced complexity while maintaining tournament-grade functionality
- Focus on user experience and entertainment value
- Clear separation of concerns with modular design

### File Structure

```
alex-treBENCH/
├── src/
│   ├── data/
│   │   └── downloader.ts     # Question acquisition system
│   ├── models/
│   │   └── config.ts         # Tournament competitor configurations
│   ├── bench/
│   │   ├── evaluator.ts      # Answer evaluation and scoring
│   │   └── runner.ts         # Tournament engine and orchestration
│   └── index.ts              # Tournament command center
├── results/                  # Tournament archives
├── cache/                    # Question and result cache
├── package.json              # Tournament manifest
└── README.md                 # This guide to glory!
```

## 🎯 Available Competitors

Currently supporting tournament-ready models via OpenRouter:

### 🏆 Championship Tier
- **OpenAI**: GPT-4 Turbo, GPT-4o, O1 Preview/Mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro

### 🥇 Professional Tier  
- **OpenAI**: GPT-3.5 Turbo, GPT-4o Mini
- **Anthropic**: Claude 3 Haiku, Claude 3 Sonnet
- **Meta**: Llama 3.1 405B, Llama 3.1 70B
- **Google**: Gemini 1.5 Flash

### 🥉 Challenger Tier
- **DeepSeek**: DeepSeek V2.5, DeepSeek Coder
- **Qwen**: Qwen 2.5 72B, Qwen 2.5 32B
- **Mistral**: Mistral Large, Mistral 7B
- **Others**: Many more available through OpenRouter

## 💰 Tournament Budget Management

Keep your tournament costs under control:

- **🎯 Smart Caching**: Results cached by model + question hash
- **📊 Strategic Sampling**: Start with smaller question sets
- **⚡ Concurrency Control**: Limit simultaneous API requests
- **📈 Usage Tracking**: Monitor costs per model in real-time
- **🎮 Quick Matches**: Use `--sample 25` for fast, cheap tournaments

### Cost Examples (Approximate)
- **Quick Tournament** (25 questions, 3 models): ~$0.05-0.15
- **Standard Tournament** (100 questions, 5 models): ~$0.20-0.60  
- **Championship** (500 questions, 10 models): ~$1.00-3.00

## 🛠️ Tournament Configuration

Key settings in `src/models/config.ts`:

```typescript
export const CONFIG = {
  maxConcurrency: 5,           // Simultaneous API calls
  testRunsPerModel: 1,         // Repetitions for consistency
  timeoutSeconds: 30,          // Request timeout
  defaultSampleSize: 50,       // Default question count
  cacheResults: true,          // Enable intelligent caching
  showProgress: true           // Live tournament feed
}
```

## 🎪 Development & Contributing

### Setup Your Development Tournament

```bash
# Install dependencies at lightning speed
bun install

# Run in development mode
bun run dev

# Build for production tournaments
bun run build

# Run the championship build
bun start

# Test the tournament system
bun test
```

### Environment Variables

- **`OPENROUTER_API_KEY`**: Your tournament access key (required)
- **`JEOPARDY_CACHE_DIR`**: Custom cache location (optional)
- **`TOURNAMENT_LOG_LEVEL`**: Logging verbosity (optional)

## 🏆 Tournament Comparison

| Feature | Original Complex | Simple Bench | **alex-treBENCH** |
|---------|------------------|--------------|-------------------|
| **Setup Complexity** | Very High | Simple | **Super Simple** |
| **Tournament Feel** | Technical | Basic | **🎯 Engaging & Fun** |
| **Evaluation Quality** | 6 strategies | 1 strategy | **🏆 4 Smart Strategies** |
| **Performance** | Slow | Fast | **⚡ Lightning Fast** |
| **Cost Control** | Basic | Good | **💰 Excellent** |
| **User Experience** | CLI | Terminal | **🎮 Tournament Drama** |
| **Result Presentation** | Text | Plain | **🏅 Championship Style** |

## 🎯 What is... Coming Next?

### 🚀 Planned Tournament Features
- **Category Championships**: Specialized tournaments by subject
- **Speed Rounds**: Lightning-fast evaluation challenges  
- **Historical Tracking**: Tournament season statistics
- **Web Interface**: Browser-based tournament viewing
- **Custom Datasets**: Import your own question sets

## 🙏 Tournament Acknowledgments

- **🏆 Jeopardy!**: For creating the ultimate question-and-answer format
- **📊 Kaggle**: For providing authentic Jeopardy! datasets
- **🚀 OpenRouter**: For unified access to tournament-worthy language models
- **⚡ Bun**: For making JavaScript fast enough for real-time tournaments

## 📞 Tournament Support

Need help running your tournaments?

- 🎯 **Quick Start Issues**: Check your API key and internet connection
- 🏆 **Tournament Questions**: Review the command examples above  
- 🚀 **Feature Requests**: Open an issue in the GitHub repository
- 💬 **Bug Reports**: Include your tournament logs and configuration

---

**🎉 Ready to Tournament!** 

What is... the champion in your next LLM tournament? Fire up alex-treBENCH and find out! 🏆

*"Thank you for playing alex-treBENCH! We have some lovely parting gifts..."* 🎁