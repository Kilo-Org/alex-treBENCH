# Debug Mode Guide

Debug mode provides detailed logging of all model interactions during benchmarking, helping you diagnose issues with model performance, prompt formatting, and answer evaluation.

## Overview

When debug mode is enabled, alex-treBENCH logs:
- **Exact prompts** sent to models
- **Raw responses** from models  
- **Parsed answers** extracted from responses
- **Grading details** including fuzzy match scores
- **Performance metrics** like response times and costs
- **Error details** when requests fail

## Enabling Debug Mode

### Command Line Flags

```bash
# Enable full debug logging
alex benchmark run --debug --model openai/gpt-4 --size quick

# Log only incorrect answers and errors (recommended for analysis)
alex benchmark run --debug --debug-errors-only --model anthropic/claude-3-haiku --size standard

# Debug with custom benchmark settings
alex benchmark run --debug --model openai/gpt-3.5-turbo --size comprehensive --grading-mode strict
```

### Configuration File

You can also enable debug mode permanently in `config/default.yaml`:

```yaml
logging:
  debug:
    enabled: true              # Enable debug logging
    log_dir: "logs/debug"      # Output directory
    log_prompts: true          # Log formatted prompts
    log_responses: true        # Log model responses
    log_grading: true          # Log grading details
    log_errors_only: false    # If true, only log incorrect answers
    include_tokens: true       # Include token counts
    include_costs: true        # Include cost information
```

## Output Files

Debug mode creates two types of files in `logs/debug/`:

### 1. JSON Lines Files (`model_interactions_*.jsonl`)

Structured data for programmatic analysis:

```json
{
    "timestamp": "2025-09-04T14:28:08.992383",
    "benchmark_id": null,
    "question_id": "q_35309_-2597327464560468404",
    "model_name": "openai/gpt-3.5-turbo",
    "category": "THAT'S MY DRINK",
    "value": 600,
    "question_text": "'Gin, lime & club soda:Gin this guy'",
    "correct_answer": "Rickey",
    "formatted_prompt": "You are a Jeopardy! contestant...",
    "raw_response": "What is a Gin Rickey?",
    "parsed_answer": "What is a gin rickey?",
    "is_correct": true,
    "match_score": 1.0,
    "match_type": "fuzzy",
    "confidence_score": 1.0,
    "response_time_ms": 751.0,
    "cost_usd": 4.95e-05,
    "tokens_input": 78,
    "tokens_output": 7,
    "grading_details": {
        "fuzzy_threshold": 0.8,
        "semantic_threshold": 0.7,
        "mode": "JEOPARDY",
        "match_details": {
            "ratio": 0.46,
            "partial_ratio": 1.0,
            "token_sort_ratio": 0.46,
            "token_set_ratio": 1.0,
            "best_score": 1.0
        }
    },
    "error": null
}
```

### 2. Summary Log Files (`debug_summary_*.log`)

Human-readable format for quick analysis:

```
2025-09-04 14:28:08,233 - DEBUG - PROMPT Qq_205799 [openai/gpt-3.5-turbo]:
  Question: 'Classic French dressing also has this 1-word name, from a key ingredient'  
  Prompt:
    You are a Jeopardy! contestant. Respond to each clue in the form of a question...
    
    Category: SALAD DRESSING
    Value: $500
    
    Clue: 'Classic French dressing also has this 1-word name, from a key ingredient'
    
    Response:

2025-09-04 14:28:10,519 - DEBUG - ✗ INCORRECT Qq_114081 [openai/gpt-3.5-turbo]:
  Answer:   What is bleach?
  Expected: bleach (or chlorine)  
  Score:    0.600 (fuzzy)
  Question: 'Lethal gases are released when you combine some toilet bowl cleansers with this common stain remover'
  Response: What is bleach?
  Parsed:   What is bleach?
```

## Common Analysis Tasks

### Finding Models Getting 0% Accuracy

```bash
# Run debug mode
alex benchmark run --debug --debug-errors-only --model suspicious-model --size quick

# Check for systematic issues
grep "✗ INCORRECT" logs/debug/debug_summary_*.log | head -10

# Look for parsing failures  
grep "ERROR" logs/debug/debug_summary_*.log

# Analyze JSON data programmatically
python -c "
import json
with open('logs/debug/model_interactions_*.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if not data['is_correct']:
            print(f'Q: {data[\"question_text\"]}')
            print(f'Expected: {data[\"correct_answer\"]}') 
            print(f'Got: {data[\"raw_response\"]}')
            print(f'Score: {data[\"match_score\"]}')
            print('---')
"
```

### Analyzing Prompt Issues

```bash
# Look for questions where the model consistently fails
grep -B5 -A5 "Score: 0.000" logs/debug/debug_summary_*.log

# Check if prompt format is causing confusion
grep -A15 "PROMPT" logs/debug/debug_summary_*.log | grep -A15 "specific-category"
```

### Performance Analysis

```bash
# Find slow responses
grep "Time: [5-9][0-9][0-9][0-9]ms\|Time: [0-9][0-9][0-9][0-9][0-9]ms" logs/debug/debug_summary_*.log

# Cost analysis
python -c "
import json
total_cost = 0
count = 0
with open('logs/debug/model_interactions_*.jsonl') as f:
    for line in f:
        data = json.loads(line)
        total_cost += data['cost_usd']
        count += 1
print(f'Average cost per question: ${total_cost/count:.6f}')
print(f'Total cost: ${total_cost:.6f}')
"
```

## Debug Mode Options

| Flag | Description | Use Case |
|------|-------------|----------|
| `--debug` | Enable full debug logging | Comprehensive analysis of all interactions |
| `--debug-errors-only` | Log only incorrect answers | Focus on problematic questions and responses |
| No flags | Standard logging only | Normal benchmarking without debug overhead |

## Performance Impact

Debug mode has minimal performance impact:
- **File I/O**: Small overhead for writing logs (~1-2% slower)
- **Memory**: Negligible increase  
- **Network**: No impact on API calls
- **Storage**: ~2MB per 1000 questions (varies by response length)

## Troubleshooting Common Issues

### Issue: Model Getting 0% Score

**Symptoms**: All questions marked incorrect despite seemingly correct answers

**Debug Steps**:
1. Run with `--debug --debug-errors-only` 
2. Check if responses are in correct Jeopardy format
3. Look at fuzzy match scores - scores >0.6 suggest format issues
4. Verify the grading mode matches your expectations

**Example Analysis**:
```bash
alex benchmark run --debug --debug-errors-only --model problematic-model --size quick
grep -A3 "Expected:" logs/debug/debug_summary_*.log | head -20
```

### Issue: Inconsistent Performance

**Symptoms**: Same model performs differently across runs

**Debug Steps**:
1. Compare prompts between runs to ensure consistency
2. Check response times for API issues  
3. Look for error patterns in specific categories

### Issue: Unexpected Costs

**Symptoms**: Costs higher than expected

**Debug Steps**:
1. Check token counts in JSON logs
2. Look for unusually long responses
3. Verify model pricing in debug output

## File Management

Debug logs can accumulate quickly. Consider:

```bash
# Clean old debug logs (keep last 10 files)
ls -t logs/debug/*.log | tail -n +11 | xargs rm -f

# Archive debug logs by date  
mkdir -p logs/archive/$(date +%Y-%m-%d)
mv logs/debug/* logs/archive/$(date +%Y-%m-%d)/

# Compress large debug files
gzip logs/debug/*.jsonl
```

## Integration with External Tools

### Python Analysis

```python
import json
import pandas as pd

# Load debug data into pandas
data = []
with open('logs/debug/model_interactions_20250904_142805.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Analysis examples
print(f"Accuracy by category:")
print(df.groupby('category')['is_correct'].mean().sort_values())

print(f"Average response time: {df['response_time_ms'].mean():.1f}ms")
print(f"Questions with low match scores:")
print(df[df['match_score'] < 0.5][['question_text', 'correct_answer', 'raw_response', 'match_score']])
```

### Jupyter Notebook Integration

Debug JSON files work excellently with Jupyter notebooks for interactive analysis, visualization, and model comparison.

## Best Practices

1. **Use `--debug-errors-only` first** - focuses on problems without overwhelming detail
2. **Run small samples initially** - debug with `--size quick` before full benchmarks  
3. **Compare models systematically** - use same debug settings across model comparisons
4. **Archive important debug sessions** - keep debug logs for models that will be used in production
5. **Monitor disk space** - debug logs for large benchmarks can consume significant storage

## Security Considerations

Debug logs contain:
- Full question text (public Jeopardy data)
- Model responses  
- API timing and cost data

**No sensitive data is logged**, but consider access controls for debug directories in shared environments.