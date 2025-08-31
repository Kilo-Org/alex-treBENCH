# Documented Tasks

This file contains step-by-step workflows for repetitive tasks that follow similar patterns in the alex-treBENCH project.

## Add New Model Support

**Last performed:** [Never - this is the initial documentation]
**Applies to:** When OpenRouter releases new models that should be supported in benchmarks
**Complexity:** Low - The system has automatic model discovery
**Time required:** 5-15 minutes (mostly verification)

### Overview

alex-treBENCH uses a **dynamic model system** that automatically discovers and supports new models released on OpenRouter. When a new model is released, it should become available automatically within 24 hours (or immediately with manual cache refresh).

### Files that may be affected:

- `data/cache/models.json` - Model cache (automatically updated)
- `src/models/model_registry.py` - Static fallback models (only if needed)
- `config/models/` - Model-specific configurations (only if needed)
- Tests files - If new model requires specific testing

### Automatic Process (Typical Case)

**🎉 In most cases, NO manual work is required!** New OpenRouter models are automatically:

1. **Discovered** when cache expires (24 hours) or is manually refreshed
2. **Cached** in `data/cache/models.json` with metadata
3. **Available** in all CLI commands (`alex models list`, `alex benchmark run --list-models`)
4. **Ready for benchmarking** immediately

### Manual Steps (Only When Needed)

#### Step 1: Force Model Cache Refresh

```bash
# Immediately fetch the latest models from OpenRouter API
alex models refresh
```

This command:

- Fetches all available models from OpenRouter API
- Updates the local cache with new models and metadata
- Displays confirmation of models fetched

#### Step 2: Verify New Model Availability

```bash
# Search for the new model by name or provider
alex models search "new-model-name"

# Or list all models to browse
alex models list

# Get detailed info about the specific model
alex models info "provider/model-id"
```

#### Step 3: Test Model Functionality

```bash
# Quick functionality test with a simple prompt
alex models test "provider/model-id" --prompt "What is 2+2?"

# Small benchmark test to verify full integration
alex benchmark run --model "provider/model-id" --size quick
```

#### Step 4: Verify Model in Benchmark Lists

```bash
# Verify model appears in benchmark model list
alex benchmark run --list-models

# Look for the new model in the output table
```

### Advanced Configuration (Rare Cases)

Only needed if the new model has special requirements:

#### Add Static Fallback (Emergency Only)

**When needed:** If model is critical and API access fails frequently

**Files to modify:** `src/models/model_registry.py`

```python
# Add to MODELS dictionary in ModelRegistry class
"provider/new-model-id": ModelConfig(
    model_id="provider/new-model-id",
    display_name="New Model Display Name",
    provider=ModelProvider.PROVIDER_NAME,
    context_window=128000,  # Check OpenRouter docs
    input_cost_per_1m_tokens=3.0,  # Check OpenRouter pricing
    output_cost_per_1m_tokens=15.0,  # Check OpenRouter pricing
    supports_streaming=True,
    capabilities=["chat", "reasoning", "analysis"]
),
```

#### Add Model-Specific Configuration

**When needed:** If model needs special prompt formatting or parameters

**Files to modify:** Create `config/models/provider-name.yaml`

```yaml
# Example: config/models/new-provider.yaml
models:
  provider/new-model-id:
    temperature: 0.1
    max_tokens: 150
    special_params:
      some_param: value
    prompt_template: "custom"
```

### Validation Checklist

After adding a new model (manually or automatically), verify:

- [ ] Model appears in `alex models list`
- [ ] Model details are correct in `alex models info MODEL_ID`
- [ ] Model can be tested with `alex models test MODEL_ID`
- [ ] Model works in quick benchmark: `alex benchmark run --model MODEL_ID --size quick`
- [ ] Cost estimates are reasonable
- [ ] Model appears in comparison commands
- [ ] Documentation is updated if model is significant

### Troubleshooting

**Problem: Model not appearing after OpenRouter release**

```bash
# Force refresh the cache
alex models refresh

# Check cache status
alex models cache --info

# Clear cache if corrupted
alex models cache --clear
alex models refresh
```

**Problem: Model appears but benchmark fails**

```bash
# Test model connection first
alex models test "model-id" --prompt "Hello"

# Check OpenRouter API key
alex health --check-api

# Try with different model parameters
alex models info "model-id"  # Check recommended settings
```

**Problem: Model costs seem wrong**

```bash
# Check cost estimates
alex models costs --model "model-id" --questions 10

# Compare with OpenRouter pricing page
alex models info "model-id"  # Check pricing section
```

### Important Notes

- **Three-tier fallback system**: API → Cache → Static backup ensures reliability
- **Cache TTL**: 24 hours by default, configurable in `config/default.yaml`
- **Automatic discovery**: New models should appear without manual intervention
- **OpenRouter dependency**: System requires valid OpenRouter API key
- **Rate limits**: OpenRouter API has rate limits (60 requests/minute default)

### Emergency Fallback

If the dynamic system fails completely:

1. Check OpenRouter API status
2. Use static fallback models (limited set)
3. Report system issues for investigation
4. Consider using cached models until issue resolved

---

_This task workflow was documented on: 2025-08-31_
_Dynamic model system supports 323+ models as of documentation date_
