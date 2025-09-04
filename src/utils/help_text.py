from rich.console import Console
from rich.markdown import Markdown

console = Console()

def show_help_with_markdown(ctx, param, value):
    """Custom help callback that renders help text using Rich markdown"""
    if not value or ctx.resilient_parsing:
        return
    
    markdown_help = """
# alex-treBENCH - Jeopardy Benchmarking System for LLMs

## 🚀 QUICK START EXAMPLES

```bash
alex benchmark run --model anthropic/claude-3-5-sonnet --size quick
alex benchmark compare --models "openai/gpt-4,anthropic/claude-3-5-sonnet"
alex models list
alex benchmark report --run-id 1 --format markdown
alex health
```

## 💡 TIP
- Use `alex COMMAND --help` for detailed options on any command.
- For complete documentation, see: **docs/USER_GUIDE.md**

## Options
- `--config, -c PATH`: Configuration file path
- `--verbose, -v`: Enable verbose logging
- `--debug`: Enable debug mode
- `--help`: Show this message and exit

## Commands
- benchmark  Benchmark management commands.
- config     Configuration management commands.
- data       Data management commands.
- database   Database management commands.
- health     Check system health and connectivity.
- models     Model management commands.
- session    Session management commands.
"""
    
    console.print(Markdown(markdown_help))
    ctx.exit()
