#!/usr/bin/env python3
"""
Alex-treBENCH CLI Entry Point

This file serves as the main entry point for the alex command-line tool.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Import and run the CLI
from main import cli

if __name__ == '__main__':
    cli()