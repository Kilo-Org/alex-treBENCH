"""
Models Command Group

Model management commands for alex-treBENCH.
"""

import click
from .list import models_list
from .search import models_search
from .info import models_info
from .refresh import models_refresh
from .cache import models_cache
from .test import models_test
from .costs import models_costs


@click.group()
def models():
    """Model management commands."""
    pass


# Register all subcommands
models.add_command(models_list, name='list')
models.add_command(models_search, name='search')
models.add_command(models_info, name='info')
models.add_command(models_refresh, name='refresh')
models.add_command(models_cache, name='cache')
models.add_command(models_test, name='test')
models.add_command(models_costs, name='costs')