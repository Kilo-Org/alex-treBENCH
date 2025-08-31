"""
Database Migration System

Handles database schema migrations using Alembic integration,
provides migration commands for CLI, and manages schema evolution.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from alembic import command
from alembic.config import Config
from alembic.environment import EnvironmentContext
from alembic.script import ScriptDirectory
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .models import Base
from src.core.database import get_db_url
from src.core.config import get_config
from src.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class MigrationManager:
    """Database migration manager with Alembic integration."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self.alembic_dir = Path("alembic")
        self.versions_dir = self.alembic_dir / "versions"
        self.config = self._create_alembic_config()

    def _create_alembic_config(self) -> Config:
        """Create Alembic configuration."""
        config = Config()
        config.set_main_option("script_location", str(self.alembic_dir))
        config.set_main_option("sqlalchemy.url", self.db_url)

        # Configure logging
        config.file_config = None

        return config

    def init_alembic(self) -> None:
        """Initialize Alembic directory structure."""
        if self.alembic_dir.exists():
            logger.info("Alembic already initialized")
            return

        try:
            # Create alembic directory structure
            self.alembic_dir.mkdir(parents=True, exist_ok=True)
            self.versions_dir.mkdir(parents=True, exist_ok=True)

            # Create alembic.ini
            self._create_alembic_ini()

            # Create env.py
            self._create_env_py()

            # Create script.py.mako
            self._create_script_py_mako()

            # Create initial migration
            self.create_initial_migration()

            logger.info("Alembic initialized successfully")

        except Exception as e:
            raise DatabaseError(f"Failed to initialize Alembic: {e}")

    def _create_alembic_ini(self) -> None:
        """Create alembic.ini configuration file."""
        ini_content = f"""# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = {self.alembic_dir}

# template used to generate migration files
# file_template = %%(rev)s_%%(slug)s

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment file as a
# subprocess, which has the benefit of preventing
# the alembic process from being polluted with
# the application's imports and code
# subprocess = false

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = {self.db_url}

[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" package
# hooks = black
# black.type = console_scripts
# black.entrypoint = black
# black.options = -l 79

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        ini_path = Path("alembic.ini")
        with open(ini_path, 'w') as f:
            f.write(ini_content)

    def _create_env_py(self) -> None:
        """Create env.py for Alembic."""
        env_content = '''import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        env_path = self.alembic_dir / "env.py"
        with open(env_path, 'w') as f:
            f.write(env_content)

    def _create_script_py_mako(self) -> None:
        """Create script.py.mako template."""
        script_content = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
'''
        script_path = self.alembic_dir / "script.py.mako"
        with open(script_path, 'w') as f:
            f.write(script_content)

    def create_initial_migration(self) -> None:
        """Create initial migration based on current models."""
        try:
            command.revision(self.config, message="Initial migration", autogenerate=True)
            logger.info("Initial migration created")
        except Exception as e:
            logger.error(f"Failed to create initial migration: {e}")
            raise

    def create_migration(self, message: str, autogenerate: bool = True) -> None:
        """Create a new migration."""
        try:
            command.revision(self.config, message=message, autogenerate=autogenerate)
            logger.info(f"Migration created: {message}")
        except Exception as e:
            raise DatabaseError(f"Failed to create migration: {e}")

    def upgrade(self, revision: str = "head") -> None:
        """Upgrade database to specified revision."""
        try:
            command.upgrade(self.config, revision)
            logger.info(f"Database upgraded to {revision}")
        except Exception as e:
            raise DatabaseError(f"Failed to upgrade database: {e}")

    def downgrade(self, revision: str = "-1") -> None:
        """Downgrade database to specified revision."""
        try:
            command.downgrade(self.config, revision)
            logger.info(f"Database downgraded to {revision}")
        except Exception as e:
            raise DatabaseError(f"Failed to downgrade database: {e}")

    def current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            script = ScriptDirectory.from_config(self.config)
            with self._get_engine().connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None

    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history."""
        try:
            script = ScriptDirectory.from_config(self.config)
            revisions = []

            for rev in script.walk_revisions():
                revisions.append({
                    'revision': rev.revision,
                    'down_revision': rev.down_revision,
                    'message': rev.doc,
                    'date': rev.date
                })

            return revisions
        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []

    def _get_engine(self) -> Engine:
        """Get SQLAlchemy engine."""
        return create_engine(self.db_url)

    def check_database_connection(self) -> bool:
        """Check if database is accessible."""
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False


# CLI Helper Functions
def cli_init_migrations() -> None:
    """CLI command to initialize migrations."""
    manager = MigrationManager()
    manager.init_alembic()
    print("Migrations initialized successfully")


def cli_create_migration(message: str) -> None:
    """CLI command to create a new migration."""
    manager = MigrationManager()
    manager.create_migration(message)
    print(f"Migration created: {message}")


def cli_upgrade_database(revision: str = "head") -> None:
    """CLI command to upgrade database."""
    manager = MigrationManager()
    manager.upgrade(revision)
    print(f"Database upgraded to {revision}")


def cli_downgrade_database(revision: str = "-1") -> None:
    """CLI command to downgrade database."""
    manager = MigrationManager()
    manager.downgrade(revision)
    print(f"Database downgraded to {revision}")


def cli_show_migration_status() -> None:
    """CLI command to show migration status."""
    manager = MigrationManager()

    if not manager.check_database_connection():
        print("❌ Database connection failed")
        return

    current = manager.current_revision()
    print(f"Current revision: {current or 'None'}")

    history = manager.get_migration_history()
    if history:
        print("\nMigration History:")
        for rev in reversed(history):
            print(f"  {rev['revision']}: {rev['message']} ({rev['date']})")
    else:
        print("No migrations found")


def cli_create_initial_migration() -> None:
    """CLI command to create initial migration."""
    manager = MigrationManager()
    manager.create_initial_migration()
    print("Initial migration created")