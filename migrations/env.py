import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import Base
import app.models  # noqa: F401  (register tables on Base.metadata)

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.test_database_url or settings.database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
