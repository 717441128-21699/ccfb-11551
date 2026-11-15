"""Mini SQL: a single-machine, columnar, batched SQL query engine.

No database or dataframe libraries are used; only numpy plus the standard
library.  The public entry point is :class:`mini_sql.engine.Engine`.
"""

from .engine import Engine, QueryResult, ExecutionStats  # noqa: F401
from .types import SQLError  # noqa: F401

__all__ = ["Engine", "QueryResult", "ExecutionStats", "SQLError"]
