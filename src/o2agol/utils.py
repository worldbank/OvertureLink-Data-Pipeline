"""
Consolidated Utilities

This module consolidates helper functions from various modules to reduce code duplication
and provide a central location for common utilities.

Sections:
- Logging and timing utilities (structlog-backed since hardening pass Phase 1)
- Filesystem and path operations
- DataFrame chunking for batch operations
- Geometry and bbox helpers
- Retry and backoff mechanisms
- Configuration helpers
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from collections.abc import Callable, Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

# =============================================================================
# Structured Logging (structlog) — hardening pass Phase 1
# =============================================================================
#
# Architecture:
#
#   ┌────────────────────────────────────────────────────────────────────┐
#   │  CLI command entry  ──▶  setup_logging(...)  ──▶  configures both  │
#   │                                                    structlog AND   │
#   │                                                    stdlib logging  │
#   │                                                                    │
#   │  bind_run_context(country=..., theme=..., release=...)             │
#   │           │                                                        │
#   │           ▼                                                        │
#   │  contextvars  ──▶  every downstream log line auto-carries          │
#   │                    {country, theme, release} via merge_contextvars │
#   │                                                                    │
#   │  Pipeline stages call  log_stage("source.read", in_count=N, ...)   │
#   │           │                                                        │
#   │           ▼                                                        │
#   │  structlog processors:                                             │
#   │    1. merge_contextvars  (inject country/theme/release)            │
#   │    2. add_log_level                                                │
#   │    3. TimeStamper                                                  │
#   │    4. _redact_secrets   (deny-list scrub)                          │
#   │    5. JSONRenderer       (one-line JSON output)                    │
#   │           │                                                        │
#   │           ▼                                                        │
#   │  stdout (and optional log file) — one JSON object per line        │
#   └────────────────────────────────────────────────────────────────────┘
#
# Existing `logging.info(...)` calls scattered through pipeline/*.py and
# config/*.py automatically flow through the same JSON formatter because
# stdlib logging is configured with the structlog ProcessorFormatter.

# Deny-list of sensitive field-name substrings (case-insensitive).
# Any log field whose key contains one of these strings has its value replaced
# with the literal string "[REDACTED]" before emission.
_SECRET_KEYS: frozenset[str] = frozenset({
    "password",
    "agol_password",
    "client_secret",
    "agol_client_secret",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "secret",
})


def _redact_value(key: str, value: Any) -> Any:
    """Recursively scrub sensitive fields from a value."""
    key_low = str(key).lower()
    if any(s in key_low for s in _SECRET_KEYS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: _redact_value(k, v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        # Lists themselves don't have keys; only redact dict members inside.
        return type(value)(
            _redact_value("", v) if not isinstance(v, dict) else _redact_value("dict", v)
            for v in value
        )
    return value


def _redact_secrets(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: replace values of sensitive fields with [REDACTED].

    Runs over the full event dict at emission time. Field name matching is
    case-insensitive substring; nested dicts are walked recursively.
    """
    return {k: _redact_value(k, v) for k, v in event_dict.items()}


def setup_logging(
    verbose: bool = False,
    target_name: Optional[str] = None,
    mode: Optional[str] = None,
    enable_file_logging: bool = False,
    trace: bool = False,
) -> None:
    """Single entry point for pipeline logging.

    Configures structlog AND stdlib logging so existing ``logging.info(...)``
    calls in the codebase automatically render as structured JSON.

    Args:
        verbose: Enable DEBUG-level logging.
        target_name: Target data type used in the optional log filename.
        mode: Operation mode used in the optional log filename.
        enable_file_logging: When True, also write JSON lines to ``logs/<name>_<mode>_<ts>.log``.
        trace: Enable DEBUG-level logging AND set ``O2AGOL_TRACE=1`` so downstream
            code can emit additional ``stage_enter``/``stage_exit`` events.
    """
    if trace:
        os.environ["O2AGOL_TRACE"] = "1"
    level = logging.DEBUG if (verbose or trace) else logging.INFO

    # Build handler list (stdout always; optional file).
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if enable_file_logging and target_name and mode:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{target_name}_{mode}_{timestamp}.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        # Keep this print so operators see the file path; structured logs go through stdout.
        print(f"Logging to: {log_file}")

    # Shared processor chain for native structlog calls (log_stage, etc.)
    # AND for stdlib logging routed through ProcessorFormatter.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _redact_secrets,
    ]

    # Configure structlog itself.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to also emit through the structlog JSON formatter,
    # so existing `logging.info(...)` calls in pipeline/*.py emit structured output
    # without rewriting them.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    for handler in handlers:
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(level)


def log_stage(stage: str, **fields: Any) -> None:
    """Emit one structured log line at a pipeline stage boundary.

    Canonical fields auto-bound from contextvars (set by ``bind_run_context``):
        country, theme, release

    Stage-specific fields passed as kwargs (typical):
        in_count, out_count, duration_ms, bytes_read, ...

    Example:
        >>> bind_run_context(country="afg", theme="roads", release="2025-07-23.0")
        >>> log_stage("source.read", in_count=12381, out_count=12340, duration_ms=842)
        # Emits:
        # {"event":"source.read","country":"afg","theme":"roads",
        #  "release":"2025-07-23.0","in_count":12381,"out_count":12340,
        #  "duration_ms":842,"level":"info","timestamp":"2026-04-08T..."}
    """
    structlog.get_logger().info(stage, **fields)


def bind_run_context(*, country: str, theme: str, release: str, **extra: Any) -> None:
    """Bind canonical run context so every downstream log line carries it.

    Call this once at command entry. Subsequent ``log_stage()`` calls and any
    stdlib ``logging.info(...)`` calls will automatically carry country/theme/release.
    """
    bind_contextvars(country=country, theme=theme, release=release, **extra)


def clear_run_context() -> None:
    """Clear any previously bound run context. Use between independent runs in the same process."""
    clear_contextvars()


class StageTimer:
    """Context manager that emits a structured log line on exit with elapsed duration_ms.

    Use this to bracket pipeline sub-stages without writing the elapsed-time math
    by hand at every call site.

    Example:
        with StageTimer("source.query_s3", in_country="afg") as t:
            result = self._run_duckdb_query(...)
            t.add(rows_returned=len(result))
        # On exit, emits:
        # {"event":"source.query_s3","country":"afg",..., "in_country":"afg",
        #  "rows_returned":41982,"duration_ms":3217,"level":"info","timestamp":...}
    """

    def __init__(self, stage: str, **fields: Any) -> None:
        self.stage = stage
        self.fields: dict[str, Any] = dict(fields)
        self._start: float = 0.0

    def add(self, **fields: Any) -> None:
        """Attach additional fields before the stage exits (e.g. row counts known mid-stage)."""
        self.fields.update(fields)

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        if os.environ.get("O2AGOL_TRACE") == "1":
            structlog.get_logger().debug(f"{self.stage}.enter", **self.fields)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_ms = int((time.perf_counter() - self._start) * 1000)
        self.fields["duration_ms"] = duration_ms
        if exc_type is not None:
            self.fields["error"] = repr(exc_val)
            structlog.get_logger().error(self.stage, **self.fields)
        else:
            structlog.get_logger().info(self.stage, **self.fields)


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time function execution.

    Legacy helper kept for backward compatibility with existing call sites.
    Prefer ``StageTimer`` for new code.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


# =============================================================================
# Filesystem and Path Operations
# =============================================================================

def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_dir() -> Path:
    """Get temporary directory for pipeline operations."""
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "overture_pipeline"
    return ensure_directory(temp_dir)


def get_project_temp_dir() -> Path:
    """Get the project's temp directory path."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "temp"


def get_pid_temp_dir() -> Path:
    """Get process-isolated temp directory for current PID."""
    import os
    temp_dir = get_project_temp_dir()
    pid_dir = temp_dir / f"pid_{os.getpid()}"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir


def clean_filename(filename: str) -> str:
    """
    Clean filename for cross-platform compatibility.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename safe for all platforms
    """
    import re
    # Replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Trim underscores from ends
    return cleaned.strip('_')


# =============================================================================
# DataFrame Chunking for Batch Operations
# =============================================================================

def chunk_dataframe(df: gpd.GeoDataFrame, chunk_size: int) -> Generator[gpd.GeoDataFrame, None, None]:
    """
    Split GeoDataFrame into chunks for batch processing.
    
    Args:
        df: GeoDataFrame to chunk
        chunk_size: Number of features per chunk
        
    Yields:
        GeoDataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size].copy()


def estimate_chunk_size(df: gpd.GeoDataFrame, max_memory_mb: int = 500) -> int:
    """
    Estimate optimal chunk size based on DataFrame memory usage.
    
    Args:
        df: GeoDataFrame to analyze
        max_memory_mb: Maximum memory per chunk in MB
        
    Returns:
        Estimated chunk size
    """
    if len(df) == 0:
        return 1000  # Default chunk size
        
    # Estimate memory per row in MB
    sample_size = min(1000, len(df))
    sample_memory_mb = df.head(sample_size).memory_usage(deep=True).sum() / (1024 * 1024)
    memory_per_row = sample_memory_mb / sample_size
    
    # Calculate chunk size to stay under memory limit
    chunk_size = int(max_memory_mb / memory_per_row) if memory_per_row > 0 else 1000
    
    # Ensure reasonable bounds
    return max(100, min(chunk_size, 50000))


# =============================================================================
# Geometry and Bbox Helpers
# =============================================================================

def validate_bbox(bbox: list[float]) -> bool:
    """
    Validate bounding box coordinates.
    
    Args:
        bbox: Bounding box as [minx, miny, maxx, maxy]
        
    Returns:
        True if valid, False otherwise
    """
    if len(bbox) != 4:
        return False
        
    minx, miny, maxx, maxy = bbox
    
    # Check longitude bounds
    if not (-180 <= minx <= 180) or not (-180 <= maxx <= 180):
        return False
        
    # Check latitude bounds  
    if not (-90 <= miny <= 90) or not (-90 <= maxy <= 90):
        return False
        
    # Check min < max
    return not (minx >= maxx or miny >= maxy)


def expand_bbox(bbox: list[float], buffer_degrees: float = 0.1) -> list[float]:
    """
    Expand bounding box by buffer amount.
    
    Args:
        bbox: Original bounding box [minx, miny, maxx, maxy]
        buffer_degrees: Buffer amount in degrees
        
    Returns:
        Expanded bounding box
    """
    minx, miny, maxx, maxy = bbox
    return [
        max(-180, minx - buffer_degrees),
        max(-90, miny - buffer_degrees), 
        min(180, maxx + buffer_degrees),
        min(90, maxy + buffer_degrees)
    ]


# =============================================================================
# Retry and Backoff Mechanisms  
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        backoff_factor: Multiplier for delay between attempts
        exceptions: Exception types to retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        logging.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                        
            raise last_exception
            
        return wrapper
    return decorator


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_yaml_file(file_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file with error handling.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Parsed YAML content as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid YAML
    """
    import yaml
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
    try:
        with open(file_path, encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {file_path}: {e}") from e


def load_json_file(file_path: Path) -> Any:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid JSON
    """
    import json
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
        
    try:
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e


def safe_get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Safely get environment variable with optional default.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    import os
    return os.getenv(key, default)