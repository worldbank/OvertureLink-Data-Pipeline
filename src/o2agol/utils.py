"""
Consolidated Utilities

This module consolidates helper functions from various modules to reduce code duplication
and provide a central location for common utilities.

Sections:
- Logging and timing utilities
- Filesystem and path operations  
- DataFrame chunking for batch operations
- Geometry and bbox helpers
- Retry and backoff mechanisms
- Configuration helpers
"""

import functools
import logging
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd

# =============================================================================
# Logging and Timing Utilities
# =============================================================================

def setup_logging(
    verbose: bool, 
    target_name: Optional[str] = None, 
    mode: Optional[str] = None, 
    enable_file_logging: bool = False
) -> None:
    """
    Configure logging with optional timestamped file output.
    
    Args:
        verbose: Enable debug-level logging if True
        target_name: Target data type for log file naming
        mode: Operation mode for log file naming  
        enable_file_logging: Create timestamped log files when True
    """
    import sys
    from datetime import datetime
    
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if enable_file_logging and target_name and mode:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{target_name}_{mode}_{timestamp}.log"
        
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
        print(f"Logging to: {log_file}")
    
    logging.basicConfig(
        level=level, 
        format="%(asctime)s [%(levelname)s] %(message)s", 
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True
    )


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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


def get_pid_temp_dir() -> Path:
    """Get process-isolated temporary directory for pipeline operations."""
    import os
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "overture_pipeline" / f"pid_{os.getpid()}"
    return ensure_directory(temp_dir)


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