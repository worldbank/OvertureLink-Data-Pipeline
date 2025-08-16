"""Temporary file management for cloud-native pipeline operations."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Optional


def get_project_temp_dir() -> Path:
    """Get the project's temp directory path."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "temp"


def get_pid_temp_dir() -> Path:
    """Get process-isolated temp directory for current PID."""
    temp_dir = get_project_temp_dir()
    pid_dir = temp_dir / f"pid_{os.getpid()}"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir


def cleanup_stale_files(retention_hours: int = 24) -> int:
    """
    Remove temp files older than retention period.
    
    Args:
        retention_hours: Files older than this will be removed
        
    Returns:
        Number of files cleaned up
    """
    temp_dir = get_project_temp_dir()
    if not temp_dir.exists():
        return 0
        
    cutoff_time = time.time() - (retention_hours * 3600)
    cleaned_count = 0
    
    for item in temp_dir.rglob("*"):
        if item.is_file():
            try:
                if item.stat().st_mtime < cutoff_time:
                    item.unlink()
                    cleaned_count += 1
                    logging.debug(f"Cleaned stale temp file: {item}")
            except (OSError, PermissionError) as e:
                logging.warning(f"Could not remove stale file {item}: {e}")
                
    # Remove empty PID directories
    for pid_dir in temp_dir.glob("pid_*"):
        if pid_dir.is_dir() and not any(pid_dir.iterdir()):
            try:
                pid_dir.rmdir()
                logging.debug(f"Removed empty PID directory: {pid_dir}")
            except OSError:
                pass
                
    if cleaned_count > 0:
        logging.info(f"Cleaned up {cleaned_count} stale temp files (>{retention_hours}h)")
        
    return cleaned_count


def get_temp_dir_size() -> int:
    """Get total size of temp directory in bytes."""
    temp_dir = get_project_temp_dir()
    if not temp_dir.exists():
        return 0
        
    total_size = 0
    for item in temp_dir.rglob("*"):
        if item.is_file():
            try:
                total_size += item.stat().st_size
            except (OSError, PermissionError):
                pass
                
    return total_size


def check_temp_size_limits(warning_gb: int = 10, limit_gb: int = 20) -> bool:
    """
    Check temp directory size against limits.
    
    Args:
        warning_gb: Log warning when exceeding this size
        limit_gb: Trigger cleanup when exceeding this size
        
    Returns:
        True if within limits, False if over hard limit
    """
    size_bytes = get_temp_dir_size()
    size_gb = size_bytes / (1024 ** 3)
    
    if size_gb > limit_gb:
        logging.error(f"Temp directory size ({size_gb:.1f}GB) exceeds limit ({limit_gb}GB)")
        # Force cleanup of files older than 1 hour
        cleaned = cleanup_stale_files(retention_hours=1)
        logging.info(f"Emergency cleanup removed {cleaned} files")
        
        # Recheck size
        new_size_gb = get_temp_dir_size() / (1024 ** 3)
        if new_size_gb > limit_gb:
            logging.error(f"Temp directory still too large ({new_size_gb:.1f}GB) after cleanup")
            return False
            
    elif size_gb > warning_gb:
        logging.warning(f"Temp directory size ({size_gb:.1f}GB) exceeds warning threshold ({warning_gb}GB)")
        
    return True


def cleanup_current_pid() -> None:
    """Clean up temp files for current process."""
    pid_dir = get_project_temp_dir() / f"pid_{os.getpid()}"
    if pid_dir.exists():
        try:
            shutil.rmtree(pid_dir)
            logging.debug(f"Cleaned up PID temp directory: {pid_dir}")
        except OSError as e:
            logging.warning(f"Could not clean PID temp directory {pid_dir}: {e}")


def register_cleanup_handlers() -> None:
    """Register signal handlers for graceful cleanup on interruption."""
    def signal_handler(signum: int, frame) -> None:
        logging.info(f"Received signal {signum}, cleaning up temp files...")
        cleanup_current_pid()
        exit(1)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def ensure_temp_structure() -> None:
    """Ensure temp directory structure exists."""
    temp_dir = get_project_temp_dir()
    
    # Create main temp directories
    (temp_dir / "duckdb").mkdir(parents=True, exist_ok=True)
    (temp_dir / "cache").mkdir(parents=True, exist_ok=True)
    
    logging.debug(f"Temp directory structure ensured: {temp_dir}")


def full_cleanup_check(
    retention_hours: Optional[int] = None,
    warning_gb: Optional[int] = None, 
    limit_gb: Optional[int] = None,
    skip_cleanup: bool = False
) -> bool:
    """
    Perform comprehensive temp directory management.
    
    Args:
        retention_hours: Files older than this will be removed (uses config default if None)
        warning_gb: Log warning when exceeding this size (uses config default if None)
        limit_gb: Trigger emergency cleanup when exceeding this size (uses config default if None)
        skip_cleanup: Skip the cleanup (for debugging)
        
    Returns:
        True if temp directory is healthy, False if issues remain
    """
    # Load config defaults if not provided
    if any(param is None for param in [retention_hours, warning_gb, limit_gb]):
        try:
            from .config import Config
            config = Config(validate_on_init=False)
            temp_settings = config.get_temp_settings()
            retention_hours = retention_hours or temp_settings['retention_hours']
            warning_gb = warning_gb or temp_settings['warning_gb']
            limit_gb = limit_gb or temp_settings['limit_gb']
        except Exception:
            # Fallback to defaults if config loading fails
            retention_hours = retention_hours or 24
            warning_gb = warning_gb or 10
            limit_gb = limit_gb or 20
    
    ensure_temp_structure()
    
    if not skip_cleanup:
        cleanup_stale_files(retention_hours)
        
    return check_temp_size_limits(warning_gb, limit_gb)