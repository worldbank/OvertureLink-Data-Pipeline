"""
Secure configuration management for Overture Maps pipeline.
World Bank compliant credential and settings management.

Usage:
    from o2agol.config.settings import Config
    config = Config()
    gis = config.create_gis_connection()

Environment Variables (AGOL_ standard):
    AGOL_PORTAL_URL: ArcGIS Online portal URL
    AGOL_USERNAME: Username for ArcGIS Online
    AGOL_PASSWORD: Password for ArcGIS Online
    AGOL_TOKEN_EXPIRATION: Token expiration in minutes (default: 9999 for large datasets)
    AGOL_LARGE_DATASET_THRESHOLD: Feature count threshold for large dataset detection (default: 10000)
    DUCKDB_MEMORY_LIMIT: Memory limit for DuckDB
    DUCKDB_THREADS: Number of threads for DuckDB
"""

import hashlib
import logging
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from arcgis.gis import GIS
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class LogContext(Enum):
    """Context-aware logging levels for pipeline operations."""
    PHASE = "PHASE"           # Major pipeline phases
    MILESTONE = "MILESTONE"   # Key completion points
    OPERATION = "OPERATION"   # Atomic operations
    DEBUG = "DEBUG"          # Technical implementation details


@dataclass
class LogEntry:
    """Structured log entry with deduplication support."""
    message: str
    context: LogContext
    source: str  # Module/function origin
    data: Optional[dict[str, Any]] = None
    _hash: Optional[str] = None  # For deduplication
    
    def __post_init__(self):
        """Generate hash for deduplication."""
        if self._hash is None:
            hash_content = f"{self.message}:{self.context.value}:{self.source}"
            self._hash = hashlib.md5(hash_content.encode()).hexdigest()


class PipelineLogger:
    """Enhanced logger with deduplication and context awareness."""
    
    def __init__(self, base_logger: logging.Logger, window_size: int = 10):
        self.logger = base_logger
        self.window_size = window_size
        self.recent_hashes: deque = deque(maxlen=window_size)
        self._message_counts: dict[str, int] = {}
    
    def should_log(self, entry: LogEntry) -> bool:
        """Determine if message should be logged based on deduplication rules."""
        if entry._hash in self.recent_hashes:
            self._message_counts[entry._hash] = self._message_counts.get(entry._hash, 0) + 1
            return False
        return True
    
    def log(self, context: LogContext, message: str, source: str = "", data: Optional[dict] = None):
        """Context-aware logging with deduplication."""
        entry = LogEntry(message, context, source, data)
        
        if not self.should_log(entry):
            return
        
        # Add to recent messages
        self.recent_hashes.append(entry._hash)
        
        # Log at appropriate level based on context
        if context in [LogContext.PHASE, LogContext.MILESTONE] or context == LogContext.OPERATION:
            self.logger.info(message)
        else:  # DEBUG
            self.logger.debug(message)
    
    def info(self, message: str, source: str = ""):
        """Standard INFO logging with deduplication."""
        self.log(LogContext.OPERATION, message, source)
    
    def debug(self, message: str, source: str = ""):
        """Standard DEBUG logging with deduplication."""
        self.log(LogContext.DEBUG, message, source)
    
    def phase(self, message: str, source: str = ""):
        """Phase transition logging."""
        self.log(LogContext.PHASE, message, source)
    
    def milestone(self, message: str, source: str = ""):
        """Milestone completion logging."""
        self.log(LogContext.MILESTONE, message, source)


@dataclass
class AGOLCredentials:
    """ArcGIS Online credential configuration."""
    portal_url: str
    username: str
    password: str
    token_expiration: int = 9999  # Extended default for large datasets
    large_dataset_threshold: int = 10000  # Feature count threshold
    
    def __post_init__(self):
        """Validate credential format."""
        if not self.portal_url.startswith(('http://', 'https://')):
            raise ValueError("Portal URL must include protocol (https://)")
        
        if not self.username:
            raise ValueError("Username cannot be empty")
        
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        if self.token_expiration < 1:
            raise ValueError("Token expiration must be positive")
        
        if self.large_dataset_threshold < 1:
            raise ValueError("Large dataset threshold must be positive")


@dataclass
class OvertureConfig:
    """Overture Maps data source configuration."""
    base_url: str
    release: str
    s3_region: str
    
    def __post_init__(self):
        """Validate Overture Maps configuration."""
        if not self.base_url.startswith(('http://', 'https://', 's3://')):
            raise ValueError("Base URL must include valid protocol")
        
        if not self.release:
            raise ValueError("Release version cannot be empty")


@dataclass
class TempConfig:
    """Temporary file management configuration."""
    retention_hours: int = 24
    warning_gb: int = 10
    limit_gb: int = 20
    
    def __post_init__(self):
        """Validate temp management configuration."""
        if self.retention_hours < 0:
            raise ValueError("Retention hours must be non-negative")
        if self.warning_gb < 1:
            raise ValueError("Warning threshold must be at least 1GB")
        if self.limit_gb <= self.warning_gb:
            raise ValueError("Size limit must be greater than warning threshold")


@dataclass
class ProcessingConfig:
    """DuckDB processing configuration."""
    memory_limit: str
    threads: int
    temp_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate processing configuration."""
        if self.threads < 1:
            raise ValueError("Thread count must be positive")
        
        # Validate memory limit format
        if not any(self.memory_limit.endswith(unit) for unit in ['MB', 'GB', 'TB']):
            raise ValueError("Memory limit must end with MB, GB, or TB")


@dataclass 
class DumpConfig:
    """Configuration for local Overture dump operations."""
    base_path: str = "./overturedump"  # Relative to current working directory
    max_memory_gb: int = 32
    chunk_size: int = 5  # Countries per processing chunk
    enable_spatial_index: bool = True
    compression: str = "zstd"
    partitioning: str = "hive"
    
    def __post_init__(self):
        """Validate dump configuration."""
        if self.max_memory_gb < 1:
            raise ValueError("Memory limit must be at least 1GB")
        if self.chunk_size < 1:
            raise ValueError("Chunk size must be positive")
        if self.compression.lower() not in ['zstd', 'gzip', 'snappy', 'lz4']:
            raise ValueError("Compression must be one of: zstd, gzip, snappy, lz4")
        if self.partitioning not in ['hive', 'none']:
            raise ValueError("Partitioning must be 'hive' or 'none'")


class ConfigurationError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


class Config:
    """
    Centralized configuration management for Overture Maps pipeline.
    
    Handles secure credential loading and validation following World Bank 
    security standards. Pipeline configuration (Overture settings, targets)
    is managed through YAML files.
    
    Environment variables loaded (in order of preference):
    1. Explicit environment file passed to constructor
    2. .env.{ENVIRONMENT} (where ENVIRONMENT=development|production|staging)
    3. .env file in project root
    4. System environment variables
    
    Example:
        # Development environment
        config = Config(environment="development")
        
        # Production with explicit env file
        config = Config(env_file=Path("/secure/production.env"))
        
        # Auto-detect from ENVIRONMENT variable
        config = Config()  # Uses $ENVIRONMENT or defaults to 'development'
    """
    
    # Class variables for GIS connection singleton
    _gis_connection: Optional[GIS] = None
    _connection_logged: bool = False
    _pipeline_logger: Optional[PipelineLogger] = None
    
    def __init__(self, 
                 environment: Optional[str] = None,
                 env_file: Optional[Path] = None,
                 validate_on_init: bool = True):
        """
        Initialize configuration with secure credential loading.
        
        Args:
            environment: Target environment (development|staging|production)
            env_file: Explicit path to environment file
            validate_on_init: Whether to validate all settings on initialization
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.project_root = self._find_project_root()
        
        # Load environment variables
        self._load_environment_variables(env_file)
        
        # Initialize configuration sections
        if validate_on_init:
            self._load_agol_config()
        self._load_overture_config()
        self._load_processing_config()
        self._load_temp_config()
        self._load_dump_config()
        
        if validate_on_init:
            self.validate()
    
    def _find_project_root(self) -> Path:
        """Find project root directory containing setup.py, pyproject.toml, or .env file."""
        current = Path(__file__).resolve()
        
        # First, try to find standard project markers
        for parent in current.parents:
            if any((parent / marker).exists() for marker in ['setup.py', 'pyproject.toml', '.git']):
                return parent
        
        # If no standard markers found, look for .env file
        for parent in current.parents:
            if (parent / '.env').exists():
                return parent
        
        # Fallback to current working directory if .env exists there
        if (Path.cwd() / '.env').exists():
            return Path.cwd()
        
        # Final fallback to current directory
        return Path.cwd()
    
    def _load_environment_variables(self, env_file: Path | None) -> None:
        """Load environment variables from appropriate source."""
        loaded_files = []
        
        if env_file:
            # Explicit environment file
            if env_file.exists():
                load_dotenv(env_file)
                loaded_files.append(str(env_file))
                logger.info(f"Loaded configuration from {env_file}")
            else:
                raise ConfigurationError(f"Specified env file not found: {env_file}")
        
        else:
            # Try environment-specific file first
            env_specific_file = self.project_root / f".env.{self.environment}"
            logger.info(f"Looking for environment config at: {env_specific_file}")
            if env_specific_file.exists():
                logger.info(f"Loading environment config from: {env_specific_file}")
                load_dotenv(env_specific_file, override=True)
                loaded_files.append(str(env_specific_file))
                logger.info(f"Loaded environment-specific config: {env_specific_file}")
            else:
                logger.info(f"Environment-specific file not found: {env_specific_file}")
            
            # Then try generic .env file
            generic_env_file = self.project_root / ".env"
            logger.info(f"Looking for generic config at: {generic_env_file}")
            if generic_env_file.exists():
                logger.info(f"Loading generic config from: {generic_env_file}")
                load_dotenv(generic_env_file, override=True)
                loaded_files.append(str(generic_env_file))
                logger.info(f"Loaded generic config: {generic_env_file}")
            else:
                logger.warning(f"No .env file found at {generic_env_file}")
        
        if not loaded_files:
            logger.warning("No .env files found, using system environment variables only")
        
        # Store for debugging
        self._loaded_env_files = loaded_files
        self._project_root_used = str(self.project_root)
        
        # Debug logging
        logger.debug(f"Project root: {self.project_root}")
        logger.debug(f"Loaded env files: {loaded_files}")
        logger.debug(f"Environment: {self.environment}")
    
    def _load_agol_config(self) -> None:
        """Load and validate ArcGIS configuration."""
        # Use AGOL_ as primary standard, ARCGIS_ as legacy fallback
        portal_url = os.getenv("AGOL_PORTAL_URL") or os.getenv("ARCGIS_PORTAL_URL")
        username = os.getenv("AGOL_USERNAME") or os.getenv("ARCGIS_USERNAME")
        password = os.getenv("AGOL_PASSWORD") or os.getenv("ARCGIS_PASSWORD")
        
        # Timeout configuration with sensible defaults
        token_expiration = int(os.getenv("AGOL_TOKEN_EXPIRATION", "9999"))
        large_dataset_threshold = int(os.getenv("AGOL_LARGE_DATASET_THRESHOLD", "10000"))
        
        if not all([portal_url, username, password]):
            missing = []
            if not portal_url:
                missing.append("AGOL_PORTAL_URL")
            if not username:
                missing.append("AGOL_USERNAME")
            if not password:
                missing.append("AGOL_PASSWORD")
            
            raise ConfigurationError(
                f"Missing required ArcGIS Online credentials: {', '.join(missing)}.\n"
                f"Please set these variables in your .env file:\n"
                f"  AGOL_PORTAL_URL=https://geowb.maps.arcgis.com\n"
                f"  AGOL_USERNAME=your_username\n"
                f"  AGOL_PASSWORD=your_password\n"
                f"  AGOL_TOKEN_EXPIRATION=9999  # Optional: Extended timeout for large datasets\n"
                f"  AGOL_LARGE_DATASET_THRESHOLD=10000  # Optional: Feature count threshold\n\n"
                f"Current .env file: {self.project_root / '.env'}"
            )
        
        try:
            self.agol = AGOLCredentials(
                portal_url=portal_url,
                username=username,
                password=password,
                token_expiration=token_expiration,
                large_dataset_threshold=large_dataset_threshold
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid ArcGIS configuration: {e}")
    
    def _load_overture_config(self) -> None:
        """Load Overture Maps configuration with sensible defaults."""
        base_url = os.getenv("OVERTURE_BASE_URL", "s3://overturemaps-us-west-2/release")
        release = os.getenv("OVERTURE_RELEASE", "2025-07-23.0")
        logger.info(f"Config loading OVERTURE_RELEASE from env: {release}")
        s3_region = os.getenv("OVERTURE_S3_REGION", "us-west-2")
        
        try:
            self.overture = OvertureConfig(
                base_url=base_url,
                release=release,
                s3_region=s3_region
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid Overture configuration: {e}")
    
    def _load_processing_config(self) -> None:
        """Load DuckDB processing configuration."""
        memory_limit = os.getenv("DUCKDB_MEMORY_LIMIT", "8GB")
        threads = int(os.getenv("DUCKDB_THREADS", "8"))
        temp_dir = os.getenv("DUCKDB_TEMP_DIR")
        
        try:
            self.processing = ProcessingConfig(
                memory_limit=memory_limit,
                threads=threads,
                temp_dir=temp_dir
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid processing configuration: {e}")
    
    def _load_temp_config(self) -> None:
        """Load temporary file management configuration."""
        retention_hours = int(os.getenv("TEMP_RETENTION_HOURS", "24"))
        warning_gb = int(os.getenv("TEMP_SIZE_WARNING_GB", "10"))
        limit_gb = int(os.getenv("TEMP_SIZE_LIMIT_GB", "20"))
        
        try:
            self.temp = TempConfig(
                retention_hours=retention_hours,
                warning_gb=warning_gb,
                limit_gb=limit_gb
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid temp management configuration: {e}")
    
    def _load_dump_config(self) -> None:
        """Load local dump configuration."""
        base_path = os.getenv("OVERTURE_DUMP_PATH", "./overturedump")
        max_memory_gb = int(os.getenv("DUMP_MAX_MEMORY", "32"))
        chunk_size = int(os.getenv("DUMP_CHUNK_SIZE", "5"))
        enable_spatial_index = os.getenv("DUMP_ENABLE_SPATIAL_INDEX", "true").lower() == "true"
        compression = os.getenv("DUMP_COMPRESSION", "zstd")
        partitioning = os.getenv("DUMP_PARTITIONING", "hive")
        
        try:
            self.dump = DumpConfig(
                base_path=base_path,
                max_memory_gb=max_memory_gb,
                chunk_size=chunk_size,
                enable_spatial_index=enable_spatial_index,
                compression=compression,
                partitioning=partitioning
            )
        except ValueError as e:
            raise ConfigurationError(f"Invalid dump configuration: {e}")
    
    def create_gis_connection(self, expiration: Optional[int] = None) -> GIS:
        """
        Create authenticated ArcGIS Online connection with extended timeout support.
        Uses singleton pattern to avoid duplicate connections and logging.
        
        Args:
            expiration: Token expiration time in minutes (uses configured default if None)
        
        Returns:
            Authenticated GIS connection object
            
        Raises:
            ConfigurationError: If connection fails due to invalid credentials
        """
        # Initialize pipeline logger if not already done
        if Config._pipeline_logger is None:
            Config._pipeline_logger = PipelineLogger(logger)
        
        # Check if we have a valid existing connection
        if Config._gis_connection is not None:
            try:
                # Test connection validity
                _ = Config._gis_connection.users.me
                Config._pipeline_logger.debug("Reusing existing AGOL connection", "config.settings")
                return Config._gis_connection
            except:
                # Connection is no longer valid, create new one
                Config._gis_connection = None
                Config._connection_logged = False
        
        # Use provided expiration or fall back to configured default
        token_expiration = expiration if expiration is not None else self.agol.token_expiration
        
        try:
            gis = GIS(
                url=self.agol.portal_url,
                username=self.agol.username,
                password=self.agol.password,
                expiration=token_expiration  # Extended expiration for long-running operations
            )
            
            # Store connection and verify
            Config._gis_connection = gis
            user_info = gis.users.me
            
            # Log connection details only once
            if not Config._connection_logged:
                Config._pipeline_logger.info(f"Connected to AGOL as {user_info.username} ({gis.properties.name})", "config.settings")
                Config._connection_logged = True
            
            return gis
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to connect to ArcGIS Online: {e}. "
                f"Please verify your credentials and portal URL."
            )
    
    def get_duckdb_settings(self) -> dict[str, Any]:
        """
        Get DuckDB configuration settings as dictionary.
        
        Returns:
            Dictionary of DuckDB settings ready for connection setup
        """
        settings = {
            'memory_limit': self.processing.memory_limit,
            'threads': self.processing.threads,
        }
        
        if self.processing.temp_dir:
            settings['temp_directory'] = self.processing.temp_dir
        
        return settings
    
    def get_temp_settings(self) -> dict[str, Any]:
        """
        Get temp management configuration settings as dictionary.
        
        Returns:
            Dictionary of temp management settings
        """
        return {
            'retention_hours': self.temp.retention_hours,
            'warning_gb': self.temp.warning_gb,
            'limit_gb': self.temp.limit_gb
        }
    
    def get_dump_settings(self) -> dict[str, Any]:
        """
        Get dump configuration settings as dictionary.
        
        Returns:
            Dictionary of dump configuration settings
        """
        return {
            'base_path': self.dump.base_path,
            'max_memory_gb': self.dump.max_memory_gb,
            'chunk_size': self.dump.chunk_size,
            'enable_spatial_index': self.dump.enable_spatial_index,
            'compression': self.dump.compression,
            'partitioning': self.dump.partitioning
        }
    
    def validate(self, include_agol: bool = True) -> None:
        """
        Comprehensive configuration validation.
        
        Raises:
            ConfigurationError: If any configuration is invalid
        """
        validation_errors = []
        
        # Test ArcGIS connection
        if include_agol:
            try:
                gis = self.create_gis_connection()
                _ = gis.users.me  # Test API access
            except Exception as e:
                validation_errors.append(f"ArcGIS connection failed: {e}")
        
        # Validate processing configuration
        if self.processing.threads > 32:
            validation_errors.append("Thread count > 32 may cause performance issues")
        
        if validation_errors:
            raise ConfigurationError(
                "Configuration validation failed:\n" + 
                "\n".join(f"  - {error}" for error in validation_errors)
            )
        
        logger.info("Configuration validation passed")
    
    def get_security_summary(self) -> dict[str, Any]:
        """
        Get security configuration summary for audit purposes.
        
        Returns:
            Dictionary with security-relevant configuration info (no secrets)
        """
        return {
            'environment': self.environment,
            'loaded_env_files': self._loaded_env_files,
            'agol_portal': self.agol.portal_url,
            'agol_username': self.agol.username,
            'overture_base_url': self.overture.base_url,
            'overture_release': self.overture.release,
            'duckdb_memory': self.processing.memory_limit,
            'duckdb_threads': self.processing.threads,
            'dump_base_path': self.dump.base_path,
            'dump_max_memory': self.dump.max_memory_gb,
            'validation_status': 'passed'  # Only returned if validate() succeeded
        }
    
    def __repr__(self) -> str:
        """Safe string representation without credentials."""
        return (
            f"Config(environment={self.environment}, "
            f"portal={self.agol.portal_url}, "
            f"overture_release={self.overture.release})"
        )


# Factory functions for common use cases
def create_development_config() -> Config:
    """Create configuration for development environment."""
    return Config(environment="development")


def create_production_config(env_file: Path) -> Config:
    """Create configuration for production environment with explicit env file."""
    return Config(environment="production", env_file=env_file)


def create_config_from_environment() -> Config:
    """Create configuration using ENVIRONMENT variable or default to development."""
    return Config()


# Convenience function for backward compatibility
def get_gis_connection() -> GIS:
    """
    Get ArcGIS connection using default configuration.
    
    This function maintains backward compatibility with existing code
    while using the new secure configuration system.
    """
    config = create_config_from_environment()
    return config.create_gis_connection()