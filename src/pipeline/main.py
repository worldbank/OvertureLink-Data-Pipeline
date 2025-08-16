# src/pipeline/main.py

# Import secure configuration system
from o2agol.config.settings import Config, ConfigurationError


def create_secure_gis_connection():
    """
    Create ArcGIS connection using secure configuration system.
    
    What's happening:
    1. Config class loads credentials from .env file with validation
    2. No credentials are hardcoded in the source
    3. Support for different .env files for dev/staging/production
    4. Comprehensive error handling and validation
    5. World Bank compliant security standards
    
    Returns:
        Authenticated GIS connection object
        
    Raises:
        ConfigurationError: If credentials are missing or invalid
    """
    try:
        # Use secure configuration system
        config = Config()
        
        # Use the secure connection method
        gis = config.create_gis_connection()
        
        print(f"Connected to: {gis.properties.portalName}")
        print(f"   User: {gis.users.me.username}")
        print(f"   Organization: {gis.properties.name}")
        print(f"   Environment: {config.environment}")
        
        return gis
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        print("\nCheck your .env file has:")
        print("   AGOL_PORTAL_URL=https://geowb.maps.arcgis.com")
        print("   AGOL_USERNAME=your_username")
        print("   AGOL_PASSWORD=your_password")
        raise
    except Exception as e:
        print(f"Connection failed: {e}")
        raise

def get_pipeline_configuration():
    """
    Get complete pipeline configuration including DuckDB and Overture settings.
    
    Returns:
        Dict containing all pipeline configuration settings
    """
    config = Config()
    
    pipeline_config = {
        'gis': config.create_gis_connection(),
        'duckdb_settings': config.get_duckdb_settings(),
        'overture_config': {
            'base_url': config.overture.base_url,
            's3_region': config.overture.s3_region,
            'release': config.overture.release
        },
        'environment': config.environment,
        'security_summary': config.get_security_summary()
    }
    
    print("Pipeline Configuration:")
    print(f"   Environment: {pipeline_config['environment']}")
    print(f"   DuckDB Memory: {pipeline_config['duckdb_settings']['memory_limit']}")
    print(f"   DuckDB Threads: {pipeline_config['duckdb_settings']['threads']}")
    print(f"   Overture Release: {pipeline_config['overture_config']['release']}")
    
    return pipeline_config

def test_configuration():
    """
    Test the complete configuration system for debugging purposes.
    """
    print("Testing Secure Configuration System")
    print("=" * 50)
    
    try:
        # Test basic configuration loading
        config = Config()
        print("Configuration loaded successfully")
        
        # Test GIS connection
        create_secure_gis_connection()
        print("GIS connection successful")
        
        # Test configuration access
        duckdb_settings = config.get_duckdb_settings()
        print(f"DuckDB settings: {duckdb_settings}")
        
        # Test security summary
        config.get_security_summary()
        print("Security validation passed")
        
        print("\nAll configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"\nConfiguration test failed: {e}")
        return False

# Updated usage examples
if __name__ == "__main__":
    # Test the configuration system
    if test_configuration():
        print("\nReady to run pipeline!")
        
        # Example: Get complete pipeline configuration
        pipeline_config = get_pipeline_configuration()
        
        # Example: Access specific components
        gis = pipeline_config['gis']
        duckdb_settings = pipeline_config['duckdb_settings']
        overture_base_url = pipeline_config['overture_config']['base_url']
        
        print("\nExample usage:")
        print(f"   GIS Portal: {gis.properties.portalName}")
        print(f"   DuckDB Memory: {duckdb_settings['memory_limit']}")
        print(f"   Overture URL: {overture_base_url}")
    else:
        print("\nFix configuration issues before running pipeline")