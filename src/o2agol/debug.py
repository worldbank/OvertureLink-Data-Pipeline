"""
Debug utilities for troubleshooting Overture Maps data and spatial queries.

This module contains diagnostic functions for analyzing Overture Divisions
data structure and country boundary queries. These functions are useful
for troubleshooting spatial filtering issues in production environments.
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb

from .config import Config
from .duck import setup_duckdb_optimized


def debug_divisions_structure(secure_config: Config) -> None:
    """
    Debug function to inspect the actual structure of divisions data.
    
    Useful for troubleshooting issues with Overture Divisions-based
    country boundary queries and understanding data schema changes.
    
    Args:
        secure_config: Secure configuration with Overture settings
        
    Logs:
        Available columns and sample country record from divisions data
    """
    con = setup_duckdb_optimized(secure_config)
    divisions_url = f"{secure_config.overture.base_url}/theme=divisions/type=division_area/*.parquet"
    
    try:
        # First, let's see what columns exist
        schema_sql = f"""
        SELECT * FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country'
        LIMIT 1
        """
        
        result = con.execute(schema_sql).fetchdf()
        logging.info("Available columns in divisions data:")
        logging.info(str(result.columns.tolist()))
        logging.info("Sample country record:")
        if not result.empty:
            logging.info(str(result.to_dict('records')[0]))
        else:
            logging.warning("No records found")
        
    except Exception as e:
        logging.error(f"Failed to query divisions structure: {e}")
    finally:
        con.close()


def debug_divisions_afghanistan(secure_config: Config) -> str:
    """
    Diagnostic query to examine Afghanistan entries in Overture divisions dataset.
    
    Returns SQL query that can be executed to troubleshoot Afghanistan-specific
    spatial boundary issues. Useful when debugging country filtering problems.
    
    Args:
        secure_config: Secure configuration with Overture settings
        
    Returns:
        SQL query string for Afghanistan divisions analysis
        
    Example:
        query = debug_divisions_afghanistan(config)
        con = setup_duckdb_optimized(config)
        result = con.execute(query).fetchdf()
        print(result)
    """
    divisions_url = f"{secure_config.overture.base_url}/theme=divisions/type=division_area/*.parquet"
    
    return f"""
    SELECT 
        country,
        names.primary as primary_name,
        names.common as common_name,
        subtype,
        ST_AREA(ST_GEOMFROMWKB(geometry)) as area_sq_degrees
    FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
    WHERE subtype = 'country' 
    AND (
        country ILIKE '%AF%' OR 
        country ILIKE '%Afghanistan%' OR
        names.primary ILIKE '%Afghanistan%' OR 
        names.common ILIKE '%Afghanistan%'
    )
    ORDER BY area_sq_degrees DESC
    LIMIT 10
    """


def execute_afghanistan_debug(secure_config: Config) -> Optional[dict]:
    """
    Execute Afghanistan divisions debug query and return results.
    
    Convenience function that runs the Afghanistan debug query and
    returns the results as a dictionary for analysis.
    
    Args:
        secure_config: Secure configuration with Overture settings
        
    Returns:
        Query results as dictionary, or None if query fails
        
    Logs:
        Debug information about Afghanistan divisions found
    """
    con = setup_duckdb_optimized(secure_config)
    
    try:
        query = debug_divisions_afghanistan(secure_config)
        result = con.execute(query).fetchdf()
        
        if result.empty:
            logging.warning("No Afghanistan divisions found in dataset")
            return None
        
        logging.info(f"Found {len(result)} Afghanistan division entries:")
        for _, row in result.iterrows():
            logging.info(f"  Country: {row['country']}, Name: {row.get('primary_name', 'N/A')}, Area: {row['area_sq_degrees']:.2f} sq degrees")
        
        return result.to_dict('records')
        
    except Exception as e:
        logging.error(f"Failed to execute Afghanistan debug query: {e}")
        return None
    finally:
        con.close()


def validate_country_divisions(secure_config: Config, iso2: str) -> bool:
    """
    Validate that divisions data exists for a specific country.
    
    Quick check to verify that Overture Divisions contains data
    for the specified country before attempting spatial queries.
    
    Args:
        secure_config: Secure configuration with Overture settings
        iso2: ISO2 country code to validate
        
    Returns:
        True if country divisions are found, False otherwise
        
    Logs:
        Validation results and any issues found
    """
    con = setup_duckdb_optimized(secure_config)
    divisions_url = f"{secure_config.overture.base_url}/theme=divisions/type=division_area/*.parquet"
    
    try:
        validation_sql = f"""
        SELECT COUNT(*) as count
        FROM read_parquet('{divisions_url}', filename=true, hive_partitioning=1)
        WHERE subtype = 'country' AND country = '{iso2.upper()}'
        """
        
        result = con.execute(validation_sql).fetchone()
        count = result[0] if result else 0
        
        if count > 0:
            logging.info(f"Validation successful: Found {count} division(s) for country {iso2.upper()}")
            return True
        else:
            logging.warning(f"Validation failed: No divisions found for country {iso2.upper()}")
            return False
            
    except Exception as e:
        logging.error(f"Division validation failed for {iso2.upper()}: {e}")
        return False
    finally:
        con.close()