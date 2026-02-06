"""
Unified configuration loading interface for Overture Maps Pipeline.

This module provides a simple interface for loading and merging configuration
from multiple sources:
- configs/agol_metadata.yml (organizational metadata)
- data/queries.yml (query-specific metadata)
- Country registry (country information)

Returns unified configuration objects for use in CLI commands.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config.countries import CountryRegistry
from .config.settings import Config
from .domain.enums import ClipStrategy
from .domain.models import Country as DomainCountry
from .domain.models import Query as DomainQuery
from .domain.models import RunOptions


def load_config(
    query: str, 
    country: str, 
    config_path: str = None,
    load_agol_config: bool = True
) -> tuple[dict[str, Any], RunOptions, DomainQuery, DomainCountry]:
    """
    Load and merge configuration from all sources.
    
    Args:
        query: Query name (e.g., 'roads', 'education')
        country: Country identifier (name, ISO2, or ISO3)
        config_path: Path to AGOL metadata configuration file (defaults to data/agol_metadata.yml)
        
    Returns:
        Tuple of (cfg, run_options, query_obj, country_obj)
        - cfg: Complete configuration dictionary
        - run_options: Runtime options for pipeline
        - query_obj: Query domain object with metadata
        - country_obj: Country domain object
    """
    
    # Set default config path if not provided
    if config_path is None:
        config_path = Path(__file__).parent / "data" / "agol_metadata.yml"
    else:
        config_path = Path(config_path)
    
    # Load AGOL metadata configuration
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    
    with open(config_path, encoding='utf-8') as f:
        agol_config = yaml.safe_load(f)
    
    # Load query definitions
    queries_file = Path(__file__).parent / "data" / "queries.yml"
    with open(queries_file, encoding='utf-8') as f:
        queries = yaml.safe_load(f)
    
    # Validate query exists
    if query not in queries:
        available = list(queries.keys())
        raise ValueError(f"Query '{query}' not found. Available: {available}")
    
    query_config = queries[query]
    
    # Resolve country information
    country_info = CountryRegistry.get_country(country)
    if not country_info:
        raise ValueError(f"Unknown country: {country}")
    
    # Create domain objects with sensible defaults
    run_options = RunOptions(
        clip=ClipStrategy.DIVISIONS,  # Default to divisions
        limit=None,
        use_bbox=False
    )
    
    geometry_split = bool(query_config.get('geometry_split'))
    
    domain_query = DomainQuery(
        theme=query_config['theme'],
        type=query_config['type'],
        filter=query_config.get('filter'),
        name=query,
        is_multilayer=bool(query_config.get('building_filter')) or geometry_split,
        geometry_split=geometry_split,
        # Filter configuration
        building_filter=query_config.get('building_filter'),
        category_filter=query_config.get('filter'),  # Use filter as category_filter
        # Pipeline attributes
        field_mappings={},
        original_config=query_config,
        # Metadata fields from queries.yml
        sector_title=query_config.get('sector_title'),
        sector_description=query_config.get('sector_description'),
        sector_tag=query_config.get('sector_tag'),
        data_type=query_config.get('data_type')
    )
    
    
    domain_country = DomainCountry(
        name=country_info.name,
        iso2=country_info.iso2,
        iso3=country_info.iso3,
        bounds=CountryRegistry.get_bounding_boxes().get(country_info.iso2, (0, 0, 0, 0))
    )
    
    
    # Merge configurations - add overture config from environment variable
    # This ensures all components use the same release from OVERTURE_RELEASE env var
    overture_config = Config(validate_on_init=load_agol_config).overture
    
    cfg = {
        **agol_config,
        'query': query_config,
        'country': {
            'name': country_info.name,
            'iso2': country_info.iso2,
            'iso3': country_info.iso3,
            'region': country_info.region
        },
        'overture': {
            'base_url': overture_config.base_url,
            's3_region': overture_config.s3_region,
            'release': overture_config.release
        }
    }
    
    return cfg, run_options, domain_query, domain_country


def format_metadata_from_config(
    config_dict: dict[str, Any], 
    query_config: DomainQuery, 
    country_config: DomainCountry
) -> dict[str, Any]:
    """
    Generate AGOL metadata from config templates and dynamic variables.
    
    Args:
        config_dict: Loaded configuration dictionary from agol_metadata.yml
        query_config: Query domain object with sector information  
        country_config: Country domain object
        
    Returns:
        Dictionary with formatted metadata for AGOL publishing
    """
    # Get templates from config
    templates = config_dict.get('templates', {})
    organization = config_dict.get('organization', {})
    
    # Create variable substitution dictionary
    variables = {
        'country_name': country_config.name,
        'iso2': country_config.iso2,
        'iso3': country_config.iso3,
        'iso3_lower': country_config.iso3.lower(),
        'sector_title': query_config.sector_title or query_config.type.title(),
        'sector_description': query_config.sector_description or f"{query_config.type} data",
        'sector_tag': query_config.sector_tag or query_config.name,
        'data_type': query_config.data_type or query_config.type.title(),
        'release': Config().overture.release,
        'update_frequency': organization.get('update_frequency', 'Monthly'),
        'contact_email': organization.get('contact_email', 'contact@example.com'),
        'license': organization.get('license', 'Internal use only'),
        'attribution': organization.get('attribution', 'Your Organization'),
        'organization': organization.get('name', 'Your Organization'),
        'current_date': datetime.now().strftime('%Y-%m-%d')
    }

    # Overture + schema notes for all queries
    variables["overture_notes"] = (
        "<b>About Overture Maps:</b> Overture Maps Foundation publishes open, global geospatial data "
        "organized into themes with consistent schemas for cross‑country analysis.<br>"
        "<b>Schema:</b> This dataset uses the Overture <b>{theme}</b> theme and <b>{type}</b> layer schema.<br>"
    ).format(theme=query_config.theme, type=query_config.type)

    # Layer notes (sectoral = three layers; others = single layer)
    if getattr(query_config, "is_multilayer", False) and query_config.name in ("education", "health", "markets"):
        variables["layer_notes"] = (
            "<b>Layers:</b><br>"
            "<ul>"
            "<li><b>places</b>: Points from Overture Places.</li>"
            "<li><b>buildings</b>: Polygons from Overture Buildings.</li>"
            "<li><b>places_combined</b>: Points combining Places with building centroids for analysis.</li>"
            "</ul><br>"
        )
    else:
        variables["layer_notes"] = (
            "<b>Layers:</b><br>"
            "<ul>"
            "<li><b>{layer_name}</b>: Features from Overture {theme}/{type}.</li>"
            "</ul><br>"
        ).format(layer_name=query_config.name, theme=query_config.theme, type=query_config.type)
    
    # Format templates with variables
    formatted_metadata = {}
    
    # Title
    title_template = templates.get('title', '{country_name} {sector_title}')
    formatted_metadata['title'] = title_template.format(**variables)
    
    # Snippet
    snippet_template = templates.get('snippet', '{country_name} {sector_description}')
    formatted_metadata['snippet'] = snippet_template.format(**variables)
    
    # Description (handle multiline)
    description_template = templates.get('description', '{sector_title} data for {country_name}')
    formatted_metadata['description'] = description_template.format(**variables)
    
    # Tags (process from query configuration)
    # The query config has tags like: "{tags_base} + ['transportation', 'roads']"
    # We need to evaluate this template with tags_base from agol_metadata.yml
    query_tags_template = query_config.original_config.get('agol', {}).get('tags', '{tags_base}')
    
    # First resolve tags_base from templates
    base_tags = templates.get('tags_base', [])
    formatted_base_tags = []
    for tag in base_tags:
        if isinstance(tag, str) and '{' in tag:
            formatted_base_tags.append(tag.format(**variables))
        else:
            formatted_base_tags.append(tag)
    
    # Add tags_base to variables for template processing
    variables['tags_base'] = formatted_base_tags
    
    # Process the query's tags template (e.g., "{tags_base} + ['transportation', 'roads']")
    try:
        # Safely evaluate the tags expression
        formatted_metadata['tags'] = eval(query_tags_template.format(**variables))
    except:
        # Fallback to just base tags if evaluation fails
        formatted_metadata['tags'] = formatted_base_tags
    
    # License info
    license_template = templates.get('license_info', '{license}')
    formatted_metadata['license_info'] = license_template.format(**variables)
    
    # Access information  
    access_template = templates.get('access_information', '{attribution}')
    formatted_metadata['access_information'] = access_template.format(**variables)
    
    # Credits/Attribution
    attribution_template = templates.get('attribution', 'Overture Maps Foundation')
    formatted_metadata['credits'] = attribution_template.format(**variables)
    
    return formatted_metadata


def load_pipeline_config(config_path: str, country: str) -> dict[str, Any]:
    """
    Load pipeline configuration (legacy interface for backward compatibility).
    
    Args:
        config_path: Path to configuration file
        country: Country identifier
        
    Returns:
        Configuration dictionary
    """
    cfg, _, _, _ = load_config("roads", country, config_path)  # Use roads as default
    return cfg


def get_target_config(cfg: dict[str, Any], query: str) -> dict[str, Any]:
    """
    Get target configuration for a specific query.
    
    Args:
        cfg: Configuration dictionary
        query: Query name
        
    Returns:
        Target configuration dictionary
    """
    return cfg.get('query', {})


def validate_query_exists(config_path: str = None, query: str = None) -> bool:
    """
    Validate that a query exists in the configuration.
    
    Args:
        config_path: Path to configuration file (not used, kept for compatibility)
        query: Query name to validate
        
    Returns:
        True if query exists, False otherwise
    """
    try:
        queries_file = Path(__file__).parent / "data" / "queries.yml"
        with open(queries_file, encoding='utf-8') as f:
            queries = yaml.safe_load(f)
        return query in queries
    except Exception:
        return False


def get_available_queries(config_path: str = None) -> list[str]:
    """
    Get list of available queries.
    
    Args:
        config_path: Path to configuration file (not used, kept for compatibility)
        
    Returns:
        List of available query names
    """
    queries_file = Path(__file__).parent / "data" / "queries.yml"
    with open(queries_file, encoding='utf-8') as f:
        queries = yaml.safe_load(f)
    return list(queries.keys())
