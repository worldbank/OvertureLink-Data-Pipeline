"""
Configuration Parser for Overture Maps Pipeline
Supports dynamic templating and enterprise metadata standards
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .countries import CountryRegistry

logger = logging.getLogger(__name__)


@dataclass
class CountryConfig:
    """Country metadata configuration"""
    name: str
    iso2: str
    iso3: str
    region: str
    bbox: list[float]


@dataclass 
class OrganizationConfig:
    """Organization metadata configuration"""
    name: str
    contact_email: str
    attribution: str
    license: str
    update_frequency: str


@dataclass
class TargetMetadata:
    """Resolved target metadata with all templates applied"""
    # Core identification
    item_title: str
    snippet: str
    description: str
    service_name: str
    tags: list[str]
    
    # Essential metadata fields
    access_information: str
    license_info: str
    
    # Technical configuration
    upsert_key: str
    item_id: Optional[str] = None


class TemplateConfigParser:
    """
    Configuration parser with dynamic templating support.
    
    Supports:
    - Country/organization metadata at top level or dynamic injection
    - Dynamic template variables: {country_name}, {sector_title}, etc.
    - Enterprise metadata standards
    - Professional tag and categorization
    - Global configuration with runtime country selection
    """
    
    def __init__(self, config_path: str | Path, country_override: Optional[str] = None):
        """
        Initialize parser with config file path and optional country override
        
        Args:
            config_path: Path to configuration file
            country_override: ISO2/ISO3 code or country name to inject dynamically
        """
        self.config_path = Path(config_path)
        self.country_override = country_override
        self.raw_config = None
        self.country = None
        self.organization = None
        self.templates = None
        self.overture = None
        self.targets = None
        self.is_global_config = False
        
        self._load_config()
        self._parse_metadata_sections()
    
    def _load_config(self) -> None:
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, encoding='utf-8') as f:
            self.raw_config = yaml.safe_load(f)
        
        logger.debug(f"Loaded enhanced config from: {self.config_path}")
    
    def _parse_metadata_sections(self) -> None:
        """Parse top-level metadata sections with dynamic country injection support"""
        # Check if this is a global config (no country section but has country_override)
        self.is_global_config = 'country' not in self.raw_config and self.country_override
        
        # Country metadata - either from config file or dynamic injection
        if self.is_global_config and self.country_override:
            # Global config mode - inject country from registry
            country_info = CountryRegistry.get_country(self.country_override)
            if not country_info:
                available_countries = [c.name for c in CountryRegistry.list_countries()[:10]]
                raise ValueError(
                    f"Country '{self.country_override}' not found in registry. "
                    f"Available countries include: {', '.join(available_countries)}... "
                    f"Use ISO2, ISO3, or full country name."
                )
            
            # Convert CountryInfo to CountryConfig
            self.country = CountryConfig(
                name=country_info.name,
                iso2=country_info.iso2,
                iso3=country_info.iso3,
                region=country_info.region,
                bbox=country_info.bbox
            )
            
            logger.info(f"Global config mode: Injected country data for {country_info.name} ({country_info.iso2})")
            
        else:
            # Traditional config mode - use country section from file
            country_data = self.raw_config.get('country', {})
            self.country = CountryConfig(
                name=country_data.get('name', 'Unknown'),
                iso2=country_data.get('iso2', 'XX'),
                iso3=country_data.get('iso3', 'XXX'),
                region=country_data.get('region', 'Unknown Region'),
                bbox=country_data.get('bbox', [-180, -90, 180, 90])
            )
            
            # If country_override provided but country exists in config, warn about override
            if self.country_override and 'country' in self.raw_config:
                logger.warning(
                    f"Country override '{self.country_override}' provided but config already has country section. "
                    f"Using config file country: {self.country.name}"
                )
        
        # Organization metadata
        org_data = self.raw_config.get('organization', {})
        self.organization = OrganizationConfig(
            name=org_data.get('name', 'Organization'),
            contact_email=org_data.get('contact_email', 'contact@example.com'),
            attribution=org_data.get('attribution', 'Organization'),
            license=org_data.get('license', 'Internal use only'),
            update_frequency=org_data.get('update_frequency', 'Monthly')
        )
        
        # Templates and other sections
        self.templates = self.raw_config.get('templates', {})
        self.overture = self.raw_config.get('overture', {})
        self.targets = self.raw_config.get('targets', {})
    
    def get_template_variables(self, target_name: str, target_config: dict[str, Any]) -> dict[str, str]:
        """
        Generate template variables for a specific target
        
        Args:
            target_name: Name of the target (e.g., 'roads', 'education')
            target_config: Target configuration dictionary
            
        Returns:
            Dictionary of template variables for substitution
        """
        # Current date/time
        now = datetime.now()
        release = self.overture.get('release', 'latest')
        release_year = release.split('-')[0] if '-' in release else str(now.year)
        
        # Base variables available to all targets 
        variables = {
            # Country/region
            'country_name': self.country.name,
            'iso2': self.country.iso2,
            'iso3': self.country.iso3,
            'iso3_lower': self.country.iso3.lower(),
            'region': self.country.region,
            
            # Organization
            'organization': self.organization.attribution,
            'contact_email': self.organization.contact_email,
            'license': self.organization.license,
            'update_frequency': self.organization.update_frequency,
            
            # Overture data
            'release': release,
            'release_year': release_year,
            
            # Current date/time
            'current_date': now.strftime('%Y-%m-%d'),
            'current_datetime': now.isoformat(),
            
            # Target-specific (from target config)
            'sector_title': target_config.get('sector_title', target_name.title()),
            'sector_description': target_config.get('sector_description', f'{target_name} data'),
            'sector_tag': target_config.get('sector_tag', target_name),
            'data_type': target_config.get('data_type', 'Geospatial Data'),
        }
        
        # Now resolve attribution template using the variables we just created
        attribution_template = self.templates.get('attribution', '')
        if attribution_template:
            variables['attribution'] = self._resolve_template_string(attribution_template, variables)
        else:
            variables['attribution'] = ''
        
        return variables
    
    def _resolve_template_string(self, template: str, variables: dict[str, str]) -> str:
        """
        Resolve template string with variables
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            
        Returns:
            Resolved string with all variables substituted
        """
        if not isinstance(template, str):
            return template
        
        resolved = template
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            resolved = resolved.replace(placeholder, str(var_value))
        
        return resolved
    
    def _resolve_tags(self, tags_config: str | list[str], variables: dict[str, str]) -> list[str]:
        """
        Resolve tags configuration with template variables and concatenation
        
        Args:
            tags_config: Tags configuration (string with + operator or list)
            variables: Template variables
            
        Returns:
            List of resolved tag strings
        """
        if isinstance(tags_config, list):
            # Simple list - resolve each tag
            return [self._resolve_template_string(tag, variables) for tag in tags_config]
        
        if isinstance(tags_config, str) and '+' in tags_config:
            # Complex expression: "{tags_base} + ['additional', 'tags']"
            # First resolve template variables
            resolved_expr = self._resolve_template_string(tags_config, variables)
            
            # Parse base tags from templates
            tags_base = self.templates.get('tags_base', [])
            if isinstance(tags_base, list):
                base_resolved = [self._resolve_template_string(tag, variables) for tag in tags_base]
            else:
                base_resolved = []
            
            # Extract additional tags from the expression
            # Look for patterns like + ['tag1', 'tag2']
            additional_match = re.search(r'\+\s*\[(.*?)\]', resolved_expr)
            additional_tags = []
            
            if additional_match:
                # Parse the list content
                list_content = additional_match.group(1)
                # Simple parsing - split by comma and clean quotes
                additional_tags = [
                    tag.strip().strip("'\"") 
                    for tag in list_content.split(',') 
                    if tag.strip()
                ]
            
            return base_resolved + additional_tags
        
        # Simple string - treat as single tag or comma-separated
        resolved = self._resolve_template_string(tags_config, variables)
        if ',' in resolved:
            return [tag.strip() for tag in resolved.split(',')]
        return [resolved]
    
    def _resolve_template_reference(self, field_value: str, template_name: str, variables: dict[str, str]) -> str:
        """
        Resolve template reference - handles both direct templates and template variable references
        
        Args:
            field_value: The field value that might be a template reference like {access_information}
            template_name: The name of the template to look up (e.g., 'access_information')
            variables: Template variables for substitution
            
        Returns:
            Resolved string with all variables substituted
        """
        if field_value == f'{{{template_name}}}':
            # This is a template reference - use the template
            template = self.templates.get(template_name, '')
            return self._resolve_template_string(template, variables)
        else:
            # Direct template string
            return self._resolve_template_string(field_value, variables)
    
    def get_target_metadata(self, target_name: str) -> TargetMetadata:
        """
        Generate complete metadata for a target with all templates resolved
        
        Args:
            target_name: Name of the target configuration
            
        Returns:
            TargetMetadata object with all fields resolved
            
        Raises:
            KeyError: If target not found
            ValueError: If required fields missing
        """
        if target_name not in self.targets:
            available = list(self.targets.keys())
            raise KeyError(f"Target '{target_name}' not found. Available: {available}")
        
        target_config = self.targets[target_name]
        agol_config = target_config.get('agol', {})
        
        # Get template variables
        variables = self.get_template_variables(target_name, target_config)
        
        # Resolve template strings - handle both direct templates and template references
        # For item_title, check if it references {title} template
        raw_title = agol_config.get('item_title', '{title}')
        if raw_title == '{title}':
            # Use the title template
            title_template = self.templates.get('title', '{country_name} {sector_title}')
            resolved_title = self._resolve_template_string(title_template, variables)
        else:
            # Direct template string
            resolved_title = self._resolve_template_string(raw_title, variables)
        
        # For snippet, check if it references {snippet} template  
        raw_snippet = agol_config.get('snippet', '{snippet}')
        if raw_snippet == '{snippet}':
            # Use the snippet template
            snippet_template = self.templates.get('snippet', '{sector_description} for {country_name}')
            resolved_snippet = self._resolve_template_string(snippet_template, variables)
        else:
            # Direct template string
            resolved_snippet = self._resolve_template_string(raw_snippet, variables)
        
        # For description, check if it references {description} template
        raw_description = agol_config.get('description', '{description}')
        if raw_description == '{description}':
            # Use the description template
            description_template = self.templates.get('description', '')
            resolved_description = self._resolve_template_string(description_template, variables)
        else:
            # Direct template string
            resolved_description = self._resolve_template_string(raw_description, variables)
        
        # For service_name, check if it references {service_name} template
        raw_service_name = agol_config.get('service_name', '{service_name}')
        if raw_service_name == '{service_name}':
            # Use the service_name template
            service_name_template = self.templates.get('service_name', '{iso3}_{sector_tag}')
            resolved_service_name = self._resolve_template_string(service_name_template, variables)
        else:
            # Direct template string
            resolved_service_name = self._resolve_template_string(raw_service_name, variables)
        
        # Resolve all template fields
        metadata = TargetMetadata(
            # Core fields
            item_title=resolved_title,
            snippet=resolved_snippet,
            description=resolved_description,
            service_name=resolved_service_name,
            
            # Tags with complex resolution
            tags=self._resolve_tags(agol_config.get('tags', []), variables),
            
            # ESRI enterprise fields - handle template references like other fields
            access_information=self._resolve_template_reference(
                agol_config.get('access_information', '{access_information}'),
                'access_information',
                variables
            ),
            license_info=self._resolve_template_reference(
                agol_config.get('license_info', '{license_info}'),
                'license_info', 
                variables
            ),
            
            # Configuration
            upsert_key=agol_config.get('upsert_key', 'id'),
            item_id=agol_config.get('item_id')
        )
        
        logger.debug(f"Generated metadata for target '{target_name}': {metadata.item_title}")
        return metadata
    
    def get_selector_config(self) -> dict[str, Any]:
        """Get selector configuration with ISO2 resolved"""
        selector = self.raw_config.get('selector', {})
        
        # Resolve ISO2 template
        if 'iso2' in selector and isinstance(selector['iso2'], str):
            variables = {'iso2': self.country.iso2}
            selector['iso2'] = self._resolve_template_string(selector['iso2'], variables)
        
        return selector
    
    def get_overture_config(self) -> dict[str, Any]:
        """Get Overture Maps configuration"""
        return self.overture.copy()
    
    def get_target_filter_config(self, target_name: str) -> dict[str, Any]:
        """
        Get data selection configuration for a target
        
        Args:
            target_name: Name of the target
            
        Returns:
            Dictionary with theme, type, filter for data selection
        """
        if target_name not in self.targets:
            raise KeyError(f"Target '{target_name}' not found")
        
        target_config = self.targets[target_name]
        return {
            'theme': target_config.get('theme'),
            'type': target_config.get('type'),
            'filter': target_config.get('filter')
        }
    
    def validate_config(self) -> list[str]:
        """
        Validate configuration completeness
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # For global configs, country section is optional (injected dynamically)
        if self.is_global_config:
            required_sections = ['overture', 'targets']
            if not self.country_override:
                issues.append("Global config requires --country parameter")
        else:
            required_sections = ['country', 'overture', 'targets']
        
        # Check required sections
        for section in required_sections:
            if section not in self.raw_config:
                issues.append(f"Missing required section: {section}")
        
        # Check country metadata (should be resolved by now)
        if not self.country.name or self.country.name == 'Unknown':
            issues.append("Country name not specified or not found in registry")
        
        # Check targets
        for target_name, target_config in self.targets.items():
            agol_config = target_config.get('agol', {})
            
            if not agol_config.get('item_title') and not self.templates.get('title'):
                issues.append(f"Target {target_name}: No title template or item_title")
            
            if not target_config.get('theme'):
                issues.append(f"Target {target_name}: Missing theme")
            
            if not target_config.get('type'):
                issues.append(f"Target {target_name}: Missing type")
        
        return issues
    
    def is_using_global_config(self) -> bool:
        """Check if this parser is using global configuration mode"""
        return self.is_global_config
    
    def get_country_source(self) -> str:
        """Get description of where country data came from"""
        if self.is_global_config:
            return f"Country registry (--country {self.country_override})"
        else:
            return f"Configuration file ({self.config_path.name})"


def create_template_config_parser(config_path: str | Path, country_override: Optional[str] = None) -> TemplateConfigParser:
    """
    Factory function to create enhanced config parser with optional country injection
    
    Args:
        config_path: Path to YAML configuration file
        country_override: Optional country code/name for global config mode
        
    Returns:
        TemplateConfigParser instance
    """
    return TemplateConfigParser(config_path, country_override)