# Overture Maps Data to ArcGIS Online Pipeline

## Purpose
This is a cloud-native ETL pipeline to extract Overture Maps data (such as roads, buildings), transform to an AGOL-ready schema, and publish as a feature layer in ArcGIS Online. This pipeline supports 176 countries worldwide and is designed to align with Overture's monthly releases.

## Pipeline Overview
- Ingest (`duck.py`): DuckDB reads Overture GeoParquet remotely from S3 with optimized spatial queries and Arrow processing for fast data access.
- Transform (`transform.py`): Normalize schema/geometry, retain stable Overture/GERS IDs, define/delete/add metadata fields.
- Publish (`publish.py`): ArcGIS Python API with smart detection - automatically creates new layers or updates existing ones with data and metadata.
- Automation: GitHub Actions (manual + schedule). Secrets via GitHub Environments for elements such as AGOL authentification.

## Requirements
- Python 3.11+ (version must be compatiable with arcgis package)
- AGOL credentials with World Bank. You can use this for your own purposes if you want with your own AGOl ccount. Add your credientials however you want. There is an example .env file for you to use securely.

## Setup for Local Use

### 1. Create Virtual Environment
Create virtual environment (optional, but recommended)

Activate virtual environment
On Windows:
`.venv\Scripts\Activate.ps1`

On macOS/Linux:
`source .venv/bin/activate`

### 2. Install Dependencies
`pip install -e .`

### 3. Define Environment
- Use the `.env` example to define your AGOL credentials
- No need to create country-specific config files, you can use the global config with a country argument

### 4. Run commands
- Global config with country parameter:
   `o2agol roads -c configs/global.yml --country afg`
   
- Or using module directly:
   `python -m o2agol.cli roads -c configs/global.yml --country pak`

Note: The pipeline automatically detects existing layers and updates them.

## Config

### Global Configuration (Recommended)
Use `configs/global.yml` with the `--country` parameter:
- **Country Selection**: Specify any country using name, ISO2, or ISO3 code (supports 176 countries)
- **Dynamic Processing**: Country metadata and bounding boxes are automatically loaded
- **Template System**: All titles, descriptions, and metadata are dynamically generated

### Legacy Country-Specific Configs
`configs/<country>.yml` files are still supported:
- `overture_release`: "latest" or a specific date
- `selector`: ISO2 country code 
- `targets`: roads/buildings with layer titles, item IDs, filters
- `mode`: initial/overwrite/append

### Clip modes
- Overture Division Clip: precise geometric clip with `ST_Intersects` and bbox pre-filtering for optimal performance. Use `--use-divisions` for production (default).
- Box: bbox-only overlap for fastest processing. Use `--use-bbox` for development and testing.

## Outputs in AGOL
- Hosted Feature Layers (roads lines, buildings polygons).
- Metadata based on config

## List of options

### Choosing query
There are sets of queries prebuilt that you can find in the configs file but you can create your own. 

#### Lines
- `roads` - Transportation networks (all roads, not filtered). `theme: transportation`

#### Points & Polygons (Multi-layer)
- `education` - Education facilities. Creates multi-layer service with both geometry types.
- `health` - Health facilities. Creates multi-layer service with both geometry types.
- `markets` - Retail facilities. Creates multi-layer service with both geometry types.

#### Points Only
- `places` - All points of interest (unfiltered places)

#### Polygons Only
- `buildings` - Building footprints (all building polygons) 

### Choosing config file
- `-c configs/global.yml --country <code>` - Global config with dynamic country selection (recommended)
- `-c configs/afg.yml` - Legacy country-specific config file

### Modes
- `--mode auto` (default) - Smart detection: automatically creates new layers or updates existing ones based on service name
- `--mode initial` - Force creation of new layer (use when auto-detection fails)
- `--mode overwrite` - Force update of existing layer (requires item_id in config)
- `--mode append` - Add data to existing layer without clearing (requires item_id in config)

### Clipping
- `--use-divisions` (default) - Uses Overture Divisions for precise boundaries with bbox pre-filtering for optimal performance
- `--use-bbox` - Fast bbox-only filtering for development and testing

### Other parameters
- `--country <code>` - Specify country by name, ISO2, or ISO3 code (e.g., `--country afghanistan`, `--country af`, `--country afg`)
- `--limit 1000` - Limits the features for testing purposes
- `--verbose` / `-v` - Enable detailed debug logging output  
- `--log-to-file` - Create timestamped log files in "/logs" directory
- `--dry-run` - Processes but doesn't publish to AGOL, good for testing without uploading every test

## Examples:

### Global Config (Recommended):
- `o2agol roads -c configs/global.yml --country afg` - Afghanistan road networks
- `o2agol education -c configs/global.yml --country pakistan --log-to-file` - Pakistan education facilities with logging
- `o2agol health -c configs/global.yml --country usa --use-bbox --limit 1000` - USA health facilities (bbox mode, 1000 features)
- `o2agol buildings -c configs/global.yml --country "south africa" --dry-run` - South Africa buildings (test mode)

### Legacy Config:
- `o2agol roads -c configs/afg.yml` - Process road networks using country-specific config
- `o2agol education -c configs/afg.yml --log-to-file` - Afghanistan education facilities using legacy config

## CI/CD
- `ci.yml`: lint, type-check, unit tests on PR/main.
- `run-pipeline.yml`: manual (`workflow_dispatch`) or schedule (cron). Uses environment secrets.

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutoral here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- Georock's post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/
- Bounding Box extracts from Natural Earth: https://github.com/sandstrom/country-bounding-boxes/tree/master