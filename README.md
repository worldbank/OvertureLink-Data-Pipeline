# Overture Maps Data to ArcGIS Online Pipeline

## Purpose
This is a cloud-native ETL pipine to extract Overture Maps data (such as roads, buildings), transform to an AGOL-ready schema, and publish as a feature layer in ArcGIS Online. This is designed to align with Overture's monthly releases and for country-specific data pulls.

## Pipeline Overview
- Ingest (`duck.py`): DuckDB reads Overture GeoParquet remotely from S3. We did this so the user doesn't have to download anything.
- Transform (`transform.py`): Normalize schema/geometry, retain stable Overture/GERS IDs, define/delete/add metadata fields.
- Publish (`publish.py`): ArcGIS Python API with smart detection - automatically creates new layers or updates existing ones with data and metadata.
- Automation: GitHub Actions (manual + schedule). Secrets via GitHub Environments for elements such as AGOL authentification.

## Requirements
- Python 3.11+ (version must be compatiable with arcgis package)
- AGOL credentials with World Bank. You can use this for your own purposes if you want with your own AGOl ccount. Add your credientials however you want. There is an example .env file for you to use securely.

## Setup for Local Use

### 1. Create Virtual Environment
Create virtual environment
`python -m venv .venv`

Activate virtual environment
On Windows:
`.\.venv\Scripts\Activate.ps1`

On macOS/Linux:
`source .venv/bin/activate`

### 2. Install Dependencies
`pip install -e .`

### 3. Define Environment
- Copy `configs/afg.yml` for your own country
- Use the .env example to define your variables  
- Build your command, for example:

### 4. Run commands
- Using CLI entry point
   `o2agol roads --config configs/afg.yml --mode initial`
   
- Or using module directly
   `python -m o2agol.cli roads --config configs/afg.yml --mode initial`

Note: The pipeline automatically detects existing layers and updates them seamlessly using smart detection mode.

## Config
`configs/<country>.yml` includes:
- `overture_release`: "latest" or a specific date.
- `selector`: Add ISO2 country code 
- `targets`: roads/buildings with layer titles, item IDs (for updates), filters.
- `mode`: initial/overwrite/append.

### Clip modes
- Overture Division Clip: geometric clip with `ST_Intersects`. Use `--use-divisions` for production (or neither flag, as it is default).
- Box: bbox-only overlap. For development due to its speed. Use `--use-bbox` 

## Outputs in AGOL
- Hosted Feature Layers (roads lines, buildings polygons).
- Metadata based on config

## Planned Improvements
- Automatically add item ID to config for update.
- Using bbox to query before division clip for optimization.
- Option for full dump for multiple uploads.
- Schema mismatch on append: ensure `id` key matches.
- Large export: chunk export by grid; raise `maxRecordCount`.

## List of options

### Choosing query
There are sets of queries prebuilt that you can find in the configs folder but you can create your own. 

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
- `-c configs/afg.yml` - points to your config file

### Modes
- `--mode auto` (default) - Smart detection: automatically creates new layers or updates existing ones based on service name
- `--mode initial` - Force creation of new layer (use when auto-detection fails)
- `--mode overwrite` - Force update of existing layer (requires item_id in config)
- `--mode append` - Add data to existing layer without clearing (requires item_id in config)

### Clipping
- `--use-divisions` (default) - Uses Overture Divisions for precise boundaries
- `--use-bbox` - Takes the bbox without clipping on the admin boundaries. This is much faster than clipping along the borders of the country polygon. 

### Other parameters
- `--limit 1000` - Limits the features for testing purposes
- `--verbose` / `-v` - Enable detailed debug logging output  
- `--log-to-file` - Create timestamped log files in "/logs" directory
- `--dry-run` - Processes but doesn't publish to AGOL, good for testing without uploading every test
- `--iso2 XX` - Override country code from config (e.g. `--iso2 af` for Afghanistan)

## Examples:

### Recommended Usage (Smart Detection)
- `o2agol education -c configs/afg.yml --log-to-file` - Automatically creates or updates Afghanistan education facilities (multi-layer service with points + polygons)
- `o2agol health -c configs/afg.yml --use-bbox --limit 1000` - Process health facilities using fast bbox filtering with 1000 feature limit for testing
- `o2agol roads -c configs/afg.yml --verbose` - Process Afghanistan roads with detailed logging

### Core Infrastructure  
- `o2agol roads -c configs/afg.yml` - Process transportation networks (auto-detects new vs update)
- `o2agol buildings -c configs/afg.yml --dry-run` - Validate buildings configuration without publishing

### Advanced Usage
- `o2agol markets -c configs/afg.yml --mode initial` - Force creation of new markets layer (override auto-detection)
- `o2agol education -c configs/afg.yml --mode overwrite --iso2 pk` - Update education layer using Pakistan country code override

## CI/CD
- `ci.yml`: lint, type-check, unit tests on PR/main.
- `run-pipeline.yml`: manual (`workflow_dispatch`) or schedule (cron). Uses environment secrets.

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutoral here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- This post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/