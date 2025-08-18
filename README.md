# Overture Maps Data to GeoJSON and ArcGIS Online Pipeline

## Purpose
Choose your Overture query, specific the country, then you're done!

This is a cloud-native ETL pipeline to extract Overture Maps data (such as roads, buildings), transform to an AGOL-ready schema, and then allows you to choose to export to .geojson or publish as a feature layer in ArcGIS Online. This pipeline supports 176 countries worldwide, allows you to use pre-built queries or your own custom queries, and is designed to align with Overture's monthly releases.

## Pipeline Overview
- Configuration (`configs\global.yml`): Easily change metadata, choose the Overture release, or add your own queries.
- Ingest (`duck.py`): DuckDB reads Overture GeoParquet remotely from S3 with optimized spatial queries and Arrow processing for fast data access.
- Transform (`transform.py`): Normalize schema/geometry, retain stable Overture/GERS IDs, define/delete/add metadata fields. Turns into an easy to use .geojson for exporting if you wish.
- Publish (`publish.py`): ArcGIS Python API with smart detection - automatically creates new layers or updates existing ones with data and metadata.

## Requirements
- Python 3.11+ (version must be compatiable with arcgis package)
- AGOL credentials for upload. Originally created for World Bank AGOL, you can use this for your own purposes if you want with your own AGOl ccount or simply use it to export to .geojson without an AGOL account. There is an example .env file for you to use securely.

## Setup for Local Use

### 1. Create Virtual Environment
Create virtual environment (optional, but recommended). Make sure it is a Python version compatible with ArcGIS. I personally run: 
`py -3.12 -m venv .venv`
You can also copy your virtual environment from ArcGIS Pro.

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

#### Upload to ArcGIS Online:
- Example with Afghanistan country parameter:
   `o2agol arcgis-upload roads --country afg`
   
- Or using module directly:
   `python -m o2agol.cli arcgis-upload roads --country pak`

#### Export to GeoJSON:
- Export Afghanistan roads (auto-filename: afg_roads.geojson):
   `o2agol geojson-download roads --country afg`
   
- Or specify output file:
   `o2agol geojson-download roads afghanistan_roads.geojson --country afg`

### List Queries
- You can list the available queries:
   `o2agol list-queries`
   or
   `python -m o2agol.cli list-queries`

And add you own query in the global config (`configs\global.yml`)

Note: The pipeline automatically detects existing AGOL layers and updates them using truncate and append.

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
- Overture Division Clip: precise geometric clip with bbox pre-filtering for optimal performance. Use `--use-divisions` for production (default).
- Box: bbox-only overlap for fastest processing. Use `--use-bbox` for development and testing.

## Outputs

### ArcGIS Online Upload
- Creates or updates Hosted Feature Layers in your ArcGIS Online organization
- Metadata automatically applied based on configuration
- Supports lines (roads), points (places), polygons (buildings), and multi-layer services

### GeoJSON Export
- Standards-compliant GeoJSON FeatureCollection files
- Includes metadata with generation timestamp and feature counts  
- Auto-generated filenames follow pattern: {iso3}_{query}.geojson
- Full Unicode support for international place names

## List of required arguments
To build your command, you need three elemenets:
- Whether you are uploading to AGOL or downloading geojson
- The query you are using
- The country you are querying

### Commands Available

#### ArcGIS Online Upload
- `arcgis-upload` - Process and upload data to ArcGIS Online
- Supports `--dry-run` for validation without publishing
- Supports all processing modes: auto, initial, overwrite, append

#### GeoJSON Export  
- `geojson-download` - Process and export data as GeoJSON file
- Auto-generates filename if not specified: {iso3}_{query}.geojson

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

### Choosing country

- `--country <code>` - Specify country by name, ISO2, or ISO3 code (e.g., `--country afghanistan`, `--country af`, `--country afg`)

## List of optional arguments
Below is a list of optional arguments. Useful if you need to tailor your command to something, or use a legacy command. Default options do not need to be added to your command.

### Useful parameters
- `--limit 1000` - Limits the features for testing purposes
- `--verbose` / `-v` - Enable detailed debug logging output  
- `--log-to-file` - Create timestamped log files in "/logs" directory
- `--dry-run` - Processes but doesn't publish to AGOL, good for testing without uploading every test

### Choosing config file
- `-c configs/global.yml --country <code>` (Default) - Global config with dynamic country selection 
- `-c configs/afg.yml` - Legacy country-specific config file

### Modes
- `--mode auto` (default) - Smart detection: automatically creates new layers or updates existing ones based on service name
- `--mode initial` - Force creation of new layer (use when auto-detection fails)
- `--mode overwrite` - Force update of existing layer (requires item_id in config)
- `--mode append` - Add data to existing layer without clearing (requires item_id in config)

### Clipping
- `--use-divisions` (default) - Uses Overture Divisions for precise boundaries with bbox pre-filtering for optimal performance
- `--use-bbox` - Fast bbox-only filtering for development and testing

## Examples:

### ArcGIS Online Upload:
- `o2agol arcgis-upload roads --country afg` - Afghanistan road networks
- `o2agol arcgis-upload education --country pakistan --log-to-file` - Pakistan education facilities with logging
- `o2agol arcgis-upload buildings --country "south africa" --dry-run` - South Africa buildings (test mode)

### GeoJSON Export:
- `o2agol geojson-download roads --country afg` - Export Afghanistan roads (auto-filename: afg_roads.geojson)
- `o2agol geojson-download health usa_health.geojson --country usa --limit 1000` - USA health facilities to specific file
- `o2agol geojson-download education --country pak --use-bbox --limit 100` - Fast export with bounding box

### Legacy Config:
- `o2agol arcgis-upload roads -c configs/afg.yml` - Process road networks using country-specific config
- `o2agol geojson-download education -c configs/afg.yml` - Export Afghanistan education facilities using legacy config

## CI/CD
- `ci.yml`: lint, type-check, unit tests on PR/main.
- `run-pipeline.yml`: manual (`workflow_dispatch`) or schedule (cron). Uses environment secrets.

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutoral here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- Georock's post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/
- Bounding Box extracts from Natural Earth: https://github.com/sandstrom/country-bounding-boxes/tree/master