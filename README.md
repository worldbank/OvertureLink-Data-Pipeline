# Overture Maps Data to GeoJSON and ArcGIS Online Pipeline

## Purpose
Choose your Overture query, specify the country, then you're done!

This cloud-native ETL pipeline allows you to query and extract Overture Maps data (such as roads, buildings) to upload to ArcGIS Online, download as .geojson for any GIS software, or save as a local dump for continual use. This pipeline supports 176 countries worldwide with its country/ISO database, allows you to use pre-built queries or your own custom queries, and is designed to align with Overture's monthly releases.

### Development status
Please note development is ongoing. The pipeline works with all the options described below. The only issue is when running large polygon datasets >8 million. Sometimes there are hang ups with ArcGIS Online during appending. We're investigating the best options to reduce these errors, such as setting the right append parameters and testing upload as a FGDB.

### Three commands
- `agol-upload` - Upload your query to ArcGIS Online
- `geojson-download` - Download your query in .geojson format
- `overture-dump` - Caches your query (by country / theme) for continued use without need for multiple downloads.

### Features
- Automatic AGOL discovery: The pipeline will either create a new feature layer or use truncate and append based on whether or not a feature layer already exists. 
- Online and cache query: Query from Overture directly, or download a dump for consecutive uses.
- Robust options: Custom configs, feature limits, debug logging, and more.

## Pipeline Overview
- Configuration (`configs\global.yml`): Easily change metadata, choose the Overture release, or add your own queries.
- Ingest (`duck.py`): DuckDB reads Overture GeoParquet remotely from S3 with spatial queries and for data access.
- Transform (`transform.py`): Normalize schema/geometry, retain stable Overture/GERS IDs, define/delete/add metadata fields. Turns into an easy to use .geojson for exporting if you wish.
- Publish (`publish.py`): ArcGIS Python API with detection, automatically creates new layers or updates existing ones with data and metadata.
- Data dump (`dump-manager.py`): Only for local country caching if you want to use it. Checks to see if there is data by country and theme for Overture and downloads if needed.

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
The Python CLI has three main commands: uploading to AGOL, downloading as geojson, or download dump for local use as needed.

#### Upload to ArcGIS Online:
- Example with Afghanistan country parameter:
   `o2agol arcgis-upload roads --country afg`
   
- Or using module directly:
   `python -m o2agol.cli arcgis-upload roads --country afg`

#### Export to GeoJSON:
- Export Afghanistan roads (auto-filename: afg_roads.geojson):
   `o2agol geojson-download roads --country afg`
   
- Or specify output file:
   `o2agol geojson-download roads afghanistan_roads.geojson --country afg`

#### Download local dump for consistent use
- Example with Afghanistan country parameter, detects and downloads local dump.
   `o2agol overture-dump roads --country afg`

### Review options
- If needed you can always review options with the --help argument.
- `o2agol --help`
- `o2agol arcgis-upload --help`
- `o2agol geojson-download --help`
- `o2agol overture-dump --help`

### List Queries
- You can list the available queries with this command:
   `o2agol list-queries`
   or
   `python -m o2agol.cli list-queries`

And add you own query in the global config (`configs\global.yml`)

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
- Overture Division Clip: precise geometric clip with bbox pre-filtering. Use `--use-divisions` for production (default).
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

### Choosing a Query
There are sets of queries prebuilt that you can find below and in the configs file, but you can create your own. 

#### Lines
- `roads` - Transportation networks (all roads, not filtered).

#### Points & Polygons (Multi-layer)
- `education` - Education facilities. Creates multi-layer service with both geometry types.
- `health` - Health facilities. Creates multi-layer service with both geometry types.
- `markets` - Retail facilities. Creates multi-layer service with both geometry types.

#### Points Only
- `places` - All points of interest (unfiltered places)

#### Polygons Only
- `buildings` - Building footprints (all building polygons) 

### Choosing a Country

- `--country <code>` - Specify country by name, ISO2, or ISO3 code (e.g., `--country afghanistan`, `--country af`, `--country afg`)

## List of optional arguments
Below is a list of optional arguments. Useful if you need to tailor your command to something, or use a legacy command. Default options do not need to be added to your command.

### Useful parameters
- `--limit 1000` - Limits the features for testing purposes
- `--verbose` / `-v` - Enable detailed debug logging output  
- `--log-to-file` - Create timestamped log files in "/logs" directory
- `--dry-run` - Processes but doesn't publish to AGOL, good for testing without uploading every test
- `-c configs/global.yml` - Choose another config (global.yml is default) 

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

### Troubleshooting

#### Common Issues
- Validation errors: Re-download with `--force-download`
- Memory errors: Reduce `DUMP_MAX_MEMORY` or use `--limit`
- Disk space: Use `o2agol list-dumps` to check space usage
- Larger data uploads (8+ million features) can sometimes hang on the upload to ArcGIS. We are actively trying to figure out the best way to deal with this.

#### Performance Tips  
- Use `--use-bbox` for development (faster spatial filtering)
- Increase `DUMP_MAX_MEMORY` if you have available RAM
- Download dumps to SSD for better query performance
- Use `--limit` for development to reduce processing time

## CI/CD
- `ci.yml`: lint, type-check, unit tests on PR/main.
- `run-pipeline.yml`: manual (`workflow_dispatch`) or schedule (cron). Uses environment secrets.

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutoral here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- Georock's post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/
- Bounding Box extracts from Natural Earth: https://github.com/sandstrom/country-bounding-boxes/tree/master