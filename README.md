# OvertureLink
## Flexible Overture data ingestion for AGOL and export workflows

## Purpose
Choose your Overture query, specify the country, then you're done!

This ETL pipeline allows you to query and extract Overture Maps data (such as roads, buildings) to upload to ArcGIS Online, download as different file types for any GIS software, or save as a local dump for continual use. This pipeline supports 176 countries worldwide with its country/ISO database, allows you to use pre-built queries or your own custom queries, and is designed to align with Overture's monthly releases. This pipeline was originally built to support the World Bank's distributed data across ArcGIS Hubs, but should work for other work flows. 

## Commands
There are three main commands when using this pipeline. 
- `overture-dump` - This command caches Overture dumps with a local caching system, which is useful for multiple country uploads.
- `arcgis-upload` - Publish processed data to ArcGIS Online feature layers as a one off. 
- `export` - Export data to GeoJSON, GeoPackage, or File Geodatabase formats without uploading to ArcGIS Online. This is useful if you just need to grab specific queried data from Overture.


## Key Features
- Automatic AGOL discovery: The pipeline will either create a new feature layer or use truncate and append based on whether or not a feature layer already exists. 
- Online and cache query: Query from Overture directly, or download a dump for consecutive uses.
- Large files can be run in batch mode to assist with uploading to AGOL.
- Robust options: Custom configs, feature limits, debug logging, and more.

## Architecture
The pipeline follows a modular Source -> Transform -> Publish/Export process:

- Data Ingestion (`pipeline/source.py`): Uses DuckDB S3 queries and local dump management. Geopackage is the default, but format can be changed.
- Schema Transformation (`pipeline/transform.py`): Normalizes schema and geometry while retaining stable Overture IDs
- ArcGIS Publishing (`pipeline/publish.py`): Handles feature layer creation, updates, and multi-layer services
- Multi-Format Export (`pipeline/export.py`): Exports to GeoJSON, GeoPackage, and File Geodatabase formats
- Configuration Management (`data folder`): Central folder for configuration of queries, metadata, and country data. 

## Requirements
- Python 3.11+ (compatible with ArcGIS Python API)
- ArcGIS Online credentials or OAuth client ID (required for `arcgis-upload` and `overture-dump` command)

The pipeline can be used for data export without ArcGIS Online credentials. Environment configuration is managed through `.env` files for secure credential storage.

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

**2FA / Browser Login (Recommended)**
- Set `AGOL_USE_OAUTH=true` and `AGOL_CLIENT_ID=...` (no username/password required).
- On first run, a browser window opens to complete sign-in and 2FA.
- You can register an ArcGIS Online app to get a Client ID (Application type: "Other").
- Optional: set `AGOL_PROFILE=your_profile` to cache the session and avoid repeated prompts in batch runs.

**Username/Password (Non‑2FA or Service Accounts)**
- Set `AGOL_USERNAME` and `AGOL_PASSWORD` as shown in `.env.example`.

### 4. Run commands
The Python CLI has three main commands: uploading to AGOL `arcgis-upload` , downloading as geojson `export`, or download dump for local use as needed `overture-dump`. 

## Outputs

### ArcGIS Online Upload
- Creates or updates Hosted Feature Layers in your ArcGIS Online organization
- Metadata automatically applied based on configuration
- Supports lines (roads), points (places), polygons (buildings), and multi-layer services

### Data Export (Multiple Formats)
- **GeoPackage (GPKG)**: SQLite-based format with multi-layer support (default)
- **GeoJSON**: Standards-compliant JSON format 
- **File Geodatabase (FGDB)**: ESRI format for ArcGIS workflows

## List of required arguments
To build your command, you need three elements:
- Whether you are uploading to AGOL or exporting data
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
- `places` - All points of interest 

#### Polygons Only
- `buildings` - Building footprints

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

## Examples:

### Help Command
- If needed you can always review options with the --help argument.
- `o2agol --help`
- `o2agol arcgis-upload --help`
- `o2agol export --help`
- `o2agol overture-dump --help`

### List Queries
- You can list the available queries with this command:
   `o2agol list-queries`
   or
   `python -m o2agol.cli list-queries`

And add you own query in the global config (`src\o2agol\data\queries.yml`)

#### Upload to ArcGIS Online:
- Example with Afghanistan country parameter:
   `o2agol arcgis-upload roads --country afg`
   
- Or using module directly:
   `python -m o2agol.cli arcgis-upload roads --country afg`

#### Export to Multiple Formats:
- Export Afghanistan roads to GeoJSON (auto-filename: afg_roads.geojson):
   `o2agol export roads --country lux`
   
- Export to GeoPackage format:
   `o2agol export buildings --country lux --format gpkg`
   
- Export to custom filename with format auto-detection:
   `o2agol export roads lux_roads.gpkg --country lux`

### ArcGIS Online Upload:
- `o2agol arcgis-upload roads --country afg` - Afghanistan road networks
- `o2agol arcgis-upload education --country pakistan --log-to-file` - Pakistan education facilities with logging
- `o2agol arcgis-upload buildings --country "south africa" --dry-run` - South Africa buildings (test mode)

### Data Export:
- `o2agol export roads --country afg` - Export Afghanistan roads to GeoJSON (auto-filename: afg_roads.geojson)
- `o2agol export health usa_health.gpkg --country usa --limit 1000` - USA health facilities to GeoPackage
- `o2agol export education --country pak --format gpkg --use-bbox --limit 100` - Fast export with bounding box to GPKG
- `o2agol export buildings --country ind --format fgdb --raw` - Raw building data to File Geodatabase

### Cache Management
Efficiently manage local cache and dump storage:

```bash
# List all cached data with metadata
o2agol list-cache

# Clear cache by country or release
o2agol clear-cache --country afg
o2agol clear-cache --release 2025-07-23.0

# Force fresh download (replaces existing)
o2agol overture-dump buildings --country afg --force-download
```

### Troubleshooting
- **Validation errors**: Use `--force-download` to refresh cached data
- **Memory errors**: Reduce `DUMP_MAX_MEMORY` environment variable or use `--limit` for testing
- **Storage issues**: Use `o2agol cache-list` to monitor disk usage
- **Performance**: Use `--use-bbox` for development, `--use-divisions` for production accuracy

## Future Enhancements
- **Advanced Transformations**: Enhanced field mapping and data transformation capabilities
- **Regional Processing**: Support for full Overture theme dumps
- **Interactive CLI**: Guided interface for a better user experience
- **Additional Output Formats**: Support for Shapefile, CSV, and other common formats
- **Cloud Integration**: Native support for cloud storage backends (S3, Azure Blob, GCS)

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutorial here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- Georock's post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/
- Bounding Box extracts from Natural Earth: https://github.com/sandstrom/country-bounding-boxes/tree/master
- This ArcGIS Pro extension that helped inspire this workflow: https://github.com/COF-RyLopez/ArcGISPro-GeoParquet-Addin
