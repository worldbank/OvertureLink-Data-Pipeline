# Overture to ArcGIS Online Pipeline

## Purpose
This is a cloud-native ETL pipine to extract Overture Maps data (such as roads, buildings), transform to an AGOL-ready schema, and publish as a feature layer in ArcGIS Online. This is designed to align with Overture's monthly releases and for country-specific data pulls.

## Pipeline Overview
- Ingest: DuckDB reads Overture GeoParquet remotely. We did this so the user doesn't have to download anything.
- Transform: Normalize schema/geometry, retain stable Overture/GERS IDs, define/delete/add metadata fields.
- Publish: ArcGIS Python API `add().publish()` for first run; `overwrite()` for updates.
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

Note: after your first upload, you can add the item ID to the config file for it to be updated. Future plans for this to be automated.

## Config
`configs/<country>.yml` includes:
- `overture_release`: "latest" or a specific date.
- `selector`: Add ISO2 country code 
- `targets`: roads/buildings with layer titles, item IDs (for updates), field maps, filters.
- `mode`: initial/overwrite/append.

### Clip modes
- Using bbox: bbox-only overlap. For development due to its speed. Use `--use-bbox` 
- Overture Division Clip: geometric clip with `ST_Intersects` and a light polygon simplify. Use `--use-divisions` for production (or neither flag, as it is default).

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

### Choosing config file
- `-c configs/afg.yml` - points to your config file

### Modes
- `--mode initial` - for when you first upload an AGOL content item (after copy ID to config file)
- `--mode overwrite` - for when you update an AGOL content item

### Clipping
- `--use-divisions` (default) - Uses Overture Divisions for precise boundaries
- `--use-bbox` - Falls back to bounding box filtering if needed

### Others
- `--limit 1000` - Limits the features for testing purposes
- `--log-to-file` - Logs generate in "/log" for debug purposes
- `--dry-run` - Processes but doesn't publish to AGOL, good for testing without uploading every test.

## Examples:
- `o2agol roads -c configs/afg.yml --mode overwrite --use-bbox --log-to-file` - Updates the Afghanistan roads content item in AGOL using bbox parameters, also logs terminal to file.
- `o2agol roads -c configs/civ.yml --mode initial` - Creates a new roads content item in AGOL for CIV, using the default Overture division clipping

## CI/CD
- `ci.yml`: lint, type-check, unit tests on PR/main.
- `run-pipeline.yml`: manual (`workflow_dispatch`) or schedule (cron). Uses environment secrets.

## Sources
- Overture documentation: https://docs.overturemaps.org/guides/divisions/
- Mark Litwintschik's post on using DuckDB: https://tech.marksblogg.com/duckdb-gis-spatial-extension.html
- Chris Holme's excellent tutoral here: https://github.com/cholmes/duckdb-geoparquet-tutorials/tree/main
- This post on spatial clipping: https://geo.rocks/post/duckdb_geospatial/