# Overture Country Dump Guidance
Originally this code was built for individual country queries, however, when querying multiple times a local dump can be more efficient. 

We've created the `overture-dump` command for this. Instead of a full dump, the command downloads a country's dump for size and speed purposes. Below is some guidance on it. These dumps can be large files, so it may take a while on the first run.

## Local Dump Processing:
- `o2agol overture-dump roads --country afg --download-only` - Download transportation theme dump (one-time)
- `o2agol overture-dump roads --country afg` - Process Afghanistan roads from local dump (fast)
- `o2agol overture-dump roads --country pak` - Process Pakistan roads using same dump (fast)
- `o2agol overture-dump buildings --country afg --force-download` - Force fresh download
- `o2agol overture-dump places --country afg --format geojson` - Export to GeoJSON instead of AGOL
- `o2agol list-dumps` - Show all downloaded dumps with metadata
- `o2agol validate-dump 2025-07-23.0 buildings` - Verify dump integrity

## Basic Usage

### 1. Download Theme Dump (One-time)
```bash
# Download specific theme dump
o2agol overture-dump roads --country afg --download-only
```

### 2. Process Multiple Countries
```bash
# Process different countries using same dump
o2agol overture-dump roads --country afg    
o2agol overture-dump roads --country pak      
o2agol overture-dump roads --country bgd     
```

### 3. Batch Processing
```bash
# Process all countries in a region
for country in afg pak bgd ind lka; do
    o2agol overture-dump roads --country $country
done
```

## Advanced Usage

### Development with Limits
```bash
# Download and test with sample data
o2agol overture-dump roads --country afg --limit 1000 --download-only
o2agol overture-dump roads --country afg --limit 1000
```

### Export to GeoJSON
```bash
# Export instead of publishing to AGOL
o2agol overture-dump places --country afg --format geojson
o2agol overture-dump education --country pak --format geojson --output schools.geojson
```

### Force Updates
```bash
# Force download latest release
o2agol overture-dump buildings --country afg --force-download --release 2025-07-23.0
```

### Validation and Management
```bash
# List available dumps
o2agol list-dumps

# Validate dump integrity  
o2agol validate-dump 2025-07-23.0 transportation

# Process with validation
o2agol overture-dump roads --country afg --dry-run --verbose
```

## Configuration

### Environment Variables (Optional)
```bash
# Custom dump location (default: /overturedump)
export OVERTURE_DUMP_PATH=/data/overture

# Memory limits (default: 32GB)
export DUMP_MAX_MEMORY=64

# Countries per processing chunk (default: 5)  
export DUMP_CHUNK_SIZE=10
```

### When to Use Dumps vs S3
**Use Dumps For:**
- Processing multiple countries (>3)
- Repeated processing of same data
- Development and testing workflows
- Batch operations
- Offline/bandwidth-limited environments

**Use S3 For:**
- One-off single country processing
- Always need latest data immediately
- Limited local storage (<200GB available)
- Cloud-native deployments without persistent storage

## Migration Guide

### Step 1: Download Common Dumps
```bash
# Download your most-used themes
o2agol overture-dump roads --download-only
o2agol overture-dump buildings --download-only  
o2agol overture-dump places --download-only
```

### Step 2: Update Scripts
```bash
# Old approach (S3 every time)
o2agol arcgis-upload roads --country afg

# New approach (local dump)  
o2agol overture-dump roads --country afg
```

### Step 3: Batch Processing
```bash
# Process multiple countries efficiently
countries="afg pak bgd ind lka"
for country in $countries; do
    o2agol overture-dump roads --country $country
    o2agol overture-dump buildings --country $country
done