"""
Global Country Registry for Overture Maps Pipeline

Centralized database of country metadata including ISO codes, names, regions,
and bounding boxes for spatial filtering operations.

Updated with Natural Earth bounding box data for comprehensive global coverage.
"""

from dataclasses import dataclass


@dataclass
class CountryInfo:
    """Complete country metadata for pipeline operations"""
    name: str
    iso2: str
    iso3: str
    region: str
    bbox: list[float]  # [min_lon, min_lat, max_lon, max_lat]
    
    def __post_init__(self):
        """Validate country data format"""
        if len(self.bbox) != 4:
            raise ValueError(f"Bounding box must have 4 coordinates, got {len(self.bbox)}")
        if not self.iso2 or len(self.iso2) != 2:
            raise ValueError(f"ISO2 code must be 2 characters, got '{self.iso2}'")
        if not self.iso3 or len(self.iso3) != 3:
            raise ValueError(f"ISO3 code must be 3 characters, got '{self.iso3}'")





class CountryRegistry:
    """Global registry for country lookups and operations"""
    
    @staticmethod
    def get_country(identifier: str) -> CountryInfo | None:
        """
        Get country information by ISO2, ISO3, or name
        
        Args:
            identifier: ISO2 code, ISO3 code, or country name
            
        Returns:
            CountryInfo object if found, None otherwise
        """
        identifier_upper = identifier.upper()
        
        # Direct ISO2 lookup
        if identifier_upper in COUNTRIES:
            return COUNTRIES[identifier_upper]
        
        # ISO3 lookup
        for country in COUNTRIES.values():
            if country.iso3.upper() == identifier_upper:
                return country
        
        # Name lookup (case-insensitive)
        identifier_lower = identifier.lower()
        for country in COUNTRIES.values():
            if country.name.lower() == identifier_lower:
                return country
            # Also check for partial matches
            if identifier_lower in country.name.lower():
                return country
        
        return None
    
    @staticmethod
    def list_countries() -> list[CountryInfo]:
        """Get list of all available countries"""
        return list(COUNTRIES.values())
    
    @staticmethod
    def list_regions() -> list[str]:
        """Get list of all available regions"""
        regions = set(country.region for country in COUNTRIES.values())
        return sorted(list(regions))
    
    @staticmethod
    def get_countries_by_region(region: str) -> list[CountryInfo]:
        """Get all countries in a specific region"""
        return [country for country in COUNTRIES.values() 
                if country.region.lower() == region.lower()]
    
    @staticmethod
    def get_bounding_boxes() -> dict[str, list[float]]:
        """Get dictionary of ISO2 codes to bounding boxes (for backward compatibility)"""
        return {iso2: country.bbox for iso2, country in COUNTRIES.items()}
    
    @staticmethod
    def validate_country_code(code: str) -> bool:
        """Check if a country code is valid"""
        return CountryRegistry.get_country(code) is not None
    
    @staticmethod
    def suggest_countries(partial: str) -> list[CountryInfo]:
        """Get list of countries matching partial name or code"""
        partial_lower = partial.lower()
        matches = []
        
        for country in COUNTRIES.values():
            if (partial_lower in country.name.lower() or 
                partial_lower in country.iso2.lower() or 
                partial_lower in country.iso3.lower()):
                matches.append(country)
        
        return matches
    

# Comprehensive country database with Natural Earth bounding boxes
# Bounding boxes use [min_lon, min_lat, max_lon, max_lat] format
COUNTRIES: dict[str, CountryInfo] = {
    'AE': CountryInfo(
        name='United Arab Emirates',
        iso2='AE',
        iso3='ARE',
        region='Middle East & North Africa',
        bbox=[51.58, 22.5, 56.4, 26.06]
    ),
    'AF': CountryInfo(
        name='Afghanistan',
        iso2='AF',
        iso3='AFG',
        region='South Asia',
        bbox=[60.53, 29.32, 75.16, 38.49]
    ),
    'AL': CountryInfo(
        name='Albania',
        iso2='AL',
        iso3='ALB',
        region='Europe',
        bbox=[19.3, 39.62, 21.02, 42.69]
    ),
    'AM': CountryInfo(
        name='Armenia',
        iso2='AM',
        iso3='ARM',
        region='Europe & Central Asia',
        bbox=[43.58, 38.74, 46.51, 41.25]
    ),
    'AO': CountryInfo(
        name='Angola',
        iso2='AO',
        iso3='AGO',
        region='Sub-Saharan Africa',
        bbox=[11.64, -17.93, 24.08, -4.44]
    ),
    'AQ': CountryInfo(
        name='Antarctica',
        iso2='AQ',
        iso3='ATA',
        region='Antarctica',
        bbox=[-180.0, -90.0, 180.0, -63.27]
    ),
    'AR': CountryInfo(
        name='Argentina',
        iso2='AR',
        iso3='ARG',
        region='Latin America & Caribbean',
        bbox=[-73.42, -55.25, -53.63, -21.83]
    ),
    'AT': CountryInfo(
        name='Austria',
        iso2='AT',
        iso3='AUT',
        region='Europe',
        bbox=[9.48, 46.43, 16.98, 49.04]
    ),
    'AU': CountryInfo(
        name='Australia',
        iso2='AU',
        iso3='AUS',
        region='Oceania',
        bbox=[113.34, -43.63, 153.57, -10.67]
    ),
    'AZ': CountryInfo(
        name='Azerbaijan',
        iso2='AZ',
        iso3='AZE',
        region='Europe & Central Asia',
        bbox=[44.79, 38.27, 50.39, 41.86]
    ),
    'BA': CountryInfo(
        name='Bosnia and Herzegovina',
        iso2='BA',
        iso3='BIH',
        region='Europe',
        bbox=[15.75, 42.65, 19.6, 45.23]
    ),
    'BD': CountryInfo(
        name='Bangladesh',
        iso2='BD',
        iso3='BGD',
        region='South Asia',
        bbox=[88.08, 20.67, 92.67, 26.45]
    ),
    'BE': CountryInfo(
        name='Belgium',
        iso2='BE',
        iso3='BEL',
        region='Europe',
        bbox=[2.51, 49.53, 6.16, 51.48]
    ),
    'BF': CountryInfo(
        name='Burkina Faso',
        iso2='BF',
        iso3='BFA',
        region='Sub-Saharan Africa',
        bbox=[-5.47, 9.61, 2.18, 15.12]
    ),
    'BG': CountryInfo(
        name='Bulgaria',
        iso2='BG',
        iso3='BGR',
        region='Europe',
        bbox=[22.38, 41.23, 28.56, 44.23]
    ),
    'BH': CountryInfo(
        name='Bahrain',
        iso2='BH',
        iso3='BHR',
        region='Middle East & North Africa',
        bbox=[50.4, 25.8, 50.7, 26.3]
    ),
    'BI': CountryInfo(
        name='Burundi',
        iso2='BI',
        iso3='BDI',
        region='Sub-Saharan Africa',
        bbox=[29.02, -4.5, 30.75, -2.35]
    ),
    'BJ': CountryInfo(
        name='Benin',
        iso2='BJ',
        iso3='BEN',
        region='Sub-Saharan Africa',
        bbox=[0.77, 6.14, 3.8, 12.24]
    ),
    'BN': CountryInfo(
        name='Brunei',
        iso2='BN',
        iso3='BRN',
        region='Southeast Asia',
        bbox=[114.2, 4.01, 115.45, 5.45]
    ),
    'BO': CountryInfo(
        name='Bolivia',
        iso2='BO',
        iso3='BOL',
        region='Latin America & Caribbean',
        bbox=[-69.59, -22.87, -57.5, -9.76]
    ),
    'BR': CountryInfo(
        name='Brazil',
        iso2='BR',
        iso3='BRA',
        region='Latin America & Caribbean',
        bbox=[-73.99, -33.77, -34.73, 5.24]
    ),
    'BS': CountryInfo(
        name='Bahamas',
        iso2='BS',
        iso3='BHS',
        region='Latin America & Caribbean',
        bbox=[-78.98, 23.71, -77.0, 27.04]
    ),
    'BT': CountryInfo(
        name='Bhutan',
        iso2='BT',
        iso3='BTN',
        region='South Asia',
        bbox=[88.81, 26.72, 92.1, 28.3]
    ),
    'BW': CountryInfo(
        name='Botswana',
        iso2='BW',
        iso3='BWA',
        region='Sub-Saharan Africa',
        bbox=[19.9, -26.83, 29.43, -17.66]
    ),
    'BY': CountryInfo(
        name='Belarus',
        iso2='BY',
        iso3='BLR',
        region='Europe & Central Asia',
        bbox=[23.2, 51.32, 32.69, 56.17]
    ),
    'BZ': CountryInfo(
        name='Belize',
        iso2='BZ',
        iso3='BLZ',
        region='Latin America & Caribbean',
        bbox=[-89.23, 15.89, -88.11, 18.5]
    ),
    'CA': CountryInfo(
        name='Canada',
        iso2='CA',
        iso3='CAN',
        region='North America',
        bbox=[-141.0, 41.68, -52.65, 73.23]
    ),
    'CD': CountryInfo(
        name='Congo (Kinshasa)',
        iso2='CD',
        iso3='COD',
        region='Sub-Saharan Africa',
        bbox=[12.18, -13.26, 31.17, 5.26]
    ),
    'CF': CountryInfo(
        name='Central African Republic',
        iso2='CF',
        iso3='CAF',
        region='Sub-Saharan Africa',
        bbox=[14.46, 2.27, 27.37, 11.14]
    ),
    'CG': CountryInfo(
        name='Congo (Brazzaville)',
        iso2='CG',
        iso3='COG',
        region='Sub-Saharan Africa',
        bbox=[11.09, -5.04, 18.45, 3.73]
    ),
    'CH': CountryInfo(
        name='Switzerland',
        iso2='CH',
        iso3='CHE',
        region='Europe',
        bbox=[6.02, 45.78, 10.44, 47.83]
    ),
    'CI': CountryInfo(
        name='Côte d\'Ivoire',
        iso2='CI',
        iso3='CIV',
        region='Sub-Saharan Africa',
        bbox=[-8.6, 4.34, -2.56, 10.52]
    ),
    'CL': CountryInfo(
        name='Chile',
        iso2='CL',
        iso3='CHL',
        region='Latin America & Caribbean',
        bbox=[-75.64, -55.61, -66.96, -17.58]
    ),
    'CM': CountryInfo(
        name='Cameroon',
        iso2='CM',
        iso3='CMR',
        region='Sub-Saharan Africa',
        bbox=[8.49, 1.73, 16.01, 12.86]
    ),
    'CN': CountryInfo(
        name='China',
        iso2='CN',
        iso3='CHN',
        region='East Asia',
        bbox=[73.68, 18.2, 135.03, 53.46]
    ),
    'CO': CountryInfo(
        name='Colombia',
        iso2='CO',
        iso3='COL',
        region='Latin America & Caribbean',
        bbox=[-78.99, -4.3, -66.88, 12.44]
    ),
    'CR': CountryInfo(
        name='Costa Rica',
        iso2='CR',
        iso3='CRI',
        region='Latin America & Caribbean',
        bbox=[-85.94, 8.23, -82.55, 11.22]
    ),
    'CU': CountryInfo(
        name='Cuba',
        iso2='CU',
        iso3='CUB',
        region='Latin America & Caribbean',
        bbox=[-84.97, 19.86, -74.18, 23.19]
    ),
    'CY': CountryInfo(
        name='Cyprus',
        iso2='CY',
        iso3='CYP',
        region='Europe',
        bbox=[32.26, 34.57, 34.0, 35.17]
    ),
    'CZ': CountryInfo(
        name='Czech Republic',
        iso2='CZ',
        iso3='CZE',
        region='Europe',
        bbox=[12.24, 48.56, 18.85, 51.12]
    ),
    'DE': CountryInfo(
        name='Germany',
        iso2='DE',
        iso3='DEU',
        region='Europe',
        bbox=[5.99, 47.3, 15.02, 54.98]
    ),
    'DJ': CountryInfo(
        name='Djibouti',
        iso2='DJ',
        iso3='DJI',
        region='Sub-Saharan Africa',
        bbox=[41.66, 10.93, 43.32, 12.7]
    ),
    'DK': CountryInfo(
        name='Denmark',
        iso2='DK',
        iso3='DNK',
        region='Europe',
        bbox=[8.09, 54.8, 12.69, 57.73]
    ),
    'DO': CountryInfo(
        name='Dominican Republic',
        iso2='DO',
        iso3='DOM',
        region='Latin America & Caribbean',
        bbox=[-71.95, 17.6, -68.32, 19.88]
    ),
    'DZ': CountryInfo(
        name='Algeria',
        iso2='DZ',
        iso3='DZA',
        region='Middle East & North Africa',
        bbox=[-8.68, 19.06, 12.0, 37.12]
    ),
    'EC': CountryInfo(
        name='Ecuador',
        iso2='EC',
        iso3='ECU',
        region='Latin America & Caribbean',
        bbox=[-80.97, -4.96, -75.23, 1.38]
    ),
    'EE': CountryInfo(
        name='Estonia',
        iso2='EE',
        iso3='EST',
        region='Europe',
        bbox=[23.34, 57.47, 28.13, 59.61]
    ),
    'EG': CountryInfo(
        name='Egypt',
        iso2='EG',
        iso3='EGY',
        region='Middle East & North Africa',
        bbox=[24.7, 22.0, 36.87, 31.59]
    ),
    'ER': CountryInfo(
        name='Eritrea',
        iso2='ER',
        iso3='ERI',
        region='Sub-Saharan Africa',
        bbox=[36.32, 12.46, 43.08, 18.0]
    ),
    'ES': CountryInfo(
        name='Spain',
        iso2='ES',
        iso3='ESP',
        region='Europe',
        bbox=[-9.39, 35.95, 3.04, 43.75]
    ),
    'ET': CountryInfo(
        name='Ethiopia',
        iso2='ET',
        iso3='ETH',
        region='Sub-Saharan Africa',
        bbox=[32.95, 3.42, 47.79, 14.96]
    ),
    'FI': CountryInfo(
        name='Finland',
        iso2='FI',
        iso3='FIN',
        region='Europe',
        bbox=[20.65, 59.85, 31.52, 70.16]
    ),
    'FJ': CountryInfo(
        name='Fiji',
        iso2='FJ',
        iso3='FJI',
        region='Oceania',
        bbox=[-180.0, -18.29, 180.0, -16.02]
    ),
    'FK': CountryInfo(
        name='Falkland Islands',
        iso2='FK',
        iso3='FLK',
        region='Other',
        bbox=[-61.2, -52.3, -57.75, -51.1]
    ),
    'FR': CountryInfo(
        name='France',
        iso2='FR',
        iso3='FRA',
        region='Europe',
        bbox=[-5.0, 42.5, 9.56, 51.15]
    ),
    'GA': CountryInfo(
        name='Gabon',
        iso2='GA',
        iso3='GAB',
        region='Sub-Saharan Africa',
        bbox=[8.8, -3.98, 14.43, 2.33]
    ),
    'GB': CountryInfo(
        name='United Kingdom',
        iso2='GB',
        iso3='GBR',
        region='Europe',
        bbox=[-7.57, 49.96, 1.68, 58.64]
    ),
    'GE': CountryInfo(
        name='Georgia',
        iso2='GE',
        iso3='GEO',
        region='Europe & Central Asia',
        bbox=[39.96, 41.06, 46.64, 43.55]
    ),
    'GH': CountryInfo(
        name='Ghana',
        iso2='GH',
        iso3='GHA',
        region='Sub-Saharan Africa',
        bbox=[-3.24, 4.71, 1.06, 11.1]
    ),
    'GL': CountryInfo(
        name='Greenland',
        iso2='GL',
        iso3='GRL',
        region='Other',
        bbox=[-73.3, 60.04, -12.21, 83.65]
    ),
    'GM': CountryInfo(
        name='Gambia',
        iso2='GM',
        iso3='GMB',
        region='Sub-Saharan Africa',
        bbox=[-16.84, 13.13, -13.84, 13.88]
    ),
    'GN': CountryInfo(
        name='Guinea',
        iso2='GN',
        iso3='GIN',
        region='Sub-Saharan Africa',
        bbox=[-15.13, 7.31, -7.83, 12.59]
    ),
    'GQ': CountryInfo(
        name='Equatorial Guinea',
        iso2='GQ',
        iso3='GNQ',
        region='Sub-Saharan Africa',
        bbox=[9.31, 1.01, 11.29, 2.28]
    ),
    'GR': CountryInfo(
        name='Greece',
        iso2='GR',
        iso3='GRC',
        region='Europe',
        bbox=[20.15, 34.92, 26.6, 41.83]
    ),
    'GT': CountryInfo(
        name='Guatemala',
        iso2='GT',
        iso3='GTM',
        region='Latin America & Caribbean',
        bbox=[-92.23, 13.74, -88.23, 17.82]
    ),
    'GW': CountryInfo(
        name='Guinea Bissau',
        iso2='GW',
        iso3='GNB',
        region='Sub-Saharan Africa',
        bbox=[-16.68, 11.04, -13.7, 12.63]
    ),
    'GY': CountryInfo(
        name='Guyana',
        iso2='GY',
        iso3='GUY',
        region='Latin America & Caribbean',
        bbox=[-61.41, 1.27, -56.54, 8.37]
    ),
    'HN': CountryInfo(
        name='Honduras',
        iso2='HN',
        iso3='HND',
        region='Latin America & Caribbean',
        bbox=[-89.35, 12.98, -83.15, 16.01]
    ),
    'HR': CountryInfo(
        name='Croatia',
        iso2='HR',
        iso3='HRV',
        region='Europe',
        bbox=[13.66, 42.48, 19.39, 46.5]
    ),
    'HT': CountryInfo(
        name='Haiti',
        iso2='HT',
        iso3='HTI',
        region='Latin America & Caribbean',
        bbox=[-74.46, 18.03, -71.62, 19.92]
    ),
    'HU': CountryInfo(
        name='Hungary',
        iso2='HU',
        iso3='HUN',
        region='Europe',
        bbox=[16.2, 45.76, 22.71, 48.62]
    ),
    'ID': CountryInfo(
        name='Indonesia',
        iso2='ID',
        iso3='IDN',
        region='Southeast Asia',
        bbox=[95.29, -10.36, 141.03, 5.48]
    ),
    'IE': CountryInfo(
        name='Ireland',
        iso2='IE',
        iso3='IRL',
        region='Europe',
        bbox=[-9.98, 51.67, -6.03, 55.13]
    ),
    'IL': CountryInfo(
        name='Israel',
        iso2='IL',
        iso3='ISR',
        region='Middle East & North Africa',
        bbox=[34.27, 29.5, 35.84, 33.28]
    ),
    'IN': CountryInfo(
        name='India',
        iso2='IN',
        iso3='IND',
        region='South Asia',
        bbox=[68.18, 7.97, 97.4, 35.49]
    ),
    'IQ': CountryInfo(
        name='Iraq',
        iso2='IQ',
        iso3='IRQ',
        region='Middle East & North Africa',
        bbox=[38.79, 29.1, 48.57, 37.39]
    ),
    'IR': CountryInfo(
        name='Iran',
        iso2='IR',
        iso3='IRN',
        region='Middle East & North Africa',
        bbox=[44.11, 25.08, 63.32, 39.71]
    ),
    'IS': CountryInfo(
        name='Iceland',
        iso2='IS',
        iso3='ISL',
        region='Europe',
        bbox=[-24.33, 63.5, -13.61, 66.53]
    ),
    'IT': CountryInfo(
        name='Italy',
        iso2='IT',
        iso3='ITA',
        region='Europe',
        bbox=[6.75, 36.62, 18.48, 47.12]
    ),
    'JM': CountryInfo(
        name='Jamaica',
        iso2='JM',
        iso3='JAM',
        region='Latin America & Caribbean',
        bbox=[-78.34, 17.7, -76.2, 18.52]
    ),
    'JO': CountryInfo(
        name='Jordan',
        iso2='JO',
        iso3='JOR',
        region='Middle East & North Africa',
        bbox=[34.92, 29.2, 39.2, 33.38]
    ),
    'JP': CountryInfo(
        name='Japan',
        iso2='JP',
        iso3='JPN',
        region='East Asia',
        bbox=[129.41, 31.03, 145.54, 45.55]
    ),
    'KE': CountryInfo(
        name='Kenya',
        iso2='KE',
        iso3='KEN',
        region='Sub-Saharan Africa',
        bbox=[33.89, -4.68, 41.86, 5.51]
    ),
    'KG': CountryInfo(
        name='Kyrgyzstan',
        iso2='KG',
        iso3='KGZ',
        region='Europe & Central Asia',
        bbox=[69.46, 39.28, 80.26, 43.3]
    ),
    'KH': CountryInfo(
        name='Cambodia',
        iso2='KH',
        iso3='KHM',
        region='Southeast Asia',
        bbox=[102.35, 10.49, 107.61, 14.57]
    ),
    'KP': CountryInfo(
        name='North Korea',
        iso2='KP',
        iso3='PRK',
        region='East Asia',
        bbox=[124.27, 37.67, 130.78, 42.99]
    ),
    'KR': CountryInfo(
        name='South Korea',
        iso2='KR',
        iso3='KOR',
        region='East Asia',
        bbox=[126.12, 34.39, 129.47, 38.61]
    ),
    'KW': CountryInfo(
        name='Kuwait',
        iso2='KW',
        iso3='KWT',
        region='Middle East & North Africa',
        bbox=[46.57, 28.53, 48.42, 30.06]
    ),
    'KZ': CountryInfo(
        name='Kazakhstan',
        iso2='KZ',
        iso3='KAZ',
        region='Europe & Central Asia',
        bbox=[46.47, 40.66, 87.36, 55.39]
    ),
    'LA': CountryInfo(
        name='Laos',
        iso2='LA',
        iso3='LAO',
        region='Southeast Asia',
        bbox=[100.12, 13.88, 107.56, 22.46]
    ),
    'LB': CountryInfo(
        name='Lebanon',
        iso2='LB',
        iso3='LBN',
        region='Middle East & North Africa',
        bbox=[35.13, 33.09, 36.61, 34.64]
    ),
    'LK': CountryInfo(
        name='Sri Lanka',
        iso2='LK',
        iso3='LKA',
        region='South Asia',
        bbox=[79.7, 5.97, 81.79, 9.82]
    ),
    'LR': CountryInfo(
        name='Liberia',
        iso2='LR',
        iso3='LBR',
        region='Sub-Saharan Africa',
        bbox=[-11.44, 4.36, -7.54, 8.54]
    ),
    'LS': CountryInfo(
        name='Lesotho',
        iso2='LS',
        iso3='LSO',
        region='Sub-Saharan Africa',
        bbox=[27.0, -30.65, 29.33, -28.65]
    ),
    'LT': CountryInfo(
        name='Lithuania',
        iso2='LT',
        iso3='LTU',
        region='Europe',
        bbox=[21.06, 53.91, 26.59, 56.37]
    ),
    'LU': CountryInfo(
        name='Luxembourg',
        iso2='LU',
        iso3='LUX',
        region='Europe',
        bbox=[5.67, 49.44, 6.24, 50.13]
    ),
    'LV': CountryInfo(
        name='Latvia',
        iso2='LV',
        iso3='LVA',
        region='Europe',
        bbox=[21.06, 55.62, 28.18, 57.97]
    ),
    'LY': CountryInfo(
        name='Libya',
        iso2='LY',
        iso3='LBY',
        region='Middle East & North Africa',
        bbox=[9.32, 19.58, 25.16, 33.14]
    ),
    'MA': CountryInfo(
        name='Morocco',
        iso2='MA',
        iso3='MAR',
        region='Middle East & North Africa',
        bbox=[-17.02, 21.42, -1.12, 35.76]
    ),
    'MD': CountryInfo(
        name='Moldova',
        iso2='MD',
        iso3='MDA',
        region='Europe & Central Asia',
        bbox=[26.62, 45.49, 30.02, 48.47]
    ),
    'ME': CountryInfo(
        name='Montenegro',
        iso2='ME',
        iso3='MNE',
        region='Europe',
        bbox=[18.45, 41.88, 20.34, 43.52]
    ),
    'MG': CountryInfo(
        name='Madagascar',
        iso2='MG',
        iso3='MDG',
        region='Sub-Saharan Africa',
        bbox=[43.25, -25.6, 50.48, -12.04]
    ),
    'MK': CountryInfo(
        name='Macedonia',
        iso2='MK',
        iso3='MKD',
        region='Europe',
        bbox=[20.46, 40.84, 22.95, 42.32]
    ),
    'ML': CountryInfo(
        name='Mali',
        iso2='ML',
        iso3='MLI',
        region='Sub-Saharan Africa',
        bbox=[-12.17, 10.1, 4.27, 24.97]
    ),
    'MM': CountryInfo(
        name='Myanmar',
        iso2='MM',
        iso3='MMR',
        region='Southeast Asia',
        bbox=[92.3, 9.93, 101.18, 28.34]
    ),
    'MN': CountryInfo(
        name='Mongolia',
        iso2='MN',
        iso3='MNG',
        region='East Asia',
        bbox=[87.75, 41.6, 119.77, 52.05]
    ),
    'MR': CountryInfo(
        name='Mauritania',
        iso2='MR',
        iso3='MRT',
        region='Sub-Saharan Africa',
        bbox=[-17.06, 14.62, -4.92, 27.4]
    ),
    'MV': CountryInfo(
        name='Maldives',
        iso2='MV',
        iso3='MDV',
        region='South Asia',
        bbox=[72.6, -0.7, 73.8, 7.1]
    ),
    'MW': CountryInfo(
        name='Malawi',
        iso2='MW',
        iso3='MWI',
        region='Sub-Saharan Africa',
        bbox=[32.69, -16.8, 35.77, -9.23]
    ),
    'MX': CountryInfo(
        name='Mexico',
        iso2='MX',
        iso3='MEX',
        region='Latin America & Caribbean',
        bbox=[-117.13, 14.54, -86.81, 32.72]
    ),
    'MY': CountryInfo(
        name='Malaysia',
        iso2='MY',
        iso3='MYS',
        region='Southeast Asia',
        bbox=[100.09, 0.77, 119.18, 6.93]
    ),
    'MZ': CountryInfo(
        name='Mozambique',
        iso2='MZ',
        iso3='MOZ',
        region='Sub-Saharan Africa',
        bbox=[30.18, -26.74, 40.78, -10.32]
    ),
    'NA': CountryInfo(
        name='Namibia',
        iso2='NA',
        iso3='NAM',
        region='Sub-Saharan Africa',
        bbox=[11.73, -29.05, 25.08, -16.94]
    ),
    'NC': CountryInfo(
        name='New Caledonia',
        iso2='NC',
        iso3='NCL',
        region='Other',
        bbox=[164.03, -22.4, 167.12, -20.11]
    ),
    'NE': CountryInfo(
        name='Niger',
        iso2='NE',
        iso3='NER',
        region='Sub-Saharan Africa',
        bbox=[0.3, 11.66, 15.9, 23.47]
    ),
    'NG': CountryInfo(
        name='Nigeria',
        iso2='NG',
        iso3='NGA',
        region='Sub-Saharan Africa',
        bbox=[2.69, 4.24, 14.58, 13.87]
    ),
    'NI': CountryInfo(
        name='Nicaragua',
        iso2='NI',
        iso3='NIC',
        region='Latin America & Caribbean',
        bbox=[-87.67, 10.73, -83.15, 15.02]
    ),
    'NL': CountryInfo(
        name='Netherlands',
        iso2='NL',
        iso3='NLD',
        region='Europe',
        bbox=[3.31, 50.8, 7.09, 53.51]
    ),
    'NO': CountryInfo(
        name='Norway',
        iso2='NO',
        iso3='NOR',
        region='Europe',
        bbox=[4.99, 58.08, 31.29, 70.92]
    ),
    'NP': CountryInfo(
        name='Nepal',
        iso2='NP',
        iso3='NPL',
        region='South Asia',
        bbox=[80.09, 26.4, 88.17, 30.42]
    ),
    'NZ': CountryInfo(
        name='New Zealand',
        iso2='NZ',
        iso3='NZL',
        region='Oceania',
        bbox=[166.51, -46.64, 178.52, -34.45]
    ),
    'OM': CountryInfo(
        name='Oman',
        iso2='OM',
        iso3='OMN',
        region='Middle East & North Africa',
        bbox=[52.0, 16.65, 59.81, 26.4]
    ),
    'PA': CountryInfo(
        name='Panama',
        iso2='PA',
        iso3='PAN',
        region='Latin America & Caribbean',
        bbox=[-82.97, 7.22, -77.24, 9.61]
    ),
    'PE': CountryInfo(
        name='Peru',
        iso2='PE',
        iso3='PER',
        region='Latin America & Caribbean',
        bbox=[-81.41, -18.35, -68.67, -0.06]
    ),
    'PG': CountryInfo(
        name='Papua New Guinea',
        iso2='PG',
        iso3='PNG',
        region='Oceania',
        bbox=[141.0, -10.65, 156.02, -2.5]
    ),
    'PH': CountryInfo(
        name='Philippines',
        iso2='PH',
        iso3='PHL',
        region='Southeast Asia',
        bbox=[117.17, 5.58, 126.54, 18.51]
    ),
    'PK': CountryInfo(
        name='Pakistan',
        iso2='PK',
        iso3='PAK',
        region='South Asia',
        bbox=[60.87, 23.69, 77.84, 37.13]
    ),
    'PL': CountryInfo(
        name='Poland',
        iso2='PL',
        iso3='POL',
        region='Europe',
        bbox=[14.07, 49.03, 24.03, 54.85]
    ),
    'PR': CountryInfo(
        name='Puerto Rico',
        iso2='PR',
        iso3='PRI',
        region='Other',
        bbox=[-67.24, 17.95, -65.59, 18.52]
    ),
    'PS': CountryInfo(
        name='Palestine',
        iso2='PS',
        iso3='PSE',
        region='Middle East & North Africa',
        bbox=[34.93, 31.35, 35.55, 32.53]
    ),
    'PT': CountryInfo(
        name='Portugal',
        iso2='PT',
        iso3='PRT',
        region='Europe',
        bbox=[-9.53, 36.84, -6.39, 42.28]
    ),
    'PY': CountryInfo(
        name='Paraguay',
        iso2='PY',
        iso3='PRY',
        region='Latin America & Caribbean',
        bbox=[-62.69, -27.55, -54.29, -19.34]
    ),
    'QA': CountryInfo(
        name='Qatar',
        iso2='QA',
        iso3='QAT',
        region='Middle East & North Africa',
        bbox=[50.74, 24.56, 51.61, 26.11]
    ),
    'RO': CountryInfo(
        name='Romania',
        iso2='RO',
        iso3='ROU',
        region='Europe',
        bbox=[20.22, 43.69, 29.63, 48.22]
    ),
    'RS': CountryInfo(
        name='Serbia',
        iso2='RS',
        iso3='SRB',
        region='Europe',
        bbox=[18.83, 42.25, 22.99, 46.17]
    ),
    'RU': CountryInfo(
        name='Russia',
        iso2='RU',
        iso3='RUS',
        region='Europe',
        bbox=[-180.0, 41.15, 180.0, 81.25]
    ),
    'RW': CountryInfo(
        name='Rwanda',
        iso2='RW',
        iso3='RWA',
        region='Sub-Saharan Africa',
        bbox=[29.02, -2.92, 30.82, -1.13]
    ),
    'SA': CountryInfo(
        name='Saudi Arabia',
        iso2='SA',
        iso3='SAU',
        region='Middle East & North Africa',
        bbox=[34.63, 16.35, 55.67, 32.16]
    ),
    'SB': CountryInfo(
        name='Solomon Islands',
        iso2='SB',
        iso3='SLB',
        region='East Asia & Pacific',
        bbox=[156.49, -10.83, 162.4, -6.6]
    ),
    'SD': CountryInfo(
        name='Sudan',
        iso2='SD',
        iso3='SDN',
        region='Middle East & North Africa',
        bbox=[21.94, 8.62, 38.41, 22.0]
    ),
    'SE': CountryInfo(
        name='Sweden',
        iso2='SE',
        iso3='SWE',
        region='Europe',
        bbox=[11.03, 55.36, 23.9, 69.11]
    ),
    'SG': CountryInfo(
        name='Singapore',
        iso2='SG',
        iso3='SGP',
        region='Southeast Asia',
        bbox=[103.6, 1.2, 104.0, 1.5]
    ),
    'SI': CountryInfo(
        name='Slovenia',
        iso2='SI',
        iso3='SVN',
        region='Europe',
        bbox=[13.7, 45.45, 16.56, 46.85]
    ),
    'SK': CountryInfo(
        name='Slovakia',
        iso2='SK',
        iso3='SVK',
        region='Europe',
        bbox=[16.88, 47.76, 22.56, 49.57]
    ),
    'SL': CountryInfo(
        name='Sierra Leone',
        iso2='SL',
        iso3='SLE',
        region='Sub-Saharan Africa',
        bbox=[-13.25, 6.79, -10.23, 10.05]
    ),
    'SN': CountryInfo(
        name='Senegal',
        iso2='SN',
        iso3='SEN',
        region='Sub-Saharan Africa',
        bbox=[-17.63, 12.33, -11.47, 16.6]
    ),
    'SO': CountryInfo(
        name='Somalia',
        iso2='SO',
        iso3='SOM',
        region='Sub-Saharan Africa',
        bbox=[40.98, -1.68, 51.13, 12.02]
    ),
    'SR': CountryInfo(
        name='Suriname',
        iso2='SR',
        iso3='SUR',
        region='Latin America & Caribbean',
        bbox=[-58.04, 1.82, -53.96, 6.03]
    ),
    'SS': CountryInfo(
        name='South Sudan',
        iso2='SS',
        iso3='SSD',
        region='Sub-Saharan Africa',
        bbox=[23.89, 3.51, 35.3, 12.25]
    ),
    'SV': CountryInfo(
        name='El Salvador',
        iso2='SV',
        iso3='SLV',
        region='Latin America & Caribbean',
        bbox=[-90.1, 13.15, -87.72, 14.42]
    ),
    'SY': CountryInfo(
        name='Syria',
        iso2='SY',
        iso3='SYR',
        region='Middle East & North Africa',
        bbox=[35.7, 32.31, 42.35, 37.23]
    ),
    'SZ': CountryInfo(
        name='Swaziland',
        iso2='SZ',
        iso3='SWZ',
        region='Sub-Saharan Africa',
        bbox=[30.68, -27.29, 32.07, -25.66]
    ),
    'TD': CountryInfo(
        name='Chad',
        iso2='TD',
        iso3='TCD',
        region='Sub-Saharan Africa',
        bbox=[13.54, 7.42, 23.89, 23.41]
    ),
    'TF': CountryInfo(
        name='French Southern Territories',
        iso2='TF',
        iso3='ATF',
        region='Other',
        bbox=[68.72, -49.78, 70.56, -48.63]
    ),
    'TG': CountryInfo(
        name='Togo',
        iso2='TG',
        iso3='TGO',
        region='Sub-Saharan Africa',
        bbox=[-0.05, 5.93, 1.87, 11.02]
    ),
    'TH': CountryInfo(
        name='Thailand',
        iso2='TH',
        iso3='THA',
        region='Southeast Asia',
        bbox=[97.38, 5.69, 105.59, 20.42]
    ),
    'TJ': CountryInfo(
        name='Tajikistan',
        iso2='TJ',
        iso3='TJK',
        region='Europe & Central Asia',
        bbox=[67.44, 36.74, 74.98, 40.96]
    ),
    'TL': CountryInfo(
        name='Timor-Leste',
        iso2='TL',
        iso3='TLS',
        region='Southeast Asia',
        bbox=[124.97, -9.39, 127.34, -8.27]
    ),
    'TM': CountryInfo(
        name='Turkmenistan',
        iso2='TM',
        iso3='TKM',
        region='Europe & Central Asia',
        bbox=[52.5, 35.27, 66.55, 42.75]
    ),
    'TN': CountryInfo(
        name='Tunisia',
        iso2='TN',
        iso3='TUN',
        region='Middle East & North Africa',
        bbox=[7.52, 30.31, 11.49, 37.35]
    ),
    'TR': CountryInfo(
        name='Turkey',
        iso2='TR',
        iso3='TUR',
        region='Middle East & North Africa',
        bbox=[26.04, 35.82, 44.79, 42.14]
    ),
    'TT': CountryInfo(
        name='Trinidad and Tobago',
        iso2='TT',
        iso3='TTO',
        region='Latin America & Caribbean',
        bbox=[-61.95, 10.0, -60.9, 10.89]
    ),
    'TW': CountryInfo(
        name='Taiwan',
        iso2='TW',
        iso3='TWN',
        region='East Asia',
        bbox=[120.11, 21.97, 121.95, 25.3]
    ),
    'TZ': CountryInfo(
        name='Tanzania',
        iso2='TZ',
        iso3='TZA',
        region='Sub-Saharan Africa',
        bbox=[29.34, -11.72, 40.32, -0.95]
    ),
    'UA': CountryInfo(
        name='Ukraine',
        iso2='UA',
        iso3='UKR',
        region='Europe',
        bbox=[22.09, 44.36, 40.08, 52.34]
    ),
    'UG': CountryInfo(
        name='Uganda',
        iso2='UG',
        iso3='UGA',
        region='Sub-Saharan Africa',
        bbox=[29.58, -1.44, 35.04, 4.25]
    ),
    'US': CountryInfo(
        name='United States',
        iso2='US',
        iso3='USA',
        region='North America',
        bbox=[-125.0, 25.0, -66.96, 49.5]
    ),
    'UY': CountryInfo(
        name='Uruguay',
        iso2='UY',
        iso3='URY',
        region='Latin America & Caribbean',
        bbox=[-58.43, -34.95, -53.21, -30.11]
    ),
    'UZ': CountryInfo(
        name='Uzbekistan',
        iso2='UZ',
        iso3='UZB',
        region='Europe & Central Asia',
        bbox=[55.93, 37.14, 73.06, 45.59]
    ),
    'VE': CountryInfo(
        name='Venezuela',
        iso2='VE',
        iso3='VEN',
        region='Latin America & Caribbean',
        bbox=[-73.3, 0.72, -59.76, 12.16]
    ),
    'VN': CountryInfo(
        name='Vietnam',
        iso2='VN',
        iso3='VNM',
        region='Southeast Asia',
        bbox=[102.17, 8.6, 109.34, 23.35]
    ),
    'VU': CountryInfo(
        name='Vanuatu',
        iso2='VU',
        iso3='VUT',
        region='East Asia & Pacific',
        bbox=[166.63, -16.6, 167.84, -14.63]
    ),
    'YE': CountryInfo(
        name='Yemen',
        iso2='YE',
        iso3='YEM',
        region='Middle East & North Africa',
        bbox=[42.6, 12.59, 53.11, 19.0]
    ),
    'ZA': CountryInfo(
        name='South Africa',
        iso2='ZA',
        iso3='ZAF',
        region='Sub-Saharan Africa',
        bbox=[16.34, -34.82, 32.83, -22.09]
    ),
    'ZM': CountryInfo(
        name='Zambia',
        iso2='ZM',
        iso3='ZMB',
        region='Sub-Saharan Africa',
        bbox=[21.89, -17.96, 33.49, -8.24]
    ),
    'ZW': CountryInfo(
        name='Zimbabwe',
        iso2='ZW',
        iso3='ZWE',
        region='Sub-Saharan Africa',
        bbox=[25.26, -22.27, 32.85, -15.51]
    ),

}