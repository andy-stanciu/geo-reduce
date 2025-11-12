import json

def load_city_mapping(json_path='data_cleaning/us_cities.json'):
    """
    Load city data and create mappings.
    
    Returns:
        city_to_idx: dict mapping city name to index (0-49)
        idx_to_city: dict mapping index to city name
        idx_to_coords: dict mapping index to (lat, lon) tuple
        city_names: list of city names in order
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    locations = data['locations']
    
    # Create mappings
    city_to_idx = {}
    idx_to_city = {}
    idx_to_coords = {}
    city_names = []
    
    for idx, loc in enumerate(locations):
        # Extract city name (before comma)
        city_name = loc['name'].split(',')[0].strip()
        
        city_to_idx[city_name] = idx
        idx_to_city[idx] = city_name
        idx_to_coords[idx] = (loc['lat'], loc['lng'])
        city_names.append(city_name)
    
    print(f"Loaded {len(city_names)} cities")
    return city_to_idx, idx_to_city, idx_to_coords, city_names


# For easy access
CITY_TO_IDX, IDX_TO_CITY, IDX_TO_COORDS, CITY_NAMES = load_city_mapping()
NUM_CITIES = len(CITY_NAMES)