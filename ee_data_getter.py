import ee
import pandas as pd
import logging
import os
try:
    import StringIO 
except ImportError:
    from io import StringIO 
from datetime import datetime

# Initialize logging to monitor the background process
logging.basicConfig(
    filename='download_crop_data.log', 
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Authenticate with Earth Engine (if needed)
def authenticate_ee():
    try:
        ee.Initialize()
        logging.info("Earth Engine API initialized successfully.")
    except ee.EEException:
        logging.warning("Authentication required. Please follow the instructions.")
        ee.Authenticate()
        ee.Initialize()

# Load the CSB field boundaries dataset
CSB_ASSET = 'projects/nass-csb/assets/csb1623/CSBAL1623'

def get_fields_for_mclean():
    """Filter fields for McLean County, Illinois."""
    # Load the CSBAL1623 FeatureCollection
    field_boundaries = ee.FeatureCollection(CSB_ASSET)

    # Load the TIGER counties dataset and filter for McLean County in Illinois
    counties = ee.FeatureCollection('TIGER/2018/Counties')
    mclean_county = counties.filter(
        ee.Filter.And(
            ee.Filter.eq('NAME', 'McLean'),  # County name
            ee.Filter.eq('STATEFP', '17')   # Illinois FIPS code
        )
    )

    # Filter the field boundaries for features within McLean County
    mclean_fields = field_boundaries.filterBounds(mclean_county.geometry())
    return mclean_fields

def feature_collection_to_dataframe(fc):
    """Convert a GEE FeatureCollection to a Pandas DataFrame."""
    features = fc.getInfo()['features']
    data = [{**f['properties'], 'geometry': f['geometry']} for f in features]
    return pd.DataFrame(data)

def main():
    authenticate_ee()

    # Get all fields for McLean County
    mclean_fields = get_fields_for_mclean()

    # Convert to DataFrame
    df = feature_collection_to_dataframe(mclean_fields)

    # Save to CSV
    csv_path = "mclean_fields.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved McLean fields to {csv_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")