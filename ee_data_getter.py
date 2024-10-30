import ee
import pandas as pd
import logging
import os
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

def get_closest_sentinel_image(geometry, date):
    """Find the closest available Sentinel-2 image."""
    start_date = (pd.to_datetime(date) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(date) + pd.Timedelta(days=3)).strftime('%Y-%m-%d')

    sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .sort('system:time_start') \
        .first()

    if sentinel.getInfo() is None:
        logging.warning(f"No Sentinel-2 image available between {start_date} and {end_date}.")
        return None

    return sentinel

def calculate_indices(sentinel):
    """Calculate NDVI, EVI, GNDVI, NDWI, and SAVI from Sentinel-2 data."""
    ndvi = sentinel.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = sentinel.expression(
        '2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))',
        {'B8': sentinel.select('B8'), 'B4': sentinel.select('B4'), 'B2': sentinel.select('B2')}
    ).rename('EVI')
    gndvi = sentinel.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    ndwi = sentinel.normalizedDifference(['B8', 'B11']).rename('NDWI')
    savi = sentinel.expression(
        '((B8 - B4) / (B8 + B4 + 0.5)) * 1.5',
        {'B8': sentinel.select('B8'), 'B4': sentinel.select('B4')}
    ).rename('SAVI')

    return ndvi.addBands([evi, gndvi, ndwi, savi])

def get_data_for_fields(field_boundary_fc, date):
    """Retrieve indices and crop type for each field."""
    sentinel = get_closest_sentinel_image(field_boundary_fc.geometry(), date)

    if sentinel is None:
        return pd.DataFrame()

    indices = calculate_indices(sentinel)

    def compute_field_stats(field):
        stats = indices.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=field.geometry(), scale=10, maxPixels=1e12, bestEffort=True
        )
        return field.set(stats)

    field_data = field_boundary_fc.map(compute_field_stats)
    valid_fields = field_data.filter(ee.Filter.notNull(['NDVI']))

    def ee_to_pandas(fc):
        features = fc.getInfo()['features']
        data = [{**f['properties'], 'geometry': f['geometry']} for f in features]
        return pd.DataFrame(data)

    return ee_to_pandas(valid_fields)

def main():
    authenticate_ee()

    # Load the CSB field boundaries from GEE
    field_boundaries = ee.FeatureCollection(CSB_ASSET)

    csv_path = "csb_field_data.csv"

    # Define the date range
    dates = pd.date_range(start="2022-06-01", end="2022-08-31")

    for date in dates:
        logging.info(f"Processing date: {date}")
        df = get_data_for_fields(field_boundaries, date.strftime('%Y-%m-%d'))

        if not df.empty:
            df['date'] = date
            # Append to CSV
            append_to_csv(df, csv_path)
            logging.info(f"Data for {date} appended to CSV.")

def append_to_csv(data, path):
    """Append DataFrame to CSV."""
    if not os.path.isfile(path):
        data.to_csv(path, index=False)
    else:
        data.to_csv(path, mode='a', header=False, index=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")