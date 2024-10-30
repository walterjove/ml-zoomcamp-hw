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
        ee.Initialize()  # Try to initialize the Earth Engine API
        logging.info("Earth Engine API initialized successfully.")
    except ee.EEException:
        logging.warning("Authentication required. Please follow the instructions.")
        ee.Authenticate()  # Trigger the authentication flow
        ee.Initialize()  # Re-initialize after authentication

# Define the Corn Belt states
states = ["Iowa", "Illinois"]

# Define the date range for the full month of August 2022
dates = pd.date_range(start="2022-06-01", end="2022-08-31", freq='W')

def get_closest_sentinel_image(county_geom, date):
    start_date = (pd.to_datetime(date) - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(date) + pd.Timedelta(days=3)).strftime('%Y-%m-%d')

    sentinel = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(county_geom) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .sort('system:time_start') \
        .first()

    if sentinel.getInfo() is None:
        logging.warning(f"No Sentinel-2 image available between {start_date} and {end_date}.")
        return None

    return sentinel

def calculate_indices(sentinel):
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

def get_fields_for_county(county_geom, date):
    cdl = ee.ImageCollection("USDA/NASS/CDL") \
        .filterDate('2022-01-01', '2022-12-31').first() \
        .select(['cropland']).clip(county_geom)

    sentinel = get_closest_sentinel_image(county_geom, date)

    if sentinel is None:
        return pd.DataFrame()

    indices = calculate_indices(sentinel)
    masked_indices = indices.updateMask(cdl.gt(0))

    fields = cdl.reduceToVectors(
        geometry=county_geom, scale=300, geometryType='polygon', maxPixels=1e12
    ).randomColumn('random')

    fields = fields.sort('random').map(lambda f: f.set('field_id', f.id()))

    def compute_field_stats(field):
        stats = masked_indices.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=field.geometry(), scale=30, maxPixels=1e12, bestEffort=True
        )
        return field.set(stats)

    field_data = fields.map(compute_field_stats)
    valid_fields = field_data.filter(ee.Filter.notNull(['NDVI']))
    sampled_fields = valid_fields.randomColumn('random').sort('random').limit(200)

    def ee_to_pandas(fc):
        features = fc.getInfo()['features']
        data = [{**f['properties'], 'geometry': f['geometry']} for f in features]
        return pd.DataFrame(data)

    return ee_to_pandas(sampled_fields)

def append_to_csv(data, path):
    """Append DataFrame to CSV, create file with headers if it doesn't exist."""
    if not os.path.isfile(path):
        data.to_csv(path, index=False)  # Write with headers
    else:
        data.to_csv(path, mode='a', header=False, index=False)  # Append without headers

def main():
    authenticate_ee()  # Ensure authentication is handled

    csv_path = "county_level_crop_data_with_indices_and_field_ids.csv"

    for state in states:
        logging.info(f"Processing state: {state}")
        counties = ee.FeatureCollection("TIGER/2018/Counties") \
            .filter(ee.Filter.eq('STATEFP', ee.FeatureCollection("TIGER/2018/States")
                                 .filter(ee.Filter.eq('NAME', state)).first().get('STATEFP')))

        for date in dates:
            for county in counties.toList(counties.size()).getInfo():
                county_geom = ee.Geometry(ee.Feature(county).geometry())
                logging.info(f"Processing {county['properties']['NAME']} on {date.date()}...")

                df = get_fields_for_county(county_geom, date.strftime('%Y-%m-%d'))
                if not df.empty:
                    df['state'] = state
                    df['county'] = county['properties']['NAME']
                    df['date'] = date

                    # Append to CSV file
                    append_to_csv(df, csv_path)
                    logging.info(f"Appended data for {county['properties']['NAME']} on {date.date()} to CSV.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")