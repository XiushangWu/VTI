import os
import time as t
import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
from shapely import wkb
from data.stats.stats_manager import stats
from data.logs.logging import setup_logger
from .split_tractories import split_trajectories_from_df
from utils import get_radian_and_radian_diff_columns, calculate_initial_compass_bearing, get_haversine_dist_df_in_meters

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
HARBORS_FILE = os.path.join(DATA_FOLDER, 'harbors.csv')
AIS_CSV_FOLDER = os.path.join(DATA_FOLDER, 'ais_csv')
GRAPH_INPUT_FOLDER = os.path.join(DATA_FOLDER, 'imputation_input')
STATISTIC_FOLDER = os.path.join(DATA_FOLDER, 'stats')
STATISTIC_JSON_FILE = os.path.join(STATISTIC_FOLDER, 'stats.ndjson')
CSV_EXTRACT_FILE_LOG = 'ais_extraction_log.txt'
SOG_STANDSTILL = 0.0

pd.set_option('future.no_silent_downcasting', True)

logging = setup_logger(name=CSV_EXTRACT_FILE_LOG, log_file=CSV_EXTRACT_FILE_LOG)

def extract_trajectories_from_csv_files():
    filenames = os.listdir(AIS_CSV_FOLDER)
    completed:int = 0
    
    logging.info(f'Began extracting trajectories from {len(filenames)} csv files')
    try:
        for file_index in range(len(filenames)):
            filename = filenames[file_index]
            filepath = os.path.join(AIS_CSV_FOLDER, filename)

            logging.info(f'Currently extracting file: {filename} (Completed ({completed}/{len(filenames)}) csv files)')        
            df = get_csv_as_df(filepath=filepath)
            completed +=1
                    
            stats.filepath = filepath
            stats.initial_rows = len(df)
            
            # Step 2: Cleanse CSV
            logging.info(f'Cleansing csv {filename}')
            df = cleanse_df(gdf=df)
            
            stats.filtered_rows = len(df)
            
            logging.info(f'Finished extracting file: {filename} (Completed ({completed}/{len(filenames)}) csv files)')        
            logging.info(f'Began crating trajectories for file: {filename}')

            if (not df.empty):
                create_trajectories_files(gdf=df)
            else:
                logging.warning(f'No data was extracted from {filename}')
        
        stats.add_to_file(STATISTIC_JSON_FILE)        
        logging.info('Finished creating trajecatories. Terminating')
    except Exception as e:
        logging.error(f'Failed extracting traectories with {repr(e)}')
        quit()

def get_csv_as_df(filepath:str) -> gpd.GeoDataFrame:
    """
    Takes a .csv file and returns df
    :param file_path: path to file
    :return: A df
    """
    
    types = {
        '# Timestamp': str,
        'Type of mobile': str,
        'MMSI': 'Int32',
        'Navigational status': str,
        'Heading': 'Int16',
        'IMO': 'Int32',
        'Callsign': str,
        'Name': str,
        'Ship type': str,
        'Cargo type': str,
        'Width': 'Int32',
        'Length': 'Int32',
        'Type of position fixing device': str,
        'Destination': str,
        'ETA': str,
        'Data source type': str
    }
    
    df = pd.read_csv(filepath, na_values=['Unknown','Undefined'], dtype=types)#, nrows=1000000)    

    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type', 'Destination', 'Type of position fixing device',
                  'Callsign'],axis=1, errors='ignore')
    
    return df
    
def cleanse_df(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Takes a df and cleanses it according to the set predicates.
    :param df: df to cleanse'
    :return: A cleansed geodataframe, sorted by timestamp (ascending)
    """    
    ship_types = ['fishing', 'tanker', 'tug', 'cargo', 'passenger', 'dredging', 'law enforcement', 'anti-pollution', 'pilot', 'pleasure', 'towing', 'port tender', 'diving', 'towing long/wide', ''] 
      
    # Remove all the rows which does not satisfy our conditions
    gdf = gdf[
            (gdf["Type of mobile"] != "Class B") &
            (gdf["MMSI"].notna()) &
            (gdf["MMSI"].notnull()) &
            (gdf['# Timestamp'].notnull()) &
            (gdf['Latitude'] >=53.5) & (gdf['Latitude'] <=58.5) &
            (gdf['Longitude'] >= 3.2) & (gdf['Longitude'] <=16.5) &
            (gdf['SOG'] <=102) & 
            (gdf['Ship type'].str.lower().isin(ship_types))
    ].reset_index()
    
    subset_columns = ['MMSI', 'Latitude', 'Longitude', '# Timestamp']  # Adjust these based on your actual columns
    gdf = gdf.drop_duplicates(subset=subset_columns, keep='first')
    
    # We round the lat and longs as we do not need 15 decimals of precision
    # This will save some computation time later.
    # We also round rot, sog and cog, as we do not need a lot of decimal precision here
    gdf['Latitude'] = np.round(gdf['Latitude'], decimals=6)
    gdf['Longitude'] = np.round(gdf['Longitude'], decimals=6)
    gdf['ROT'] = np.round(gdf['ROT'], decimals=2)
    gdf['SOG'] = np.round(gdf['SOG'], decimals=2)
    gdf['COG'] = np.round(gdf['COG'], decimals=2)
    gdf['# Timestamp'] = pd.to_datetime(gdf['# Timestamp'], format="%d/%m/%Y %H:%M:%S").sort_values()
    gdf['# Timestamp'].astype('int64')//1e9
    gdf['# Timestamp'] = (gdf['# Timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()    

    # Rename the columns
    gdf = gdf.rename(columns={
            '# Timestamp':'timestamp',
            'Type of mobile':'type_of_mobile',
            'Navigational status':'navigational_status',
            'Ship type':'ship_type',
            'Type of position fixing device':'type_of_position_fixing_device',
        })

    # lower case names in the columns
    gdf.columns = map(str.lower, gdf.columns)
    
    if (gdf.empty):
        return gdf
    
    # Grouping by the columns 'imo', 'name', 'length', 'width', and 'ship_type'
    # and filling missing values within each group with the first non-null value and afterwars last non-value
    gdf[['imo', 'name', 'length', 'width', 'ship_type']] = gdf.groupby('mmsi')[['imo', 'name', 'length', 'width', 'ship_type']].transform(lambda x: x.ffill().bfill()).infer_objects()
    
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf['longitude'], gdf['latitude']), crs="EPSG:4326")
    gdf = gdf.drop(columns=['index'], errors='ignore')
    gdf = gdf.to_crs(epsg="3857") # to calculate in meters
        
    return gdf     
   
def create_trajectories_files(gdf: gpd.GeoDataFrame):
    """Given a nonempty dataframe, splits trajectories and creates txt files. Further creates static file"""
    try:
        
        if (gdf.empty):
            return gdf    

        harbors_df = extract_harbors_df()
        
        trajectories_df = gdf.sort_values('timestamp').groupby('mmsi')
        
        logging.info(f'Began creating trajectories for {trajectories_df.ngroups} mmsi')
        
        removed_due_to_draught_before_split = 0
        removed_due_to_length = 0 
        removed_due_to_draught_after_split = 0
        total_trajectories_after_split = 0
        rows_per_trajectory_after_split = []
        lengths = []
        for mmsi, trajectory_df in trajectories_df:
            if trajectory_df.draught.isnull().all() or trajectory_df.draught.isna().all():
                removed_due_to_draught_before_split += 1
                continue
             
            sub_trajectories_df = split_trajectories_from_df(harbors_df=harbors_df, trajectory_df=trajectory_df) 
                    
            if sub_trajectories_df is not None and not sub_trajectories_df.empty:
                for _, sub_trajectory_df in sub_trajectories_df.groupby('sub_trajectory_id'):
                    if sub_trajectory_df is None or sub_trajectory_df.empty:
                        continue
                    
                    if len(sub_trajectory_df) < 2:
                        removed_due_to_length += 1
                        continue
                    
                    # issue with draught
                    if sub_trajectory_df.draught.isnull().all() or sub_trajectory_df.draught.isna().all():
                        removed_due_to_draught_after_split += 1
                        continue
                    
                    filtered_sub_trajectory = sub_trajectory_df.copy()
                    
                    if sub_trajectory_df.draught.isnull().any() or sub_trajectory_df.draught.isna().any():
                        max_draught = sub_trajectory_df['draught'].max()
                        filtered_sub_trajectory['draught'] = sub_trajectory_df['draught'].fillna(max_draught)

                    # filter away trajectory running in circles
                    # Create a boolean mask to identify rows where the next row has different latitude or longitude
                    mask = (sub_trajectory_df['latitude'] != sub_trajectory_df['latitude'].shift(-1)) | (sub_trajectory_df['longitude'] != sub_trajectory_df['longitude'].shift(-1))

                    # Apply the mask to keep only the rows where the next row has different latitude or longitude
                    filtered_sub_trajectory = sub_trajectory_df[mask]    

                    if len(filtered_sub_trajectory) < 2:
                        removed_due_to_length += 1
                        continue
                    
                    total_trajectories_after_split += 1
                    trajectory_rows = len(filtered_sub_trajectory)
                    rows_per_trajectory_after_split.append(trajectory_rows)
                    distance_travelled = calculate_distance_traveled(trajectory_df=filtered_sub_trajectory)

                    lengths.append(distance_travelled)
                                    
                    write_trajectory_to_input_graph_folder(filtered_sub_trajectory)
        
        stats.trajectory_counts = len(trajectories_df)  
        stats.rows_per_trajectory.extend(trajectories_df.apply(len))
        stats.trajectory_removed_due_to_draught = removed_due_to_draught_before_split
        stats.trajectory_removed_due_to_draught_after_split = removed_due_to_draught_after_split
        stats.trajectory_counts_after_split = total_trajectories_after_split
        stats.rows_per_trajectory_after_split.extend(rows_per_trajectory_after_split)
        stats.distance_travelled_m_per_trajectory_after_split.extend(lengths)
    except Exception as e:
        print(f'Error occurred with error: {repr(e)}')
        return

def extract_harbors_df() -> gpd.GeoDataFrame:    
    df = pd.read_csv(HARBORS_FILE, na_values=['Unknown','Undefined']) 
    df['geometry'] = df['geometry'].apply(lambda x: wkb.loads(bytes.fromhex(x)))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.drop(columns=['index'], errors='ignore')
    gdf = gdf.to_crs(epsg="3857") # to calculate in meters
    
    return gdf  
    
def calculate_distance_traveled(trajectory_df: gpd.GeoDataFrame) -> float:
    gdf_next = trajectory_df.shift(-1)
    trajectory_df, gdf_next = get_radian_and_radian_diff_columns(trajectory_df, gdf_next)
    
    bearing_df = calculate_initial_compass_bearing(df_curr=trajectory_df, df_next=gdf_next)
    trajectory_df['cog'] = trajectory_df['cog'].fillna(bearing_df)

    trajectory_df['dist'] = get_haversine_dist_df_in_meters(df_curr=trajectory_df, df_next=gdf_next).fillna(0)
    distance_travelled = trajectory_df['dist'].sum()

    return distance_travelled

def write_trajectory_to_input_graph_folder(sub_trajectory_df: gpd.GeoDataFrame):
    if (sub_trajectory_df.empty):
        return
        
    datetime_object = dt.datetime.fromtimestamp(sub_trajectory_df.iloc[0].timestamp, tz=dt.timezone.utc)
    str_datetime = datetime_object.strftime('%d/%m/%Y %H:%M:%S').replace('/', '-').replace(' ', '_').replace(':', '-') 
    foldername = str(sub_trajectory_df.iloc[0].mmsi)
    filename = f'{foldername}_{str_datetime}.txt'
    vessel_folder = sub_trajectory_df.iloc[0].ship_type.replace(' ', '_').replace('/', '_')
    folderpath = os.path.join(GRAPH_INPUT_FOLDER, vessel_folder, foldername)
        
    os.makedirs(folderpath, exist_ok=True)  # Create the folder if it doesn't exist
    
    file_path = os.path.join(folderpath, filename)
    
    sub_trajectory_df[['latitude', 'longitude', 'timestamp', 'sog', 'cog', 'draught', 'ship_type', 'navigational_status']].reset_index(drop=True).to_csv(file_path, sep=',', index=True, header=True, mode='w')