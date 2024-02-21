import os
import tarfile
import zipfile
import numpy as np
import requests
import pandas as pd
import geopandas as gpd
import datetime as dt
import logging 
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sys import stdout

INPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
GTI_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), '../../GTI/data/ais')

logging.basicConfig(level=logging.INFO)
    
def get_csv_files_in_interval(interval: str):
    """
    Downloads and processes all the available ais data in the given interval.
    :param interval: The interval (date) to download and process.
    """
    dates = interval.split('::')
    csv_files_on_server = connect_to_to_ais_web_server_and_get_data()
    begin_index = None
    end_index = None

    for csv_file in csv_files_on_server:
        if begin_index is None:
            if dates[0] in csv_file:
                begin_index = csv_files_on_server.index(csv_file)
                continue
        if end_index is None:
            if dates[1] in csv_file:
                end_index = csv_files_on_server.index(csv_file)

    files_to_download = csv_files_on_server[begin_index:end_index]

    file: str
    downloaded:int = 0
    
    logging.info('Began extracting csv files')
    
    for file in files_to_download:
        extract_csv_file(file_name=file)
        downloaded += 1
        stdout.write(f'\rDownloaded {file}.  Completed ({downloaded}/{len(files_to_download)})')
        stdout.flush()
    logging.info('\nCompleted extracting csv files')

def connect_to_to_ais_web_server_and_get_data():
    """
    Connects to the ais web server and gets the .csv files (ais data) located there.
    :param logging: A logging for loggin warning/errors
    :return: A list with the names of the .zip/.rar files available for download on the web server. Example: 'aisdk-2022-01-01.zip'
    """
    logging.info('Began retrievel of data from https://web.ais.dk/aisdata/')
    
    html = urlopen("https://web.ais.dk/aisdata/")
    
    try:    
        soup = BeautifulSoup(html, 'html.parser')
        
        all_links = soup.find_all('a', href=lambda href: True and 'aisdk' in href)
        all_links_as_string = [link.string for link in all_links]

        logging.info(f'\nCompleted fetching {len(all_links_as_string)} ais download links')
        return all_links_as_string
    except Exception as e:
        logging.error('Fetching AIS data failed with: %s', repr(e))

def extract_csv_file(file_name: str):
    """
    Downloads the given file, runs it through the pipeline and adds the file to the log.
    :param file_name: The file to be downloaded, cleansed and inserted
    """
    download_file_from_ais_web_server(file_name)

    try:
        if ".zip" in file_name: 
            file_name = file_name.replace('.zip', '.csv')
        else:
            file_name = file_name.replace('.rar', '.csv')
        # df = cleanse_csv_file_and_convert_to_df(file_name=file_name)
    except Exception as e:
        logging.error(f'Failed to unzip file {file_name} with error message: {repr(e)}')

def download_file_from_ais_web_server(file_name: str):
    """
    Downloads a specified file from the webserver into the CSV_FILES_FOLDER.
    It will also unzip it, as well as delete the compressed file afterwards.
    :param file_name: The file to be downloaded. Example 'aisdk-2022-01-01.zip'
    """
    download_result = requests.get('https://web.ais.dk/aisdata/' + file_name, allow_redirects=True)
    download_result.raise_for_status()

    path_to_compressed_file = INPUT_FOLDER + file_name

    try:
        f = open(path_to_compressed_file,'wb')
        f.write(download_result.content)
    except Exception as e:
        logging.exception(f'Failed to retrieve file from ais web server, with messega: {repr(e)}')
    finally:
        f.close()
    
    try:
        if ".zip" in path_to_compressed_file: 
            with zipfile.ZipFile(path_to_compressed_file, 'r') as zip_ref:
                zip_ref.extractall(INPUT_FOLDER)
        elif ".rar" in path_to_compressed_file:
            with tarfile.RarFile(path_to_compressed_file) as rar_ref:
                rar_ref.extractall(path=INPUT_FOLDER)
        else:
            logging.error(f'File {file_name} must either be of type .zip or .rar. Not extracted')
            
        os.remove(path_to_compressed_file)

    except Exception as e:
        logging.exception(f'Failed with error: {e}')
        quit()

def extract_trajectories_from_csv_files():
    file_names = os.listdir(INPUT_FOLDER)
    existing_trips = os.listdir(OUTPUT_FOLDER)
    trajectory_id:int = len(existing_trips)
    completed:int = 0
    
    logging.info(f'Began extracting trajectories from {len(file_names)} csv files')
    
    for file_index in range(len(file_names)):
        file_name = file_names[file_index]
        
        stdout.write(f'\rCurrently extracting file: {file_name} (Completed ({completed}/{len(file_names)}) csv files)')
        stdout.flush()
        
        completed +=1
        
        df = cleanse_csv_file_for_specific_mmsi_and_covert_to_df(file_name, 219423000)
        trajectory_id = create_trajectories(trajectory_id, df)
        print('\ndone')
    
    logging.info('Finished creating trajecatories. Terminating')

def cleanse_csv_file_and_convert_to_df(file_name: str):
    """
    Takes a .csv file and cleanses it according to the set predicates.
    :param file_name: File name to cleanse. Example: 'aisdk-2022-01-01.csv'
    :return: A cleansed geodataframe, sorted by timestamp (ascending)
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

    df = pd.read_csv(str.format("{0}/{1}", INPUT_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types) #, nrows=1000000)    
    
    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type', 'Destination', 'Length', 'Width', 'Type of position fixing device',
                  'Callsign', 'Name'],axis=1, errors='ignore')
    
    # Remove all the rows which does not satisfy our conditions
    df = df[
            (df["Type of mobile"] != "Class B") &
            (df["MMSI"].notna()) &
            (df["MMSI"].notnull()) &
            (df['# Timestamp'].notnull()) &
            (df['Latitude'] >=53.5) & (df['Latitude'] <=58.5) &
            (df['Longitude'] >= 3.2) & (df['Longitude'] <=16.5) &
            (df['SOG'] >= 0.1) & (df['SOG'] <=102)
           ].reset_index()

    subset_columns = ['MMSI', 'Latitude', 'Longitude', '# Timestamp']  # Adjust these based on your actual columns
    df = df.drop_duplicates(subset=subset_columns, keep='first')

    # We round the lat and longs as we do not need 15 decimals of precision
    # This will save some computation time later.
    # We also round rot, sog and cog, as we do not need a lot of decimal precision here
    df['Latitude'] = np.round(df['Latitude'], decimals=6)
    df['Longitude'] = np.round(df['Longitude'], decimals=6)
    df['ROT'] = np.round(df['ROT'], decimals=2)
    df['SOG'] = np.round(df['SOG'], decimals=2)
    df['COG'] = np.round(df['COG'], decimals=2)
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format="%d/%m/%Y %H:%M:%S").sort_values()
    df['# Timestamp'].astype('int64')//1e9
    df['# Timestamp'] = (df['# Timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()    


    # Rename the columns
    df = df.rename(columns={
            '# Timestamp':'timestamp',
            'Type of mobile':'type_of_mobile',
            'Navigational status':'navigational_status',
            'Ship type':'ship_type',
            'Type of position fixing device':'type_of_position_fixing_device',
        })
    
    # lower case names in the columns
    df.columns = map(str.lower, df.columns)
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    df = df.drop(columns=['index'], errors='ignore')
    df = df.to_crs(epsg="3857") # to calculate in meters
    
    return df

def cleanse_csv_file_for_specific_mmsi_and_covert_to_df(file_name: str, mmsi: int):
    """
    Takes a .csv file and cleanses it according to the set predicates.
    :param file_name: File name to cleanse. Example: 'aisdk-2022-01-01.csv'
    :return: A cleansed geodataframe, sorted by timestamp (ascending)
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

    df = pd.read_csv(str.format("{0}/{1}", INPUT_FOLDER, file_name), na_values=['Unknown','Undefined'], dtype=types) #, nrows=1000000)    
    
    # Remove unwanted columns containing data we do not need. This saves a little bit of memory.
    # errors='ignore' is sat because older ais data files may not contain these columns.
    df = df.drop(['A','B','C','D','ETA','Cargo type','Data source type', 'Destination', 'Length', 'Width', 'Type of position fixing device',
                  'Callsign', 'Name'],axis=1, errors='ignore')
    
    df = df[df['MMSI'] == mmsi]
    
    # Remove all the rows which does not satisfy our conditions
    df = df[
            (df["Type of mobile"] != "Class B") &
            (df["MMSI"].notna()) &
            (df["MMSI"].notnull()) &
            (df['# Timestamp'].notnull()) &
            (df['Latitude'] >=53.5) & (df['Latitude'] <=58.5) &
            (df['Longitude'] >= 3.2) & (df['Longitude'] <=16.5) &
            (df['SOG'] >= 0.1) & (df['SOG'] <=102)
           ].reset_index()

    subset_columns = ['MMSI', 'Latitude', 'Longitude', '# Timestamp']  # Adjust these based on your actual columns
    df = df.drop_duplicates(subset=subset_columns, keep='first')

    # We round the lat and longs as we do not need 15 decimals of precision
    # This will save some computation time later.
    # We also round rot, sog and cog, as we do not need a lot of decimal precision here
    df['Latitude'] = np.round(df['Latitude'], decimals=6)
    df['Longitude'] = np.round(df['Longitude'], decimals=6)
    df['ROT'] = np.round(df['ROT'], decimals=2)
    df['SOG'] = np.round(df['SOG'], decimals=2)
    df['COG'] = np.round(df['COG'], decimals=2)
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'], format="%d/%m/%Y %H:%M:%S").sort_values()
    df['# Timestamp'].astype('int64')//1e9
    df['# Timestamp'] = (df['# Timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()    

    # Rename the columns
    df = df.rename(columns={
            '# Timestamp':'timestamp',
            'Type of mobile':'type_of_mobile',
            'Navigational status':'navigational_status',
            'Ship type':'ship_type',
            'Type of position fixing device':'type_of_position_fixing_device',
        })
    
    # lower case names in the columns
    df.columns = map(str.lower, df.columns)
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
    df = df.drop(columns=['index'], errors='ignore')
    df = df.to_crs(epsg="3857") # to calculate in meters
    
    #write_to_output_folder(df)
    
    return df
    
def create_trajectories(trajectory_id: int, gdf: gpd.GeoDataFrame) -> int:
    if (gdf.empty):
        raise 'No trajectories found'
    
    for mmsi, locations in gdf.sort_values('timestamp').groupby('mmsi'):
        sub_trajectories = trajectory_in_motion_filter(trajectory_id=trajectory_id, sorted_gdf=locations)
        sub_trajectories = split_trajectories(sub_trajectories)
        trajectory_id += 1
        
        
        write_gti_trajectories(sub_trajectories)

#missing last point FIX
def trajectory_in_motion_filter(trajectory_id: int,sorted_gdf: gpd.GeoDataFrame):
    radius_threshold = 50  
    moving = True
    sub_trajectories = []
    consecutive_points_within_radius = []
    
    sorted_gdf = sorted_gdf.reset_index(drop=True)  
    
    if sorted_gdf.empty:
        return sorted_gdf
    
    for idx in range(len(sorted_gdf) - 1):
        current_location = sorted_gdf.iloc[idx]
        next_location = sorted_gdf.iloc[idx + 1]
        distance = current_location.geometry.distance(next_location.geometry)

        if (moving): # if moving
            if (distance <= radius_threshold): #check if within 50 meters distance (possible stopped)
                consecutive_points_within_radius.append(next_location)                    
                if len(consecutive_points_within_radius) >= 5:
                    moving = False
            else:
                if consecutive_points_within_radius: #if nonempty
                    sub_trajectories.extend(consecutive_points_within_radius)                        
                    consecutive_points_within_radius = [] # reset
                sub_trajectories.append(next_location) 
        else:
            if (distance > radius_threshold):
                consecutive_points_within_radius = []
                sub_trajectories.append(next_location)
                moving = True
            else:
                pass # we have already concluded that we are not moving

    if sub_trajectories:
        sub_trajectories_df = gpd.GeoDataFrame(sub_trajectories, columns=sorted_gdf.columns)
        sub_trajectories_df['trajectory_id'] = trajectory_id
        return sub_trajectories_df
    else:
        empty_df = pd.DataFrame(columns=sorted_gdf.columns)
        empty_gdf = gpd.GeoDataFrame(empty_df)
        return empty_gdf

def split_trajectories(sorted_gdf: gpd.GeoDataFrame):
    if (sorted_gdf.empty):
        return sorted_gdf
    
    sub_trajectory_id = 0
    sub_trajectories = []
    
    current_trajectory = []
    sorted_gdf.loc[0, 'sub_trajectory_id'] = sub_trajectory_id
    current_trajectory.append(sorted_gdf.iloc[0])  
    sub_trajectories.append(current_trajectory)
            
    for idx in range(1, len(sorted_gdf)):
        for sub_trajectory in sub_trajectories:
            current_location = sub_trajectory[-1]
            next_location = sorted_gdf.iloc[idx]
        
            max_speed_knots = max(current_location.sog, next_location.sog)
            
            if (next_location_is_possible_to_reach(current_location, next_location, max_speed_knots)):
                sub_trajectory.append(next_location)
                pass
            else:
                sub_trajectory_id += 1
                sorted_gdf.loc[idx, 'sub_trajectory_id'] = sub_trajectory_id
                sub_trajectories.append([next_location])    
    if (sub_trajectories):             
        sub_trajectories_df = gpd.GeoDataFrame(sub_trajectories, columns=sorted_gdf.columns)
        return sub_trajectories_df
    else:
        empty_df = pd.DataFrame(columns=sorted_gdf.columns)
        empty_gdf = gpd.GeoDataFrame(empty_df)
        return empty_gdf

            
def next_location_is_possible_to_reach(current_location, next_location, max_speed_knots):
    """
    Check if it's possible to reach the next location from the current location
    within a given threshold based on the maximum speed.
    """
    
    distance = current_location.geometry.distance(next_location.geometry) # distance is in meters
    time_diff = next_location.timestamp - current_location.timestamp
    
    if time_diff == 0:
        return False
    
    # Convert distance from meters to nautical miles
    distance_nm = distance / 1852  # 1 meter = 0.00053995680345572 nautical miles

    # Calculate speed in nautical miles per hour (knots)
    speed_knots = distance_nm / (time_diff / 3600)  # Convert time_diff from seconds to hours
    
    return speed_knots <= max_speed_knots         
    
def write_gti_trajectories(gdf: gpd.GeoDataFrame):
    # Group by 'trajectory_id' and 'trajectory_sub_id' and iterate over each group
    if (gdf.empty):
        return
    for (trajectory_id, trajectory_sub_id), trajectories in gdf.groupby(['trajectory_id', 'trajectory_sub_id']):        
        file_name = os.path.join(GTI_OUTPUT_FOLDER, f"trip_{trajectory_id}_{trajectory_sub_id}.txt")
        with open(file_name, 'w') as file:
            # Iterate over rows in the group
            for idx, row in trajectories.reset_index().iterrows():
                # Write timestamp, latitude, and longitude to the file
                file.write(f"{idx},{row['mmsi']},{row['latitude']},{row['longitude']},{row['timestamp']}\n")

def write_to_output_folder(gdf: gpd.GeoDataFrame):
    if (gdf.empty):
        return
    gdf.sort_values('timestamp')
    for mmsi, trajectories in gdf.groupby('mmsi'):        
        file_name = os.path.join(OUTPUT_FOLDER, f"{mmsi}.txt")
        with open(file_name, 'w') as file:
            # Iterate over rows in the group
            for idx, row in trajectories.reset_index().iterrows():
                # Write timestamp, latitude, and longitude to the file
                file.write(f"{idx},{row['mmsi']},{row['latitude']},{row['longitude']},{row['timestamp']}\n")

#get_csv_files_in_interval("2024-02-10::2024-02-12")
extract_trajectories_from_csv_files()