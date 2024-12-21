import os
import tarfile
import zipfile
import requests
from sys import stdout
from bs4 import BeautifulSoup
from urllib.request import urlopen
from data.stats.stats_manager import stats
from data.logs.logging import setup_logger
from .extract_trajectories_from_csv import get_csv_as_df, cleanse_df, create_trajectories_files

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
AIS_CSV_FOLDER = os.path.join(DATA_FOLDER, 'ais_csv')
STATISTIC_FOLDER = os.path.join(DATA_FOLDER, 'stats')
STATISTIC_JSON_FILE = os.path.join(STATISTIC_FOLDER, 'stats.ndjson')
ETL_LOG = 'etl_log.txt'

logging = setup_logger(name=ETL_LOG, log_file=ETL_LOG)

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
    
    try:
        for file in files_to_download:
            logging.info(f'Extracting {file}')
            extract_csv_file(file_name=file)
            downloaded += 1
            stdout.write(f'\rDownloaded {file}.  Completed ({downloaded}/{len(files_to_download)})')
            stdout.flush()
        logging.info('\nCompleted extracting csv files')
    except Exception as e:
        logging.error(f'Failed to extract csv "{file}" file with error: {repr(e)}')

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
    :param filename: The file to be downloaded, cleansed and inserted
    """
    download_file_from_ais_web_server(filename)

    try:
        if ".zip" in filename: 
            filename = filename.replace('.zip', '.csv')
        else:
            filename = filename.replace('.rar', '.csv')
        
        csv_filepath = os.path.join(AIS_CSV_FOLDER, filename)
        stats.filepath = csv_filepath

        # Step 1: Read CSV
        logging.info(f'Read csv {file_name}')
        
        df = get_csv_as_df(filepath=csv_filepath) 
        
        stats.initial_rows = len(df)
        
        # Step 2: Cleanse CSV
        logging.info(f'Cleansing csv {file_name}')
        df = cleanse_df(gdf=df)
        stats.filtered_rows = len(df)
        
        # Step 3: Create trajectories
        create_trajectories_files(df)
        stats.add_to_file(STATISTIC_JSON_FILE)

        logging.info(f'Finished creating trajectories for {filename}')
        os.remove(csv_filepath)
                
    except Exception as e:
        logging.error(f'Failed to extract file {filename} with error message: {repr(e)}')
        quit()

def download_file_from_ais_web_server(filename: str):
    """
    Downloads a specified file from the webserver into the CSV_FILES_FOLDER.
    It will also unzip it, as well as delete the compressed file afterwards.
    :param filename: The file to be downloaded. Example 'aisdk-2022-01-01.zip'
    """
    download_result = requests.get('https://web.ais.dk/aisdata/' + filename, allow_redirects=True)
    download_result.raise_for_status()

    path_to_compressed_file = AIS_CSV_FOLDER + filename
    os.makedirs(AIS_CSV_FOLDER, exist_ok=True)

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
                zip_ref.extractall(path=AIS_CSV_FOLDER)
        elif ".rar" in path_to_compressed_file:
            with tarfile.RarFile(path_to_compressed_file) as rar_ref:
                rar_ref.extractall(path=AIS_CSV_FOLDER)
        else:
            logging.error(f'File {filename} must either be of type .zip or .rar. Not extracted')
            
        os.remove(path_to_compressed_file)

    except Exception as e:
        logging.exception(f'Failed with error: {e}')
        quit()

get_csv_files_in_interval("2023-03-01::2024-04-01")
