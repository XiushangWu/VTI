import json
import os
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import numpy as np
from tslearn.metrics import dtw
import csv
from imputation_module.imputation import process_imputated_trajectory
from utils import haversine_distance

def load_geojson_extract_coordinates(file_path):
    with open(file_path, 'r') as f:
        geojson = json.load(f)
    
    coordinates = []
    for feature in geojson['features']:
        lon, lat = feature['geometry']['coordinates']
        coordinates.append([lat, lon])  

    return np.array(coordinates)
    
def load_csv_extract_coordinates(file_path):
    data = pd.read_csv(file_path)
    coordinates = data[['latitude', 'longitude']]
    
    return coordinates.values

def load_csv_extract_coordinates_gti(file_path):
    data = pd.read_csv(file_path, header=None)
    coordinates = data.iloc[:, [2, 1]]
    
    return coordinates.values
    

def normalize_distance(distance, len_traj1, len_traj2):
    total_points = len_traj1 + len_traj2
    return distance / total_points if total_points else 0  # Avoid division by zero

def vectorized_haversine_distance(orig_traj, imp_traj):
    # Expand dimensions and compute all pairwise haversine distances in a vectorized manner
    orig_lat, orig_lon = np.radians(orig_traj[:, 0]), np.radians(orig_traj[:, 1])
    imp_lat, imp_lon = np.radians(imp_traj[:, 0]), np.radians(imp_traj[:, 1])

    dlat = orig_lat[:, np.newaxis] - imp_lat
    dlon = orig_lon[:, np.newaxis] - imp_lon

    a = np.sin(dlat / 2.0)**2 + np.cos(orig_lat[:, np.newaxis]) * np.cos(imp_lat) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371.0 * c  # Earth radius in kilometers
    return distance

def dynamic_time_warping(original_trajectory, imputed_trajectory):

    original_trajectory = np.array(original_trajectory)
    imputed_trajectory = np.array(imputed_trajectory)

    # Generate distance matrix using vectorized Haversine calculation
    dist_matrix = vectorized_haversine_distance(original_trajectory, imputed_trajectory)
    
    P_len, Q_len = dist_matrix.shape
    dtw_matrix = np.full((P_len, Q_len), float('inf'))
    dtw_matrix[0, 0] = dist_matrix[0, 0]

    # Populate the DTW matrix
    for i in range(1, P_len):
        for j in range(1, Q_len):
            dtw_matrix[i, j] = dist_matrix[i, j] + min(dtw_matrix[i-1, j],    # Insertion
                                                       dtw_matrix[i, j-1],    # Deletion
                                                       dtw_matrix[i-1, j-1])  # Match

    normalized_dtw = normalize_distance(dtw_matrix[-1, -1], P_len, Q_len)
    return normalized_dtw


def frechet_distance(original_trajectory, imputed_trajectory):
    dist_matrix = vectorized_haversine_distance(np.array(original_trajectory), np.array(imputed_trajectory))
    
    P_len, Q_len = dist_matrix.shape
    dp = np.full((P_len, Q_len), np.inf)
    dp[0, 0] = dist_matrix[0, 0]

    for i in range(1, P_len):
        for j in range(1, Q_len):
            dp[i, j] = max(min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]), dist_matrix[i, j])

    normalized_frechet = normalize_distance(dp[-1, -1], P_len, Q_len)
    return normalized_frechet



def compare_imputed(imputed_trajectory_path, original_trajectory_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, path, type):
    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data//stats//evaluation//all//{type}//{path}')
    os.makedirs(output_directory, exist_ok=True)
    stats_file = os.path.join(output_directory, f'imputedR_evaluation.csv')
    
    original_files = {}
    for root, dirs, files in os.walk(original_trajectory_path):
        for file in files:
            original_files[file.replace('.txt', '')] = os.path.join(root, file)

    file_counter = 0

    with open(stats_file, mode='w', newline='') as csvfile:
        fieldnames = ['Trajectory', 'Original Length', 'Imputed Length', 'DTW', 'Frechet Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for root, dirs, files in os.walk(imputed_trajectory_path):
            for file in files:
                if file.endswith('nodes.geojson'):
                    base_name = file.replace('.txt_nodes.geojson', '')
                    if base_name in original_files:  
                        file_path = os.path.join(root, file)
                        ofile_path = original_files[base_name]
                        print(f"Found DGIVT file: matching file {file_counter}")
                        
                        imputed_trajectory = process_imputated_trajectory(file_path)
                        original_trajectory = load_csv_extract_coordinates(ofile_path)
                        
                        original_len = len(original_trajectory)
                        imputed_len = len(imputed_trajectory)
                        dtw = dynamic_time_warping(original_trajectory, imputed_trajectory)
                        fd = frechet_distance(original_trajectory, imputed_trajectory)
                        
                        writer.writerow({
                            'Trajectory': base_name,
                            'Original Length': original_len,
                            'Imputed Length': imputed_len,
                            'DTW': dtw,
                            'Frechet Distance': fd
                        })

                        file_counter += 1

def compare_linear(original_trajectory_path, path, type, sparsed_trajectories):
    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data//stats//evaluation//all//{type}//{path}')
    os.makedirs(output_directory, exist_ok=True)
    stats_file = os.path.join(output_directory, f'linear_{type}_{path}_evaluation.csv')

    original_files = {}
    for root, dirs, files in os.walk(original_trajectory_path):
        for file in files:
            if file.endswith('.txt'):  # Adjust based on your file extension
                original_files[file] = os.path.join(root, file)

    file_counter = 0

    with open(stats_file, mode='w', newline='') as csvfile:
        fieldnames = ['Trajectory', 'Original Length', 'Linear Length', 'DTW', 'Frechet Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each sparsed trajectory file
        for root, dirs, files in os.walk(sparsed_trajectories):
            for file in files:
                if file in original_files:
                    file_path = os.path.join(root, file)
                    ofile_path = original_files[file]
                    print(f"Found linear file: matching file {file_counter}")
                    
                    # Load trajectories
                    sparsed_trajectory = load_csv_extract_coordinates(file_path)
                    original_trajectory = load_csv_extract_coordinates(ofile_path)
                    
                    # Compute metrics
                    original_len = len(original_trajectory)
                    linear_len = len(sparsed_trajectory)
                    dtw = dynamic_time_warping(original_trajectory, sparsed_trajectory)
                    fd = frechet_distance(original_trajectory, sparsed_trajectory)
                    
                    # Write results to CSV
                    writer.writerow({
                        'Trajectory': file,
                        'Original Length': original_len,
                        'Linear Length': linear_len,
                        'DTW': dtw,
                        'Frechet Distance': fd
                    })

                    file_counter += 1


def compare_gti(imputed_trajectory_path, original_trajectory_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, path, type, imputed_trajectories_gti):
    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data//stats//evaluation//all//{type}//{path}')
    os.makedirs(output_directory, exist_ok=True)
    stats_file = os.path.join(output_directory, f'gti_{type}_{path}_evaluation.csv')

    original_files = {}
    for root, dirs, files in os.walk(original_trajectory_path):
        for file in files:
            if file.endswith('.txt'):  # Adjust based on your file extension
                original_files[file] = os.path.join(root, file)
    
    file_counter = 0

    with open(stats_file, mode='w', newline='') as csvfile:
        fieldnames = ['Trajectory', 'Original Length', 'GTI Length', 'DTW', 'Frechet Distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each sparsed trajectory file
        for root, dirs, files in os.walk(imputed_trajectories_gti):
            for file in files:
                if file in original_files:
                    file_path = os.path.join(root, file)
                    ofile_path = original_files[file]
                    print(f"Found GTI file: matching file {file_counter}")
                    
                    # Load trajectories
                    gti_trajectory = load_csv_extract_coordinates_gti(file_path)
                    original_trajectory = load_csv_extract_coordinates(ofile_path)
                    
                    original_len = len(original_trajectory)
                    linear_len = len(gti_trajectory)
                    dtw = dynamic_time_warping(original_trajectory, gti_trajectory)
                    fd = frechet_distance(original_trajectory, gti_trajectory)
                    
                    writer.writerow({
                        'Trajectory': file,
                        'Original Length': original_len,
                        'GTI Length': linear_len,
                        'DTW': dtw,
                        'Frechet Distance': fd
                    })

                    file_counter += 1