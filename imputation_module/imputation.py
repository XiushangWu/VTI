import os
import csv
import json
import networkx as nx
import numpy as np
from sqlalchemy import Tuple
from data.logs.logging import setup_logger
from utils import haversine_distance, heuristics, adjust_edge_weights_for_draught, nodes_within_radius, nodes_to_geojson, edges_to_geojson
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple
from data.logs.logging import setup_logger
from shapely.geometry import Point, box
from scipy.spatial import cKDTree
import time
import concurrent.futures
from pathlib import Path

LOG_PATH = 'imputation_log.txt'
IMPUTATION_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data//output_imputation')
#IMPUTATION_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data//txt_data_25_trajectories//20240831//data_multiple_imputed') # imputed trajectory

OUTPUT_FOLDER_RAW = os.path.join(IMPUTATION_OUTPUT, 'raw')
OUTPUT_FOLDER_PROCESSED = os.path.join(IMPUTATION_OUTPUT, 'processed')

CELLS = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), 'data//cells.txt')

GRAPH_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

logging = setup_logger(name=LOG_PATH, log_file=LOG_PATH)

if not os.path.exists(IMPUTATION_OUTPUT):
    os.makedirs(IMPUTATION_OUTPUT)

def load_geojson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def create_graph_from_geojson(nodes_geojson_path, edges_geojson_path):
    try: 
        G = nx.Graph()
        
        # Load GeoJSON files
        nodes_geojson = load_geojson(nodes_geojson_path)
        edges_geojson = load_geojson(edges_geojson_path)
        
        # Add nodes
        for feature in nodes_geojson['features']:
            node_id = tuple(feature['geometry']['coordinates'][::-1])  
            G.add_node(node_id, **feature['properties'])
        
        # Add edges
        for feature in edges_geojson['features']:
            start_node = tuple(feature['geometry']['coordinates'][0][::-1])  
            end_node = tuple(feature['geometry']['coordinates'][1][::-1])  
            G.add_edge(start_node, end_node, **feature['properties'])
        
        return G
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve graph: {repr(e)}')

# Load GeoJSON files and create a graph
def load_geojson_to_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, **feature['properties'])
        for edge in feature.get('edges', []):
            G.add_edge(coords, tuple(edge['node']), **edge['properties'])
    return G

def merge_graphs(G1, G2):
    for node, attrs in G2.nodes(data=True):
        # Check if necessary attributes exist and are not None
        if attrs.get('latitude') is not None and attrs.get('longitude') is not None:
            # Add the node to G1 only if it has valid latitude and longitude
            G1.add_node(node, **attrs)

    # Iterate over all edges in G2
    for u, v, attrs in G2.edges(data=True):
        # Add the edge to G1 only if both nodes exist in G1 (ensuring both have valid lat/lon)
        if G1.has_node(u) and G1.has_node(v):
            G1.add_edge(u, v, **attrs)
    
    return G1

def find_relevant_cells(trajectory_points, cells_df):
    relevant_cell_ids = set()
    for trajectory in trajectory_points:
        point = Point(trajectory['properties']['longitude'], trajectory['properties']['latitude'])
        # Check which cells contain the point
        for index, row in cells_df.iterrows():
            cell_polygon = box(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat'])
            if cell_polygon.contains(point):
                relevant_cell_ids.add(row['cell_id'])
    return list(relevant_cell_ids)



def revert_graph_changes(G, added_nodes, added_edges):
    for edge in added_edges:
        if G.has_edge(*edge):
            G.remove_edge(*edge)
    for node in added_nodes:
        if G.has_node(node):
            G.remove_node(node)


def generate_output_files_and_stats(G, imputed_paths, file_name, type, size, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, trajectory_points, execution_time, add_execution_time):
    unique_nodes = []
    seen_nodes = set()
    edges = []

    for path in imputed_paths:
        for node in path:
            if node not in seen_nodes:
                unique_nodes.append(node)
                seen_nodes.add(node) 
        for i in range(len(path)-1):
            edges.append((path[i], path[i+1]))

    # Output path construction and file writing
    # imputation_output_path = os.path.join(OUTPUT_FOLDER_RAW, f'fishing/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}/{file_name}')
    imputation_output_path = os.path.join(OUTPUT_FOLDER_RAW, f'cargo/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}/{file_name}')

    # refined_output_path = os.path.join(OUTPUT_FOLDER_PROCESSED, f'fishing/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}/{file_name}')
    refined_output_path = os.path.join(OUTPUT_FOLDER_PROCESSED, f'cargo/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}/{file_name}')

    if not os.path.exists(imputation_output_path):
        os.makedirs(imputation_output_path)

    # GeoJSON output
    imputed_nodes_file_path = os.path.join(imputation_output_path, f'{file_name}_nodes.geojson')
    imputed_edges_file_path = os.path.join(imputation_output_path, f'{file_name}_edges.geojson')
    nodes_to_geojson(G, list(unique_nodes), imputed_nodes_file_path)
    edges_to_geojson(G, edges, imputed_edges_file_path)


    # Statistics
    stats = {
        'file_name': file_name,
        'trajectory_points': len(trajectory_points),
        'imputed_paths': len(unique_nodes),
        'execution_time_seconds': add_execution_time + execution_time
    }

    process_imputated_trajectory(imputed_nodes_file_path, refined_output_path, f'{file_name}_nodes.geojson')

    output_directory  = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'data/stats/imputation_stats/area/{type}/{size}')
    os.makedirs(output_directory, exist_ok=True)
    stats_file = os.path.join(output_directory, f'{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}_imputation.csv')

    write_header = not os.path.exists(stats_file)
    with open(stats_file, mode='a', newline='') as csvfile:
        fieldnames = ['file_name', 'trajectory_points', 'imputed_paths', 'execution_time_seconds']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(stats)

def add_nodes_and_edges(G, trajectory_points, edge_dist_threshold):
    start_time = time.time()

    added_nodes = []
    added_edges = []

    for i in range(len(trajectory_points) - 1):
        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])

        if start_point not in G:
            G.add_node(start_point, **start_props)
            added_nodes.append(start_point)
        if end_point not in G:
            G.add_node(end_point, **end_props)
            added_nodes.append(end_point)

    # node_positions = np.array([(data['latitude'], data['longitude']) for node, data in G.nodes(data=True)])
    node_positions = []
    # 遍历图G的每个节点及其对应的属性数据
    for node, data in G.nodes(data=True):
        if 'latitude' in data and 'longitude' in data:
            latitude = data['latitude']
            longitude = data['longitude']
            node_positions.append((latitude, longitude))
        else:
            print('no latitude or longitude' + data)
    node_positions = np.array(node_positions)

    tree = cKDTree(node_positions)

    for i in range(len(trajectory_points) - 1):

        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])

        start_point_idx = tree.query_ball_point([start_point[0], start_point[1]], edge_dist_threshold)
        for idx in start_point_idx:
            node_point = tuple(node_positions[idx])
            if node_point != start_point:
                distance = haversine_distance(start_point[0], start_point[1], node_point[0], node_point[1])
                G.add_edge(start_point, node_point, weight=distance)
                G.add_edge(node_point, start_point, weight=distance)
                added_edges.append((start_point, node_point))
                added_edges.append((node_point, start_point))

        # Query for end point
        end_point_idx = tree.query_ball_point([end_point[0], end_point[1]], edge_dist_threshold)
        for idx in end_point_idx:
            node_point = tuple(node_positions[idx])
            if node_point != end_point:
                distance = haversine_distance(end_point[0], end_point[1], node_point[0], node_point[1])
                G.add_edge(end_point, node_point, weight=distance)
                G.add_edge(node_point, end_point, weight=distance)
                added_edges.append((end_point, node_point))
                added_edges.append((node_point, end_point))

    end_time = time.time()
    execution_time = end_time - start_time 
    return G, added_nodes, added_edges, execution_time, tree, node_positions

def find_and_impute_paths(G, trajectory_points, file_name, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size, added_nodes, added_edges, add_execution_time, tree, node_positions):
    start_time = time.time()
    imputed_paths = []

    for i in range(len(trajectory_points) - 1):

        start_props = trajectory_points[i]["properties"]
        end_props = trajectory_points[i + 1]["properties"]

        start_point = (start_props["latitude"], start_props["longitude"])
        end_point = (end_props["latitude"], end_props["longitude"])

        direct_path_exists = G.has_edge(start_point, end_point)

        if direct_path_exists:
            path = [start_point, end_point]
            imputed_paths.append(path)
        else:
            try:
                start_draught = start_props["draught"]
                #GG = adjust_edge_weights_for_draught(G, start_point, end_point, tree, node_positions, start_draught)
                path = nx.astar_path(G, start_point, end_point, heuristic=heuristics, weight='weight')
                imputed_paths.append(path)
            except nx.NetworkXNoPath:
                path = [start_point, end_point]










    end_time = time.time()
    execution_time = end_time - start_time 
    print(f"Imputation took: {add_execution_time + execution_time} \n")

    generate_output_files_and_stats(G, imputed_paths, file_name, type, size, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, trajectory_points, execution_time, add_execution_time)
    
    revert_graph_changes(G, added_nodes, added_edges)

    return imputed_paths

def load_graphs_and_impute_trajectory(file_name, file_path, G, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size):
    trajectory_points = []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                trajectory_point = {
                    "properties": {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "timestamp": float(row["timestamp"]),
                        "sog": float(row["sog"]),
                        "cog": float(row["cog"]),
                        "draught": float(row["draught"]),
                        "ship_type": row["ship_type"],
                    }
                }
                trajectory_points.append(trajectory_point)
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory to impute: {repr(e)}')

    new_g, added_nodes, added_edges, add_execution_time, tree, node_positions = add_nodes_and_edges(G, trajectory_points, edge_dist_threshold)
    find_and_impute_paths(new_g, trajectory_points, file_name, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size, added_nodes, added_edges, add_execution_time, tree, node_positions)

def load_intersecting_graphs_and_impute_trajectory(file_name, file_path, graphs, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size):
    G = nx.Graph()
    cells_df = pd.read_csv(CELLS) 
    start_time = time.time()

    trajectory_points = []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                trajectory_point = {
                    "properties": {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "timestamp": float(row["timestamp"]),
                        "sog": float(row["sog"]),
                        "cog": float(row["cog"]),
                        "draught": float(row["draught"]),
                        "ship_type": row["ship_type"],
                    }
                }
                trajectory_points.append(trajectory_point)
    except Exception as e:
        logging.warning(f'Error occurred trying to retrieve trajectory to impute: {repr(e)}')

    relevant_cell_ids = find_relevant_cells(trajectory_points, cells_df)
    
    for cell_id in relevant_cell_ids:
        node_path = os.path.join(GRAPH_OUTPUT, f"{graphs}//{cell_id}//nodes.geojson")
        edge_path = os.path.join(GRAPH_OUTPUT, f"{graphs}//{cell_id}//edges.geojson")
        G_cell = create_graph_from_geojson(node_path, edge_path)
        G = merge_graphs(G, G_cell)
    
    end_time = time.time()
    execution_time = end_time - start_time 
    print("Reading graph took:", execution_time)

    new_g, added_nodes, added_edges = add_nodes_and_edges(G, trajectory_points, edge_dist_threshold)

    find_and_impute_paths(new_g, trajectory_points, file_name, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)
    print("Imputation done")

def calculate_center_position(p1, p2):
    """Calculate the center position of the trajectory segment."""
    center_pos = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    return center_pos

def best_fit(segment: List[Tuple[float, float]]):
    """Perform linear least squares regression on a segment of positions."""
    # extract X (longitude) and Y (latitude) coordinates
    x_coords = np.array([position[0] for position in segment])
    y_coords = np.array([position[1] for position in segment])
    
    # stack X coordinates with a column of ones for the intercept term
    matrix = np.vstack([x_coords, np.ones(len(x_coords))]).T
    
    # calculate the least squares solution
    regression_result = np.linalg.lstsq(matrix, y_coords, rcond=None)
    
    # unpack the slope and intercept
    slope, intercept = regression_result[0]
    
    # calculate the residual sum of squares, handle cases with no residuals
    residuals = regression_result[1]
    if residuals.size == 0:
        residuals_value = 0
    else:
        residuals_value = residuals[0]

    # compute the predicted Y-values based on the best-fit line
    fitted_segment = []
    fitted_segment += [(x, slope * x + intercept) for x in x_coords]

    return fitted_segment, residuals_value

def refine_trajectory(trajectory: List[Tuple[float,float]], epsilon=1e-6):
    if len(trajectory) < 3:
        refined_geometries = [Point(x, y) for x, y in trajectory]
        return gpd.GeoDataFrame(geometry=refined_geometries)
    
    anchor: int = 0
    window_size: int = 3
    final_trajectory = []
    previous_fit = trajectory[:2]
    turn_detected = False
    i = 0
    while (anchor + window_size) <= len(trajectory):
        # extract the current segment
        current_segment:list = trajectory[anchor:anchor + window_size]

        # compute best fit for the current segment
        best_fit_segment, residual = best_fit(current_segment)

        if residual > epsilon and ((anchor + window_size + 1) < len(trajectory)):
            extended_segment = trajectory[anchor:anchor + window_size - 1]
            extended_segment.append(trajectory[anchor + window_size + 1])

            _, residual = best_fit(extended_segment)

            if residual > epsilon:
                turn_detected = True 
            else:
                new_point = calculate_center_position(extended_segment[-2], extended_segment[-1])
                current_segment[-1] = new_point

        if (turn_detected):
            turn_detected = False
            best_fit_segment, _ = best_fit(previous_fit)
            final_trajectory.extend(best_fit_segment)
            anchor += (window_size - 1)
            window_size = 3
            previous_fit = trajectory[anchor:anchor + 2]
        else:
            previous_fit = current_segment
            window_size += 1

    # Add the last refined sub-trajectory
    final_trajectory.extend(previous_fit)
    if len(final_trajectory) != len(trajectory):
        print("Lengths are not equal in 4")
    refined_geometries = [Point(x, y) for x, y in final_trajectory]

    return gpd.GeoDataFrame(geometry=refined_geometries)

def process_imputated_trajectory(filepath_nodes:str, output_folder:str, filename:str):
    nodes_gdf = gpd.read_file(filepath_nodes)
    coordinates = nodes_gdf[['latitude', 'longitude']].apply(lambda x: (x['longitude'], x['latitude']), axis=1).tolist()

    nodes_refined_gdf4 = refine_trajectory(coordinates)

    os.makedirs(output_folder, exist_ok=True)

    new_filepath = os.path.join(output_folder, filename)
    nodes_refined_gdf4.to_file(new_filepath, driver='GeoJSON')