import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import json
import math
from math import radians, sin, cos, atan2, degrees, sqrt
from datetime import timedelta, datetime
from scipy.spatial import cKDTree

pd.set_option('future.no_silent_downcasting', True)
THETA_ANGLE_PENALTY = 50

"""
    Functions used in the imputation module
"""
def calculate_bearing_difference(bearing1, bearing2):
    diff = abs(bearing1 - bearing2) % 360
    return min(diff, 360 - diff)

def calculate_bearing(point1, point2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])

    # Calculate the difference in longitudes
    dlon = lon2 - lon1

    # Calculate the bearing using the atan2 function
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = atan2(y, x)

    # Convert bearing from radians to degrees
    bearing = degrees(bearing)

    # Normalize bearing to range from 0 to 360 degrees
    bearing = (bearing + 360) % 360

    return bearing

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def heuristics(coord1, coord2):

    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [coord1[0], coord1[1], coord2[0], coord2[1]])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    distance = R * c

    total_cost = distance

    return total_cost

def nodes_to_geojson(G, nodes, file_path):
    features = []    
    for node in nodes:
        node_properties = G.nodes[node]

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]]
            },
            "properties": node_properties
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

def edges_to_geojson(G, edges, file_path):
    features = []
    for start_node, end_node in edges:
        if G.has_edge(start_node, end_node):
            weight = G[start_node][end_node]['weight']
        else:
            weight = 'undefined'  # Use a placeholder if the edge doesn't exist; adjust as needed

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]
            },
            "properties": {
                "weight": weight  # Add the edge's weight to the properties
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(file_path, 'w') as f:
        json.dump(geojson, f)

def nodes_within_radius(G, point, radius):
    node_positions = np.array([(data['latitude'], data['longitude']) for node, data in G.nodes(data=True)])
    
    tree = cKDTree(node_positions)

    query_point = np.array([point[0], point[1]])  
    
    indices_within_radius = tree.query_ball_point(query_point, radius)

    nodes_within = [list(G.nodes)[i] for i in indices_within_radius]
    
    return nodes_within

def adjust_edge_weights_for_draught(G, start_point, end_point, tree, node_positions, start_draught, base_penalty=1000):
    radius = haversine_distance(start_point[0], start_point[1], end_point[0], end_point[1])

    radiusNew = radius / 100 

    start_indices_within_radius = tree.query_ball_point([start_point[0], start_point[1]], radiusNew)

    end_indices_within_radius = tree.query_ball_point([end_point[0], end_point[1]], radiusNew)


    start_nodes_within = set([list(G.nodes)[i] for i in start_indices_within_radius])
    end_nodes_within = set([list(G.nodes)[i] for i in end_indices_within_radius])
    
    relevant_nodes = start_nodes_within.union(end_nodes_within)

    valid_nodes = set()
    for node in relevant_nodes:
        node_depth = abs(G.nodes[node].get('avg_depth', G.nodes[node].get('draught')))
        if node_depth >= start_draught * 1.2:
            valid_nodes.add(node)

    valid_nodes.add(start_point)
    valid_nodes.add(end_point)

    subgraph = G.subgraph(valid_nodes).copy() 

    return subgraph

"""
    Functions used in the graph module
"""
def export_graph_to_geojson(G, nodes_file_path, edges_file_path):
    # Nodes
    nodes_features = []
    for node in G.nodes:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [node[1], node[0]] 
            },
            "properties": G.nodes[node]
        }
        nodes_features.append(feature)

    nodes_geojson = {
        "type": "FeatureCollection",
        "features": nodes_features
    }

    with open(nodes_file_path, 'w') as f:
        json.dump(nodes_geojson, f)

    # Edges
    edges_features = []
    for start_node, end_node, data in G.edges(data=True):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[start_node[1], start_node[0]], [end_node[1], end_node[0]]]
            },
            "properties": {
                "weight": data.get('weight', None)
            }
        }
        edges_features.append(feature)

    edges_geojson = {
        "type": "FeatureCollection",
        "features": edges_features
    }

    with open(edges_file_path, 'w') as f:
        json.dump(edges_geojson, f)

def calculate_initial_compass_bearing(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Summary: Calculates the compas bearing between consecutive rows in a dataframe

    Args:
        df_curr (gpd.GeoDataFrame): dataframe containing all first positions
        df_next (gpd.GeoDataFrame): dataframe containing all next positions

    Returns:
        gpd.GeoDataFrame: df_curr with a bearing
    """

    assert len(df_curr) == len(df_next), "DataFrames must have the same length"
    
    # Calculate bearing
    x = np.sin(df_curr['diff_lon']) * np.cos(df_curr['lat_rad'])     
    y = np.cos(df_next['lat_rad']) * np.sin(df_curr['lat_rad']) - (
        np.sin(df_next['lat_rad']) * np.cos(df_curr['lat_rad']) * np.cos(df_curr['diff_lon'])
    )
    
    # Bearing is now between -180 - 180
    bearing_rad = np.arctan2(x, y)     
    
    # Convert to degrees and normalize such that bearing is between 0 and 360
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360
    
    # Handle the last row 
    bearing_deg.fillna(0, inplace=True)

    return bearing_deg

def get_haversine_dist_df_in_meters(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    return: distance in meters
    """
    assert len(df_curr) == len(df_next), "DataFrames must have the same length"
    
    # Haversine formula
    a = np.sin(df_curr['diff_lat']/2)**2 + np.cos(df_curr['lat_rad']) * np.cos(df_next['lat_rad']) * np.sin(df_curr['diff_lon']/2)**2    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # Radius of earth in kilometers is 6371
    distance_km = 6371 * c
    dist = distance_km * 1000
    dist.fillna(0, inplace=True)
    
    return dist

def get_radian_and_radian_diff_columns(df_curr:gpd.GeoDataFrame, df_next:gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    # Explicitly mark df_curr and df_next as copies to avoid SettingWithCopyWarning
    df_curr = df_curr.copy()
    df_next = df_next.copy()
    
    # Convert to radians
    df_curr['lat_rad'] = np.radians(df_curr['latitude'])
    df_curr['lon_rad'] = np.radians(df_curr['longitude'])
    df_next['lat_rad'] = np.radians(df_next['latitude'])
    df_next['lon_rad'] = np.radians(df_next['longitude'])
    
    # Calculate differences
    df_curr['diff_lon'] = df_curr['lon_rad'] - df_next['lon_rad']
    df_curr['diff_lat'] = df_curr['lat_rad'] - df_next['lat_rad']
    
    
    return (df_curr.fillna(0).infer_objects(), df_next)

# Function to calculate distance and adjust based on COG
def adjusted_distance(x, y):
    # Calculate Haversine distance
    haversine_dist = geodesic((x[0], x[1]), (y[0], y[1])).m
    
    # Example COG adjustment: if COG difference is > 45 degrees, increase distance
    cog_diff = abs(x[2] - y[2])
    cog_diff = min(cog_diff, 360-cog_diff)

    # apply the angle penalty
    haversine_dist = np.sqrt(haversine_dist ** 2 + (THETA_ANGLE_PENALTY * cog_diff /180) ** 2)

    return haversine_dist