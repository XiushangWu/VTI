import time
import csv
import gc
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

from geographiclib.geodesic import Geodesic
import geopandas as gpd
import geopy.distance

from tqdm.notebook import tqdm
import pickle
from scipy import spatial
import numpy as np

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def geojson_to_fgb():
    graph_path = Path('data') / 'output_graph' / 'final_graph_cargo_0.0008_0.0016_180'
    output_path = Path('data') / 'new_graph'
    output_path.mkdir(parents=True, exist_ok=True)
    for file in graph_path.rglob('*.geojson'):
        parent = file.parents[0].name
        new_file = output_path / f'{parent}_{file.stem}.fgb'
        if new_file.exists():
            continue
        print(file)
        gdf = gpd.read_file(file)
        gdf.to_file(new_file)

geojson_to_fgb()

@time_function
def load_file(path: Path) -> gpd.GeoDataFrame:
    print(f'geopandas reading file: {path}')
    gdf = gpd.read_file(path)
    return gdf

@time_function
def load_edge_and_node_gdfs(graph_path: Path) -> tuple[list[gpd.GeoDataFrame], list[gpd.GeoDataFrame]]:
    edge_gdfs, node_gdfs = [], []
    for file in graph_path.rglob("*.fgb"):
        if file.stem.endswith("edges"):
            edge_gdfs.append(load_file(file))
        if file.stem.endswith("nodes"):
            node_gdfs.append(load_file(file))
    return edge_gdfs, node_gdfs

def load_edge_from_gpd_row(row, graph: nx.Graph):
    start_point, end_point = row["geometry"].coords
    graph.add_edge(tuple(start_point), tuple(end_point), weight=row['weight'])

def load_node_from_gpd_row(row, graph: nx.Graph):
    graph.add_node(row['geometry'].coords[0], **row)

@time_function
def create_graph_from_node_and_edge_gdfs(graph_path: Path) -> nx.Graph:
    graph = nx.Graph()
    start_time = time.perf_counter()
    for file in graph_path.rglob("*.fgb"):
        if file.stem.endswith("nodes"):
            node_gdf = load_file(file)
            node_gdf.apply(load_node_from_gpd_row, axis=1, graph=graph)
            end_time = time.perf_counter()
            print(f"file:{file} dfSize:{node_gdf.shape[0]} graphSize:{graph.number_of_nodes()} time:{end_time - start_time:.4f} seconds")
            start_time = end_time
            del node_gdf
            gc.collect()
        elif file.stem.endswith("edges"):
            edge_gdf = load_file(file)
            edge_gdf.apply(load_edge_from_gpd_row, axis=1, graph=graph)
            end_time = time.perf_counter()
            print(f"file:{file} dfSize:{edge_gdf.shape[0]} graphSize:{graph.number_of_nodes()} time:{end_time - start_time:.4f} seconds")
            start_time = end_time
            del edge_gdf
            gc.collect()
    return graph

main_graph = None

def load_main_graph() -> None:
    global main_graph
    if main_graph is not None:
        print("main_graph already loaded")
        return
    pickle_file_path = Path('data') / 'main_graph.gpickle'
    if pickle_file_path.exists():
        print("loading from binary")
        with open(pickle_file_path, 'rb') as f:
            main_graph = pickle.load(f)
        return
    print("loading from .fgb")
    main_graph = create_graph_from_node_and_edge_gdfs(Path('data') / 'new_graph')
    print("dump graph to binary")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(main_graph, f, pickle.HIGHEST_PROTOCOL)

load_main_graph()
print(main_graph.number_of_nodes(), main_graph.number_of_edges())

from math import radians, sin, cos, asin, sqrt

def get_nodes_within_range(node: tuple[float, float], all_nodes: np.array, radius: float) -> list[tuple[float, float]]:
    curr_node = np.array(node)
    distance_array = np.linalg.norm(all_nodes - curr_node, axis=1)
    return [tuple(row) for row in all_nodes[distance_array < radius]]

def haversine_distance(coord1: tuple[float, float], coord2: tuple[float, float]):
    R = 6372.8
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c

def custom_distance(coord1: tuple[float, float], coord2: tuple[float, float]):
    return geopy.distance.geodesic(coord1[::-1], coord2[::-1]).km

def heuristics(coord1, coord2):
    return custom_distance(coord1, coord2)

def generate_curve_polygon(start: dict, end: dict, buffer_dist: float, bezier_factor: float) -> Polygon:
    start_xy = start['xy'][::-1]
    end_xy = end['xy'][::-1]
    start_cog = start['cog']

    n_points = 15

    def cog_to_rad(cog):
        return np.radians((450 - cog) % 360)

    def calculate_control_points(start_xy, end_xy, start_cog, factor=bezier_factor):
        start_cog_rad = cog_to_rad(start_cog)
        start_control = (start_xy[0] + factor * np.sin(start_cog_rad), start_xy[1] + factor * np.cos(start_cog_rad))
        return [start_xy, start_control, end_xy]

    def bezier_curve(start_point, control_point_1, end_point, n_points=10):
        points = []
        for t in np.linspace(0, 1, n_points):
            x = (1 - t)**2 * start_point[0] + 2 * (1 - t) * t * control_point_1[0] + t**2 * end_point[0]
            y = (1 - t)**2 * start_point[1] + 2 * (1 - t) * t * control_point_1[1] + t**2 * end_point[1]
            points.append((x, y))
        return points

    control_points = calculate_control_points(start_xy, end_xy, start_cog)
    bezier_points = bezier_curve(control_points[0], control_points[1], control_points[2], n_points)

    line = LineString(bezier_points)
    buffered_area = line.buffer(buffer_dist)

    return buffered_area

def generate_straight_polygon(start_node: dict, end_node: dict, width: float) -> Polygon:
    line = LineString([start_node['xy'][::-1], end_node['xy'][::-1]])
    return line.buffer(width)

def read_input_trajectory(file_path: Path) -> list[dict]:
    trajectory_points = []
    with open(file_path.with_suffix('.txt'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            latitude, longitude = float(row["latitude"]), float(row["longitude"])
            trajectory_points.append({
                "xy": (longitude, latitude),
                "latitude": latitude,
                "longitude": longitude,
                "timestamp": float(row["timestamp"]),
                "sog": float(row["sog"]),
                "cog": float(row["cog"]),
                "draught": float(row["draught"]),
                "ship_type": row["ship_type"],
            })
    return trajectory_points

def find_and_connect_with_possible_neighbors(
    graph: nx.Graph,
    node: tuple[float, float],
    neighbor_radius: float,
    kd_tree: spatial.KDTree,
) -> set[tuple[float, float]]:
    neighbor_nodes_idxs = kd_tree.query_ball_point(node, neighbor_radius)
    added_nodes = set()
    for idx in neighbor_nodes_idxs:
        neighbor = tuple(kd_tree.data[idx])
        if neighbor != node:
            graph.add_node(neighbor, **graph.nodes[neighbor])
            graph.add_edge(node, neighbor)
            added_nodes.add(neighbor)
    return added_nodes

@time_function
def build_kd_tree(graph: nx.Graph) -> spatial.KDTree:
    return spatial.KDTree(graph.nodes())

def estimate_gap_dist_and_cog(start_node: dict, end_node: dict) -> tuple[float, float]:
    avg_speed = (start_node["sog"] + end_node["sog"]) / 2.0
    time_diff = end_node["timestamp"] - start_node["timestamp"]
    gap_distance_est = avg_speed * 1.852 / 3600 * time_diff
    cog_diff = abs(start_node["cog"] - end_node["cog"])
    cog_diff = min(cog_diff, 360 - cog_diff)
    return gap_distance_est, cog_diff

@time_function
def find_path(
    trajectory_nodes: list[dict],
    graph: nx.Graph,
    neighbor_radius: float,
    trajectory_dist_threshold: float,
    curve_buffer_dist: float,
    straight_buffer_dist: float,
    bezier_factor: float
) -> tuple[list[tuple[float, float]], set, set]:
    path = [(trajectory_nodes[0]['longitude'], trajectory_nodes[0]['latitude'])]
    ship_draught = trajectory_nodes[0]['draught']
    kd_tree = build_kd_tree(graph)

    to_be_removed_nodes = set()
    auxiliary_nodes = set()
    imputed_nodes = set()
    input_nodes = set()

    for node in trajectory_nodes:
        xy = node['xy']
        graph.add_node(xy, geometry=Point(*xy), sog=node['sog'], cog=node['cog'], draught=node['draught'])
        to_be_removed_nodes.add(xy)
        input_nodes.add(xy)

    for i in tqdm(range(len(trajectory_nodes) - 1)):
        start_node, end_node = trajectory_nodes[i], trajectory_nodes[i + 1]
        start_pos = start_node['xy']
        end_pos = end_node['xy']

        geodesic_dist = custom_distance(start_pos, end_pos)
        if geodesic_dist <= trajectory_dist_threshold:
            path.append(end_pos)
        else:
            gap_distance_est, cog_diff = estimate_gap_dist_and_cog(start_node, end_node)
            polygon = generate_curve_polygon(start_node, end_node, curve_buffer_dist, bezier_factor) if cog_diff > 50 else generate_straight_polygon(start_node, end_node, straight_buffer_dist)
            polygon_points = [point[::-1] for point in polygon.boundary.coords]
            for new_node in polygon_points:
                graph.add_node(new_node, geometry=Point(*new_node))
            to_be_removed_nodes.update(polygon_points)
            auxiliary_nodes.update(polygon_points)

            neighbor_nodes = find_and_connect_with_possible_neighbors(graph, start_pos, neighbor_radius, kd_tree)
            neighbor_nodes |= find_and_connect_with_possible_neighbors(graph, end_pos, neighbor_radius, kd_tree)
            auxiliary_nodes |= neighbor_nodes

            def astar_weight(a, b, attrs):
                if not polygon.contains(Point(*a[::-1])) or not polygon.contains(Point(*b[::-1])):
                    return None
                if graph.nodes[a].get("draught", ship_draught) < ship_draught or graph.nodes[b].get("draught", ship_draught) < ship_draught:
                    return None
                return custom_distance(a, b)

            try:
                new_path = nx.astar_path(graph, start_pos, end_pos, heuristic=heuristics, weight=astar_weight)
                imputed_nodes.update(new_path[1:-1])
                path += new_path[1:]
            except nx.NetworkXNoPath:
                path.append(end_pos)
    return path, input_nodes, imputed_nodes, to_be_removed_nodes, auxiliary_nodes

@time_function
def generate_output_files_for_nodes(
    nodes: list[tuple[float, float]],
    graph: nx.Graph,
    file_path: Path
) -> None:
    file_path.mkdir(parents=True, exist_ok=True)
    node_features = [{
        "type": "Feature",
        "geometry": graph.nodes[node]["geometry"],
        "properties": None
    } for node in nodes]
    gpd.GeoDataFrame.from_features(node_features).to_file(file_path.with_suffix('.fgb'))

@time_function
def generate_output_files_for_edges(
    edges: list[list[list[float, float], list[float, float]]],
    file_path: Path
) -> None:
    file_path.mkdir(parents=True, exist_ok=True)
    edge_features = [{
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": edge
        },
        "properties": None
    } for edge in edges]
    gpd.GeoDataFrame.from_features(edge_features).to_file(file_path.with_suffix('.fgb'))

def impute(
    graph: nx.Graph,
    input_file_path: Path,
    output_file_path: Path,
    trajectory_filename: str,
    neighbor_radius: float,
    trajectory_dist_threshold: float,
    curve_buffer_dist: float,
    straight_buffer_dist: float,
    bezier_factor: float
) -> None:
    trajectory_nodes = read_input_trajectory(input_file_path / trajectory_filename)
    path, input_nodes, imputed_nodes, to_be_removed_nodes, auxiliary_nodes = find_path(
        trajectory_nodes, graph, neighbor_radius, trajectory_dist_threshold, curve_buffer_dist, straight_buffer_dist, bezier_factor)

    updated_output_path = output_file_path / trajectory_filename

    generate_output_files_for_nodes(list(input_nodes), graph, updated_output_path / 'input_nodes')
    generate_output_files_for_nodes(list(imputed_nodes), graph, updated_output_path / 'imputed_nodes')
    generate_output_files_for_nodes(list(auxiliary_nodes), graph, updated_output_path / 'auxiliary_nodes')
    edges = [[path[i], list(path[i + 1])] for i in range(len(path) - 1)]
    generate_output_files_for_edges(edges, updated_output_path / 'edges')

    graph.remove_nodes_from(to_be_removed_nodes)

# Uncomment and adjust these lines to run `impute`
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_4gaps_20km'
# input_file_path = Path('data') / 'trajectory'
# output_file_path = Path('data') / 'new_output'
# output_file_path.mkdir(parents=True, exist_ok=True)
#
# neighbor_radius = 10.0064
# trajectory_dist_threshold = 1
# curve_buffer_dist = 10.01
# straight_buffer_dist = 10.03
# bezier_factor = 10.15
#
# impute(main_graph, input_file_path, output_file_path, trajectory_filename, neighbor_radius, trajectory_dist_threshold, curve_buffer_dist, straight_buffer_dist, bezier_factor)

def csv_to_geojson(dir: Path):
    for file in dir.rglob('*.txt'):
        trajectory_nodes = read_input_trajectory(file.name)
        features = [{
            "type": "Feature",
            "geometry": Point(node['longitude'], node['latitude']),
            "properties": None
        } for node in trajectory_nodes]

        node_gdf = gpd.GeoDataFrame.from_features(features)
        input_file_geojson_path = dir / file.stem
        node_gdf.to_file(input_file_geojson_path.with_suffix('.fgb'))
