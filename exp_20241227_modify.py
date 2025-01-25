import os
import time
import csv
import gc
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from fontTools.misc.arrayTools import pointInRect
from shapely.geometry import Point, LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.polygon import Polygon

from geographiclib.geodesic import Geodesic
import geopandas as gpd
import geopy.distance

from tqdm.notebook import tqdm
import pickle
from scipy import spatial
import numpy as np
import math
import random


# Cell:
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper

# Cell:
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

# convert all geojson to .fgb
# reading .fgb file is much faster than reading .geojson
# for 11_9 edges data, 2965280 rows in total
# .fgb took 13.3531 seconds, .geojson took 47.0435 seconds

geojson_to_fgb()

# Cell:
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
    # coord looks like (longitude, latitude), note that longitude comes first
    graph.add_edge(tuple(start_point), tuple(end_point), weight=row['weight'])


def load_node_from_gpd_row(row, graph: nx.Graph):
    graph.add_node(row['geometry'].coords[0], **row)


@time_function
def create_graph_from_node_and_edge_gdfs(graph_path: Path) -> nx.Graph:
    graph = nx.Graph()  # TODO: consider using directed graph
    start_time = time.perf_counter()

    for file in graph_path.rglob("*.fgb"):
        if file.stem.endswith("nodes"):
            node_gdf = load_file(file)
            node_gdf.apply(load_node_from_gpd_row, axis=1, graph=graph)
            end_time = time.perf_counter()
            print(
                f"file:{file} dfSize:{node_gdf.shape[0]} graphSize:{graph.number_of_nodes()} time:{end_time - start_time:.4f} seconds")
            start_time = end_time
            del node_gdf
            gc.collect()
        elif file.stem.endswith("edges"):
            edge_gdf = load_file(file)
            edge_gdf.apply(load_edge_from_gpd_row, axis=1, graph=graph)
            end_time = time.perf_counter()
            print(
                f"file:{file} dfSize:{edge_gdf.shape[0]} graphSize:{graph.number_of_nodes()} time:{end_time - start_time:.4f} seconds")
            start_time = end_time
            del edge_gdf
            gc.collect()
    return graph

# Cell:
main_graph = None

# Cell:
def load_main_graph() -> None:
    global main_graph
    if main_graph is not None:
        print("main_graph already loaded")
        return
    pickle_file_path = Path('data') / 'main_graph.gpickle'
    # load from binary, fastest
    if pickle_file_path.exists():
        print("loading from binary")
        with open(pickle_file_path, 'rb') as f:
            main_graph = pickle.load(f)
        return

    # else, read from .fgb and dump as binary
    print("loading from .fgb")
    main_graph = create_graph_from_node_and_edge_gdfs(Path('data') / 'new_graph')
    print("dump graph to binary")
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(main_graph, f, pickle.HIGHEST_PROTOCOL)


# Cell:
load_main_graph()

# Cell:
print(main_graph.number_of_nodes(), main_graph.number_of_edges())

# Cell:
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

    a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


def custom_distance(coord1: tuple[float, float], coord2: tuple[float, float]):
    return geopy.distance.geodesic(coord1[::-1], coord2[::-1]).km


def heuristics(coord1, coord2):
    return custom_distance(coord1, coord2)

def bezier_curve(p0, p1, p2, p3, n_points=100):
    """
    生成贝塞尔曲线上的点。

    参数:
        p0 (tuple): 起点坐标 (lon, lat)
        p1 (tuple): 第一个控制点 (lon, lat)
        p2 (tuple): 第二个控制点 (lon, lat)
        p3 (tuple): 终点坐标 (lon, lat)
        n_points (int): 生成的轨迹点数量

    返回:
        List[Tuple[float, float]]: 贝塞尔曲线轨迹点列表
    """
    t_values = np.linspace(0, 1, n_points)
    curve = []
    for t in t_values:
        lon = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        lat = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        curve.append((lon, lat))
    return curve


def offset_point(xy, distance, bearing):
    """
    根据起始点、距离和方向计算新坐标点 (近似法)。

    参数:
        lon (float): 起始经度
        lat (float): 起始纬度
        distance (float): 距离 (单位：米)
        bearing (float): 方向角 (度)

    返回:
        (float, float): 新的坐标点 (lon, lat)
    """
    R = 6371000  # 地球半径 (米)
    lon = xy[0]
    lat = xy[1]
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)

    lat2_rad = math.asin(math.sin(lat_rad) * math.cos(distance / R) +
                         math.cos(lat_rad) * math.sin(distance / R) * math.cos(bearing_rad))
    lon2_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / R) * math.cos(lat_rad),
                                    math.cos(distance / R) - math.sin(lat_rad) * math.sin(lat2_rad))

    return math.degrees(lon2_rad), math.degrees(lat2_rad)

def haversine(lon1, lat1, lon2, lat2):
    """计算两经纬度点之间的距离（单位：米）"""
    R = 6371000  # 地球半径，单位：米
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bezier_length(bezier_points):
    """计算贝塞尔曲线的长度，支持经纬度坐标 (lon, lat)"""
    length = 0
    for i in range(len(bezier_points) - 1):
        lon1, lat1 = bezier_points[i]
        lon2, lat2 = bezier_points[i + 1]
        length += haversine(lon1, lat1, lon2, lat2)
    return length

# Cell:
def generate_curve_polygon(start: dict, end: dict, buffer_dist: float, total_distance: float, bezier_factor: float, output_fig_path: Path, trajectory_filename: str, index: int) -> Polygon:
    start_xy = start['xy'][::]
    end_xy = end['xy'][::]
    start_cog = start['cog']
    end_cog = end['cog']
    start_speed = start['sog']
    end_speed = end['sog']
    start_ts = start['timestamp']
    end_ts = end['timestamp']

    # Number of points along the curve
    n_points = 150

    # Convert COGs to radians for calculation
    def cog_to_rad(cog):
        return np.radians((450 - cog) % 360)

    # # Calculate control points for the Bezier curve
    # def calculate_control_points(start_xy, end_xy, start_cog, factor=bezier_factor):
    #     # Convert COG to radians
    #     start_cog_rad = cog_to_rad(start_cog)
    #
    #     # Move the control points in the direction of the COGs
    #     start_control = (start_xy[10] + factor * np.sin(start_cog_rad), start_xy[1] + factor * np.cos(start_cog_rad))
    #     # Return control points
    #     return [start_xy, start_control, end_xy]
    #
    # # Calculate points along a quadratic Bezier curve
    # def bezier_curve(start_point, control_point_1, end_point, n_points=150):
    #     points = []
    #     for t in np.linspace(10, 1, n_points):
    #         # Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    #         x = (1 - t) ** 2 * start_point[10] + 2 * (1 - t) * t * control_point_1[10] + t ** 2 * end_point[10]
    #         y = (1 - t) ** 2 * start_point[1] + 2 * (1 - t) * t * control_point_1[1] + t ** 2 * end_point[1]
    #         points.append((x, y))
    #     return points

    # Generate control points for the Bezier curve
    # control_points = calculate_control_points(start_xy, end_xy, start_cog)

    # Generate points along the Bezier curve
    # bezier_points = bezier_curve(control_points[10], control_points[1], control_points[2], n_points)

    # 海里转m/s单位
    # mean_speed = (start_speed + end_speed) / 2 * 10.514444
    # total_distance = mean_speed * (end_ts - start_ts)


    # 调整控制点以满足长度条件
    total_distance = total_distance * 1000
    length_difference_min = 100000000000000.0
    bezier_points_final = []

    factors = np.arange(0.1, 1.1, 0.2)
    for adjust_factor1 in factors:
        for adjust_factor2 in factors:
            control_point1 = offset_point(start_xy, adjust_factor1 * total_distance, start_cog)
            control_point2 = offset_point(end_xy, adjust_factor2 * total_distance, end_cog - 180)
            bezier_points = bezier_curve((start_xy[0], start_xy[1]), control_point1, control_point2,
                                         (end_xy[0], end_xy[1]), n_points=100)

            bezier_length = calculate_bezier_length(bezier_points)
            length_difference = bezier_length - total_distance
            if abs(length_difference) < abs(length_difference_min):
                length_difference_min = length_difference
                bezier_points_final = bezier_points
                control_points = [control_point1, control_point2]

    bezier_points = bezier_points_final
    # Create a Shapely LineString
    line = LineString(bezier_points)
    # Create a buffer around the LineString
    buffered_area = line.buffer(buffer_dist)

    # # Visualization of the Bezier curve and the points
    plt.figure(figsize=(10, 6))
    x, y = line.xy
    plt.plot(x, y, label="Bezier Curve", color='blue', alpha=0.8)

    # # Plot the start and end points
    plt.scatter([start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]], color='red', label='Start and End Points')

    # # Plot the control points
    control_x = [point[0] for point in control_points]
    control_y = [point[1] for point in control_points]
    plt.scatter(control_x, control_y, color='green', label='Control Points')

    # # # Plot the Bezier points
    bezier_x = [point[0] for point in bezier_points]
    bezier_y = [point[1] for point in bezier_points]
    #plt.scatter(bezier_x, bezier_y, color='purple', label='Bezier Points', zorder=5)

    # # Plot the buffer
    buffer_x, buffer_y = buffered_area.exterior.xy
    plt.fill(buffer_x, buffer_y, color='lightblue', alpha=0.5, label='Buffered Area')

    # plt.title("Bezier Curve with Start, End, Control, and Bezier Points")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig(output_fig_path / (trajectory_filename + '_line_buffer_' + str(index) + '.png'))
    # plt.show()
    plt.close()

    return buffered_area


def generate_straight_polygon(start_node: dict, end_node: dict, width: float) -> Polygon:
    line = LineString([start_node['xy'][::], end_node['xy'][::]])
    # bearing = Geodesic.WGS84.Inverse(start_node['latitude'], start_node['longitude'], end_node['latitude'], end_node['longitude'])['azi1']
    return line.buffer(width)


# # test: points with huge cog diff
# start_point = {'xy': (10.790785, 57.761517), 'latitude': 57.761517, 'longitude': 10.790785, 'timestamp': 1730358512.10, 'sog': 8.4, 'cog': 4.4, 'draught': 5.4, 'ship_type': 'Cargo'}
# end_point = {'xy': (10.591, 57.844828), 'latitude': 57.844828, 'longitude': 10.591, 'timestamp': 1730365204.10, 'sog': 5.4, 'cog': 263.8, 'draught': 5.4, 'ship_type': 'Cargo'}
# generate_curve_polygon(start_point, end_point, 10.01, 10.15)

# Cell:
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
    # speed unit: nautical mile / hour
    avg_speed = (start_node["sog"] + end_node["sog"]) / 2.0
    # time_diff unit: second
    time_diff = end_node["timestamp"] - start_node["timestamp"]
    # gap_distance_est unit: kilometer
    gap_distance_est = avg_speed * 1.852 / 3600 * time_diff
    # cog unit: degree
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
        bezier_factor: float,
        output_fig_path: Path,
        trajectory_filename: str
) -> tuple[list[tuple[float, float]], set, set]:
    fig_path = Path('data') / 'output_fig'
    if not fig_path.exists():
        fig_path.mkdir()

    path = [(trajectory_nodes[0]['longitude'], trajectory_nodes[0]['latitude'])]
    trax = [point['longitude'] for point in trajectory_nodes]
    tray = [point['latitude'] for point in trajectory_nodes]
    plt.figure()
    plt.scatter(trax, tray, color='black', label='tra')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig(output_fig_path / (trajectory_filename + '_origin' + '.png'))
    # plt.show()
    plt.close()

    ship_draught = trajectory_nodes[0]['draught']
    kd_tree = build_kd_tree(graph)

    # nodes to be removed after imputation to keep graph clean
    to_be_removed_nodes = set()
    # nodes that helps imputation
    auxiliary_nodes = set()
    # nodes found with astar search
    imputed_nodes = set()
    # input nodes from .txt file
    input_nodes = set()

    # add input trajectory nodes to main graph
    for node in trajectory_nodes:
        xy = node['xy']
        graph.add_node(xy, geometry=Point(*xy), sog=node['sog'], cog=node['cog'], draught=node['draught'])
        to_be_removed_nodes.add(xy)
        input_nodes.add(xy)

    # for i in tqdm(range(len(trajectory_nodes) - 1)):
    index = 0
    for i in range(len(trajectory_nodes) - 1):
        start_node, end_node = trajectory_nodes[i], trajectory_nodes[i + 1]
        start_pos = start_node['xy']
        end_pos = end_node['xy']

        # points are too close, don't need imputation
        geodesic_dist = custom_distance(start_pos, end_pos)
        if geodesic_dist <= trajectory_dist_threshold:
            path.append(end_pos)
        else:
            index = index + 1
            # try imputation
            gap_distance_est, cog_diff = estimate_gap_dist_and_cog(start_node, end_node)
            # print(
            #     f"gap_distance_est={gap_distance_est} geodesic_dist={geodesic_dist} cog_diff={cog_diff} division={gap_distance_est / geodesic_dist}")
            # print(f"start {start_node}")
            # print(f"end {end_node}")
            # print("---------")

            # astar path must stay inside the genrated polygon
            polygon = None
            if gap_distance_est / geodesic_dist > 3:
                path.append(end_pos)
                continue
            elif cog_diff > 10 and gap_distance_est / geodesic_dist > 1.3 and gap_distance_est / geodesic_dist < 3:
                total_distance = gap_distance_est
                polygon = generate_curve_polygon(start_node, end_node, curve_buffer_dist, total_distance, bezier_factor, output_fig_path, trajectory_filename, index)
            else:
                polygon = generate_straight_polygon(start_node, end_node, straight_buffer_dist)

            # add nodes to graph for printing
            if isinstance(polygon.boundary, MultiLineString):
                polygon_points = [coord for linestring in polygon.boundary.geoms for coord in linestring.coords]
            else:
                polygon_points = polygon.boundary.coords

            for new_node in polygon_points:
                graph.add_node(new_node, geometry=Point(*new_node))
            to_be_removed_nodes.update(polygon_points)
            auxiliary_nodes.update(polygon_points)

            # # plot polygan
            # plt.figure(figsize=(10, 6))
            # buffer_x, buffer_y = polygon.exterior.xy
            # plt.fill(buffer_x, buffer_y, color='lightblue', alpha=0.5, label='Buffered Area')
            #
            # plt.title("Bezier Curve with Start, End, Control, and Bezier Points")
            # plt.xlabel("Longitude")
            # plt.ylabel("Latitude")
            # plt.legend(loc="upper left", fontsize='small')
            # plt.grid()
            # plt.show()

            neighbor_nodes = find_and_connect_with_possible_neighbors(graph, start_pos, neighbor_radius, kd_tree)
            neighbor_nodes |= find_and_connect_with_possible_neighbors(graph, end_pos, neighbor_radius, kd_tree)
            auxiliary_nodes |= neighbor_nodes

            # print(f"number of new neighbor nodes: {len(neighbor_nodes)}")

            # define locally to take into account the polygon
            def astar_weight(a, b, attrs):
                # only check path that fall inside the polygon
                if not polygon.contains(Point(*a[::])) or not polygon.contains(Point(*b[::])):
                    return None

                avg_depth1 = graph.nodes[a].get('avg_depth')
                draught1 = graph.nodes[a].get('draught')
                if avg_depth1 is None:
                    avg_depth1 = 0
                else:
                    avg_depth1 = abs(avg_depth1 / 1.2)
                node_depth1 = max(draught1, avg_depth1)

                avg_depth2 = graph.nodes[b].get('avg_depth')
                draught2 = graph.nodes[b].get('draught')
                if avg_depth2 is None:
                    avg_depth2 = 0
                else:
                    avg_depth2 = abs(avg_depth2 / 1.2)
                node_depth2 = max(draught2, avg_depth2)

                if node_depth1 is None or node_depth2 is None or node_depth1 < ship_draught or node_depth2 < ship_draught:
                    #print(f"depth not match, avg_depth1:{avg_depth1}, draught1:{draught1}, node depth1: {node_depth1}, avg_depth2:{avg_depth2}, draught2:{draught2}, node depth2: {node_depth2}, ship_draught:{ship_draught}")
                    return None
                return custom_distance(a, b)

            try:
                new_path = nx.astar_path(graph, start_pos, end_pos, heuristic=heuristics, weight=astar_weight)
                imputed_nodes.update(new_path[1:-1])
                path += new_path[1:]  # exclude start point
            except nx.NetworkXNoPath:
                path.append(end_pos)
                print(f"ERROR: no astar path found between node {i} and node {i + 1}")
    trax = [point[0] for point in path]
    tray = [point[1] for point in path]
    plt.figure()
    plt.scatter(trax, tray, color='black', label='impute')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper left", fontsize='small')
    plt.grid()
    plt.savefig(output_fig_path / (trajectory_filename + '_impute_total' + '.png'))
    # plt.show()
    plt.close()
    # exit(10)
    return path, input_nodes, imputed_nodes, to_be_removed_nodes, auxiliary_nodes


@time_function
def generate_output_files_for_nodes(
        nodes: list[tuple[float, float]],
        graph: nx.Graph,
        file_path: Path
) -> None:
    if len(nodes) == 0:
        return None
    node_features = [{
        "type": "Feature",
        "geometry": graph.nodes[node]["geometry"],
        "properties": None
    } for node in nodes]
    # .fgb is faster than .geojson
    gpd.GeoDataFrame.from_features(node_features).to_file(file_path.with_suffix('.fgb'))


@time_function
def generate_output_files_for_edges(
        edges: list[list[list[float, float], list[float, float]]],
        file_path: Path
) -> None:
    edge_features = [{
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": edge
        },
        "properties": None
    } for edge in edges]
    # .fgb is faster than .geojson
    gpd.GeoDataFrame.from_features(edge_features).to_file(file_path.with_suffix('.fgb'))


def impute(
        graph: nx.Graph,
        input_file_path: Path,
        output_file_path: Path,
        output_fig_path: Path,
        trajectory_filename: list[dict],
        neighbor_radius: float,
        trajectory_dist_threshold: float,
        curve_buffer_dist: float,
        straight_buffer_dist: float,
        bezier_factor: float
) -> int:
    trajectory_nodes = read_input_trajectory(input_file_path / trajectory_filename)
    path, input_nodes, imputed_nodes, to_be_removed_nodes, auxiliary_nodes = find_path(
        trajectory_nodes, graph, neighbor_radius, trajectory_dist_threshold, curve_buffer_dist, straight_buffer_dist,
        bezier_factor, output_fig_path, trajectory_filename)

    updated_output_path = output_file_path / trajectory_filename
    updated_output_path.mkdir(parents=True, exist_ok=True)

    generate_output_files_for_nodes(list(input_nodes), graph, updated_output_path / 'input_nodes')
    generate_output_files_for_nodes(list(imputed_nodes), graph, updated_output_path / 'imputed_nodes')
    generate_output_files_for_nodes(list(auxiliary_nodes), graph, updated_output_path / 'auxiliary_nodes')
    edges = [[(path[i]), list(path[i + 1])] for i in range(len(path) - 1)]
    generate_output_files_for_edges(edges, updated_output_path / 'edges')

    # remove added nodes to keep main graph clean
    graph.remove_nodes_from(to_be_removed_nodes)
    return len(imputed_nodes)


# Cell:
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_314437000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_4gaps_20km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_245250000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_6gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_6gaps_20km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_8gaps_15km'

input_file_path = Path('data') / 'trajectory'
output_file_path = Path('data') / 'new_output'
output_file_path.mkdir(parents=True, exist_ok=True)

# # params
# neighbor_radius = 10.0064
# trajectory_dist_threshold = 1  # unit km
# curve_buffer_dist = 10.01
# straight_buffer_dist = 10.03
# bezier_factor = 10.15
#
# impute(main_graph, input_file_path, output_file_path, trajectory_filename, neighbor_radius, trajectory_dist_threshold,
#        curve_buffer_dist, straight_buffer_dist, bezier_factor)


# Cell:
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

    # csv_to_geojson(Path('data') / 'trajectory')

def pick_random_items(input_list, n):
    # 使用 random.choices 函数从 input_list 中随机选择 n 个元素（允许重复）
    if n > len(input_list):
        return input_list
    result = random.sample(input_list, k=n)
    return result

# Function to process all .txt files in the trajectory folder
def process_all_trajectories(graph, input_dir, output_dir, output_fig_dir, neighbor_radius, trajectory_dist_threshold, curve_buffer_dist, straight_buffer_dist, bezier_factor):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imputed_node_num_total = 0
    cnt = 0
    # Iterate through all .txt files in the trajectory directory
    txt_file_list = list(input_dir.glob("*.txt"))
    # txt_file_list = pick_random_items(txt_file_list, 10)

    for txt_file in txt_file_list:
        trajectory_filename = txt_file.name
        print(f"Processing {trajectory_filename}")

        # Use the impute function for each file
        imputed_node_num = impute(
            graph,
            input_file_path=input_dir,
            output_file_path=output_dir,
            output_fig_path=output_fig_dir,
            trajectory_filename=trajectory_filename,
            neighbor_radius=neighbor_radius,
            trajectory_dist_threshold=trajectory_dist_threshold,
            curve_buffer_dist=curve_buffer_dist,
            straight_buffer_dist=straight_buffer_dist,
            bezier_factor=bezier_factor
        )
        imputed_node_num_total =  imputed_node_num_total + imputed_node_num
        cnt = cnt + 1

    return imputed_node_num_total, cnt

# Specify input and output paths and parameters

# input_file_path = Path('data') / 'trajectory'
# output_file_path = Path('data') / 'new_output'
# output_file_path.mkdir(parents=True, exist_ok=True)

# Parameters
neighbor_radius = 0.0064
trajectory_dist_threshold = 1  # unit km
curve_buffer_dist = 0.03  # km
straight_buffer_dist = 0.03  # km
bezier_factor = 0.15

input_file_root_path = Path('data') / 'input_imputation/test/sparsed/all'
output_file_root_path = Path('data') / 'new_output'
output_fig_root_path = Path('data') / 'output_fig'

trajectory_type_list = ['2_gap', '4_gap', '6_gap', '8_gap']
# trajectory_type_list = ['single_gap']

random.seed(42)
for trajectory_type in trajectory_type_list:
    type_path = os.path.join(input_file_root_path, trajectory_type)
    if os.path.isdir(type_path):
        trajectory_type_path = os.path.join(input_file_root_path, trajectory_type)

        if 'realistic' in trajectory_type_path:
            ship_type_list = os.listdir(trajectory_type_path)
            cnt = 0
            imputed_node_num_total = 0
            start_time = time.time()

            for ship_type in ship_type_list:
                ship_type_path = os.path.join(trajectory_type_path, ship_type)
                input_file_path = ship_type_path
                output_file_path = output_file_root_path / trajectory_type / '10' / ship_type
                output_fig_path = output_fig_root_path / trajectory_type / '10' / ship_type
                output_file_path.mkdir(parents=True, exist_ok=True)
                output_fig_path.mkdir(parents=True, exist_ok=True)

                print(f"Processing {input_file_path}")
                print(f"Processing {output_file_path}")
                # Process all trajectories
                imputed_node_num, trajectory_num = process_all_trajectories(
                    graph=main_graph,
                    input_dir=ship_type_path,
                    output_dir=output_file_path,
                    output_fig_dir=output_fig_path,
                    neighbor_radius=neighbor_radius,
                    trajectory_dist_threshold=trajectory_dist_threshold,
                    curve_buffer_dist=curve_buffer_dist,
                    straight_buffer_dist=straight_buffer_dist,
                    bezier_factor=bezier_factor
                )
                cnt = cnt + trajectory_num
                imputed_node_num_total = imputed_node_num_total + imputed_node_num

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"程序运行时间 mean:{trajectory_type_path} : {elapsed_time / cnt} 秒")
            print(f"impute node num mean:{trajectory_type_path} : {imputed_node_num_total / cnt}")

        else:
            hole_list = os.listdir(trajectory_type_path)
            for hole in hole_list:
                cnt = 0
                imputed_node_num_total = 0
                start_time = time.time()

                hole_path = os.path.join(trajectory_type_path, hole)
                if os.path.isdir(hole_path):
                    ship_type_list = os.listdir(hole_path)

                    for ship_type in ship_type_list:
                        ship_type_path = os.path.join(hole_path, ship_type)
                        input_file_path = ship_type_path
                        output_file_path = output_file_root_path / trajectory_type / hole / ship_type
                        output_fig_path = output_fig_root_path / trajectory_type / hole / ship_type
                        output_file_path.mkdir(parents=True, exist_ok=True)
                        output_fig_path.mkdir(parents=True, exist_ok=True)

                        print(f"Processing {input_file_path}")
                        print(f"Processing {output_file_path}")
                        # Process all trajectories
                        imputed_node_num, trajectory_num = process_all_trajectories(
                            graph=main_graph,
                            input_dir=ship_type_path,
                            output_dir=output_file_path,
                            output_fig_dir = output_fig_path,
                            neighbor_radius=neighbor_radius,
                            trajectory_dist_threshold=trajectory_dist_threshold,
                            curve_buffer_dist=curve_buffer_dist,
                            straight_buffer_dist=straight_buffer_dist,
                            bezier_factor=bezier_factor
                        )
                        cnt = cnt + trajectory_num
                        imputed_node_num_total = imputed_node_num_total + imputed_node_num
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"程序运行时间 mean:{hole_path} : {elapsed_time / cnt} 秒")
                print(f"impute node num mean:{hole_path} : {imputed_node_num_total / cnt}")