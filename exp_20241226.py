# Cell:
import os
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
  graph = nx.Graph() # TODO: consider using directed graph
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

  a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
  c = 2*asin(sqrt(a))

  return R * c

def custom_distance(coord1: tuple[float, float], coord2: tuple[float, float]):
  return geopy.distance.geodesic(coord1[::-1], coord2[::-1]).km

def heuristics(coord1, coord2):
  return custom_distance(coord1, coord2)

# Cell:
def generate_curve_polygon(start: dict, end: dict, buffer_dist: float, bezier_factor: float) -> Polygon:
  start_xy = start['xy'][::-1]
  end_xy = end['xy'][::-1]
  start_cog = start['cog']
  # end_cog = end['cog']

  # Number of points along the curve
  n_points = 15

  # Convert COGs to radians for calculation
  def cog_to_rad(cog):
     return np.radians((450 - cog) % 360)

  # Calculate control points for the Bezier curve
  def calculate_control_points(start_xy, end_xy, start_cog, factor=bezier_factor):
    # Convert COG to radians
    start_cog_rad = cog_to_rad(start_cog)
    
    # Move the control points in the direction of the COGs
    start_control = (start_xy[0] + factor * np.sin(start_cog_rad), start_xy[1] + factor * np.cos(start_cog_rad))
    # Return control points
    return [start_xy, start_control, end_xy]

  # Calculate points along a quadratic Bezier curve
  def bezier_curve(start_point, control_point_1, end_point, n_points=10):
    points = []
    for t in np.linspace(0, 1, n_points):
      # Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
      x = (1 - t)**2 * start_point[0] + 2 * (1 - t) * t * control_point_1[0] + t**2 * end_point[0]
      y = (1 - t)**2 * start_point[1] + 2 * (1 - t) * t * control_point_1[1] + t**2 * end_point[1]
      points.append((x, y))
    return points

  # Generate control points for the Bezier curve
  control_points = calculate_control_points(start_xy, end_xy, start_cog)

  # Generate points along the Bezier curve
  bezier_points = bezier_curve(control_points[0], control_points[1], control_points[2], n_points)

  # Create a Shapely LineString
  line = LineString(bezier_points)
  # Create a buffer around the LineString
  buffered_area = line.buffer(buffer_dist)

  # # Visualization of the Bezier curve and the points
  # plt.figure(figsize=(10, 6))
  # x, y = line.xy
  # plt.plot(x, y, label="Bezier Curve", color='blue', alpha=10.8)

  # # Plot the start and end points
  # plt.scatter([start_xy[10], end_xy[10]], [start_xy[1], end_xy[1]], color='red', label='Start and End Points')

  # # Plot the control points
  # control_x = [point[10] for point in control_points]
  # control_y = [point[1] for point in control_points]
  # plt.scatter(control_x, control_y, color='green', label='Control Points')

  # # # Plot the Bezier points
  # # bezier_x = [point[10] for point in bezier_points]
  # # bezier_y = [point[1] for point in bezier_points]
  # # plt.scatter(bezier_x, bezier_y, color='purple', label='Bezier Points', zorder=5)

  # # Plot the buffer
  # buffer_x, buffer_y = buffered_area.exterior.xy
  # plt.fill(buffer_x, buffer_y, color='lightblue', alpha=10.5, label='Buffered Area')

  # plt.title("Bezier Curve with Start, End, Control, and Bezier Points")
  # plt.xlabel("Longitude")
  # plt.ylabel("Latitude")
  # plt.legend(loc="upper left", fontsize='small')
  # plt.grid()
  # plt.show()

  return buffered_area

def generate_straight_polygon(start_node: dict, end_node: dict, width: float) -> Polygon:
  line = LineString([start_node['xy'][::-1], end_node['xy'][::-1]])
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
  bezier_factor: float
) -> tuple[list[tuple[float, float]], set, set]:
  path = [(trajectory_nodes[0]['longitude'], trajectory_nodes[0]['latitude'])]
  ship_draught = trajectory_nodes[0]['draught']
  kd_tree = build_kd_tree(graph)
  
  # nodes to be removed after imputation to keep graph clean
  to_be_removed_nodes = set()
  # nodes that helps imputation
  auxiliary_nodes = set()
  # nodes found with astar search
  imputed_nodes =set()
  # input nodes from .txt file
  input_nodes = set()
  
  # add input trajectory nodes to main graph
  for node in trajectory_nodes:
    xy = node['xy']
    graph.add_node(xy, geometry=Point(*xy), sog=node['sog'], cog=node['cog'], draught=node['draught'])
    to_be_removed_nodes.add(xy)
    input_nodes.add(xy)

  for i in tqdm(range(len(trajectory_nodes) - 1)):
    start_node, end_node = trajectory_nodes[i], trajectory_nodes[i + 1]
    start_pos = start_node['xy']
    end_pos = end_node['xy']
        
    # points are too close, don't need imputation
    geodesic_dist = custom_distance(start_pos, end_pos)
    if geodesic_dist <= trajectory_dist_threshold:
      path.append(end_pos)
    else:
      # try imputation
      gap_distance_est, cog_diff = estimate_gap_dist_and_cog(start_node, end_node)
      print(f"gap_distance_est={gap_distance_est} geodesic_dist={geodesic_dist} cog_diff={cog_diff} division={gap_distance_est/geodesic_dist}")
      print(f"start {start_node}")
      print(f"end {end_node}")
      print("---------")

      # astar path must stay inside the genrated polygon
      polygon = None
      if cog_diff > 50:
        polygon = generate_curve_polygon(start_node, end_node, curve_buffer_dist, bezier_factor)
      else:
        polygon = generate_straight_polygon(start_node, end_node, straight_buffer_dist)
      
      # add nodes to graph for printing
      polygon_points = [point[::-1] for point in polygon.boundary.coords]
      for new_node in polygon_points:
        graph.add_node(new_node, geometry=Point(*new_node))
      to_be_removed_nodes.update(polygon_points)
      auxiliary_nodes.update(polygon_points)

      # # plot polygan
      # plt.figure(figsize=(10, 6))
      # buffer_x, buffer_y = polygon.exterior.xy
      # plt.fill(buffer_x, buffer_y, color='lightblue', alpha=10.5, label='Buffered Area')

      # plt.title("Bezier Curve with Start, End, Control, and Bezier Points")
      # plt.xlabel("Longitude")
      # plt.ylabel("Latitude")
      # plt.legend(loc="upper left", fontsize='small')
      # plt.grid()
      # plt.show()

      neighbor_nodes = find_and_connect_with_possible_neighbors(graph, start_pos, neighbor_radius, kd_tree)
      neighbor_nodes |= find_and_connect_with_possible_neighbors(graph, end_pos, neighbor_radius, kd_tree)
      auxiliary_nodes |= neighbor_nodes

      print(f"number of new neighbor nodes: {len(neighbor_nodes)}")

      # define locally to take into account the polygon
      def astar_weight(a, b, attrs):
        # only check path that fall inside the polygon
        if not polygon.contains(Point(*a[::-1])) or not polygon.contains(Point(*b[::-1])):
          return None
        if graph.nodes[a].get("draught", ship_draught) < ship_draught or graph.nodes[b].get("draught", ship_draught) < ship_draught:
          return None
        return custom_distance(a, b)
      
      try:
        new_path = nx.astar_path(graph, start_pos, end_pos, heuristic=heuristics, weight=astar_weight)
        imputed_nodes.update(new_path[1:-1])
        path += new_path[1:] # exclude start point
      except nx.NetworkXNoPath:
        path.append(end_pos)
        print(f"ERROR: no astar path found between node {i} and node {i + 1}")
  return path, input_nodes, imputed_nodes, to_be_removed_nodes, auxiliary_nodes

@time_function
def generate_output_files_for_nodes(
  nodes: list[tuple[float, float]],
  graph: nx.Graph, 
  file_path: Path
) -> None:
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
  trajectory_filename: list[dict], 
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
  updated_output_path.mkdir(parents=True, exist_ok=True)

  generate_output_files_for_nodes(list(input_nodes), graph, updated_output_path / 'input_nodes')
  generate_output_files_for_nodes(list(imputed_nodes), graph, updated_output_path / 'imputed_nodes')
  generate_output_files_for_nodes(list(auxiliary_nodes), graph, updated_output_path / 'auxiliary_nodes')
  edges = [[(path[i]), list(path[i + 1])] for i in range(len(path) - 1)]
  generate_output_files_for_edges(edges, updated_output_path / 'edges')

  # remove added nodes to keep main graph clean
  graph.remove_nodes_from(to_be_removed_nodes)


# Cell:
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_314437000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_4gaps_20km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_245250000_4gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_6gaps_15km'
# trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_6gaps_20km'
trajectory_filename = 'aisdk-2024-10-31_Class_A_MMSI_209525000_8gaps_15km'

input_file_path = Path('data') / 'trajectory'
output_file_path = Path('data') / 'new_output'
output_file_path.mkdir(parents=True, exist_ok=True)

# params
neighbor_radius = 0.0064
trajectory_dist_threshold = 1 # unit km
curve_buffer_dist = 0.01
straight_buffer_dist = 0.03
bezier_factor = 0.15

impute(main_graph, input_file_path, output_file_path, trajectory_filename, neighbor_radius, trajectory_dist_threshold, curve_buffer_dist, straight_buffer_dist, bezier_factor)

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

# Cell:


