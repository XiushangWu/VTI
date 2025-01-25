import os
import pandas as pd
from graph_construction_module.graph import create_graphs_for_cells, process_all_cells
from imputation_module.imputation_v2 import load_graphs_and_impute_trajectory, load_intersecting_graphs_and_impute_trajectory
from evaluation.compare import compare_linear, compare_imputed, compare_gti
from copy import deepcopy
import networkx as nx
import json
import os
from concurrent.futures import ThreadPoolExecutor

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
        print(f'Error occurred trying to retrieve graph: {repr(e)}')

def load_complete_graph(graph_path):
    G = nx.Graph()

    # Walk through the directories in the graph_path
    for root, dirs, files in os.walk(graph_path):
        node_file = None
        edge_file = None
        
        # Check if the current directory has the required geojson files
        for file in files:
            if file == 'nodes.geojson':
                node_file = os.path.join(root, file)
            elif file == 'edges.geojson':
                edge_file = os.path.join(root, file)

        # If both files are found, load them into a graph and merge it with the main graph
        if node_file and edge_file:
            print("begin graph read node_file:{}".format(node_file))
            G_sub = create_graph_from_geojson(node_file, edge_file)
            G = nx.compose(G, G_sub)  # Merge the current subgraph into the main graph
            print("end graph read node_file:{} ".format(node_file))

        # # If both files are found, load them into a graph and merge it with the main graph
        # if node_file and edge_file and root.endswith('11_10'):
        #     print("begin graph read node_file:{}".format(node_file))
        #     G_sub = create_graph_from_geojson(node_file, edge_file)
        #     G = nx.compose(G, G_sub)  # Merge the current subgraph into the main graph
        #     print("end graph read node_file:{} ".format(node_file))

    return G

def load_all_graph_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):

    original_graph = load_complete_graph(graph_path)
    # nx.write_graphml(original_graph, "graph.graphml") ######

    # original_graph = load_complete_graph(graph_path)
    #original_graph = nx.read_graphml("graph.graphml") ######

    for root, dirs, files in os.walk(sparse_trajectories):
        print("Impute the trajectories in folder {}".format(root))
        # if 'data_combined_txt' not in root: ######
        #     print("skip") ######
        #     continue ######
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Imputing trajectory {file_name}")
                load_graphs_and_impute_trajectory(file_name, file_path, original_graph, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)

    # for root, dirs, files in os.walk(sparse_trajectories):
    #     print("Impute the trajectories in folder {}".format(root))
    #     # if 'data_combined_txt' not in root: ######
    #     #     print("skip") ######
    #     #     continue ######
    #     for file_name in files:
    #         if file_name.endswith('.txt') and 'aisdk-2024-10-31_Class_A_MMSI_209525000_8gaps_15km.txt' in file_name :
    #             file_path = os.path.join(root, file_name)
    #             print(f"Imputing trajectory {file_name}")
    #             load_graphs_and_impute_trajectory(file_name, file_path, original_graph, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)

                
def load_intersecting_graphs_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold):
    for root, dirs, files in os.walk(sparse_trajectories):
        print("Impute the trajectories in folder {}".format(root))
        # if 'data_combined_txt' not in root:
        #     print("skip")
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                print(f"Imputing trajectory {file_name}")
                load_intersecting_graphs_and_impute_trajectory(file_name, file_path, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, type, size)
            
sparse = 4000 # 500, 1000, 2000, 4000, 8000
types = 'many_gap' #'many_gap', 'single_gap', 'realistic'
def process_trajectory(size, type):
    CELLS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI//data//cells.txt')

    node_dist_threshold = [0.0008]
    edge_dist_threshold = 0.0016 
    cog_angle_threshold = 180
    graph_output_name = 'final_graph_cargo' #final_graph_cargo and final_graph_fishing
        
    for node_dist_threshold in node_dist_threshold:
        edge_dist_threshold = node_dist_threshold * 2 
        graph_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//output_graph//{graph_output_name}_{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')
        cells_data = pd.read_csv(CELLS, index_col='cell_id')
        sparse_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//input_imputation//test//sparsed_exam//all//{type}//{size}')
        #sparse_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI//data//txt_data_25_trajectories//20240831//data_multiple')

        """
            Create graphs and connect them
        """
        #create_graphs_for_cells(node_dist_threshold, edge_dist_threshold, cog_angle_threshold, graph_output_name)
        #process_all_cells(cells_data, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, graph_output_name)

        """
            Impute all trajectories in test folder
        """

        load_intersecting_graphs_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold)
        #load_all_graph_process_trajectories(type, size, sparse_trajectories, graph_path, node_dist_threshold, edge_dist_threshold, cog_angle_threshold)

#         print("comparing trajectories")
#         imputed_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'VTI/data/output_imputation/area/{type}/{size}/{node_dist_threshold}_{edge_dist_threshold}_{cog_angle_threshold}')

#         original_trajectories = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VTI/data/input_imputation/test/original_area')

#         #compare_linear(original_trajectories_linear_dgivt, size, type, sparse_trajectories_dgivt)
#         compare_imputed(imputed_trajectories, original_trajectories, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, size, type)
#         #compare_gti(imputed_trajectories, original_trajectories_gti, node_dist_threshold, edge_dist_threshold, cog_angle_threshold, size, type, sparse_trajectories_gti, imputed_trajectories_gti)

# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(process_trajectory, size, type) for size in sparse for type in types]

#     for future in futures:
#         future.result()

process_trajectory(sparse, types)