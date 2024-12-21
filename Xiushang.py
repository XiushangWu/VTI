import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint

# Base directory where the extracted folders are located
base_dir = 'C:/Users/HU84VR/Downloads/VTI/data/output_imputation/raw/skagen/many_gap/4000/0.0008_0.0016_180'

# Initialize an empty list to store DataFrames for each trajectory
trajectories = []

# Loop through each folder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith(".geojson"):
                try:
                    gdf = gpd.read_file(file_path)
                    gdf = gdf.explode(index_parts=False)  # Explode multi-part geometries
                    trajectories.append(gdf)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Combine all trajectories
if trajectories:
    all_trajectories = pd.concat(trajectories, ignore_index=True)

    # Plot the world basemap
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Plot the world basemap and the trajectories on top
    fig, ax = plt.subplots(figsize=(15, 10))
    world.plot(ax=ax, color='lightgray')

    # Plot the trajectories
    all_trajectories.plot(ax=ax, marker='o', color='red', markersize=1, label="Trajectories")

    # Add plot details
    plt.title("Trajectories based on GeoJSON files")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()
