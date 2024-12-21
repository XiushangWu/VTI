import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Path where the extracted files are located
input_folder = r'C:\Users\HU84VR\Downloads\VTI\data\output_imputation\raw\skagen\many_gap\4000'

# Path to save the plots
save_path = r'C:\Users\HU84VR\Downloads\Skagen raw output'

# Ensure the save directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Walk through the folder and find all "nodes.geojson" files
node_files = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if 'nodes.geojson' in file:
            node_files.append(os.path.join(root, file))

# Iterate over all node files and plot trajectories
for node_file in node_files:

    # Load the geojson file using GeoPandas
    gdf = gpd.read_file(node_file)

    # Create a figure for each plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract coordinates from the GeoDataFrame
    ax.plot(gdf['geometry'].x, gdf['geometry'].y, 'bo-', markersize=5, label='Trajectory')

    # Set plot labels and title
    title = os.path.basename(os.path.dirname(node_file))  # Get folder name for the title
    ax.set_title(f'Trajectory Plot: {title}', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True)
    plt.legend()

    # Save the plot
    save_file_name = f"{title}_trajectory.png"
    save_file_path = os.path.join(save_path, save_file_name)
    fig.savefig(save_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the plot to avoid display

print("All plots saved successfully.")
