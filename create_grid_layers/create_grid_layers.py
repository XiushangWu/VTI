import os
import warnings
import rasterio
import psycopg2
import sqlalchemy
import geoalchemy2
import numpy as np
import pandas as pd
import geopandas as gpd
from .connect import connect
from .config import load_config
from pyproj import Transformer, CRS
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point, box
from geoalchemy2.shape import from_shape
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Float, Integer, ForeignKey

#DEPTH_MAP = 'C:/Users/alexf/Desktop/ddm_50m.dybde.tiff'
DEPTH_MAP = '/srv/P-10/ddm_50m.dybde.tiff'

# Assuming load_config is a function you've defined to load your database configuration
config = load_config()

# Database setup
engine_url = f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
engine = create_engine(engine_url)
Base = declarative_base()


class DepthPoint(Base):
    __tablename__ = 'depth_points'
    id = Column(Integer, primary_key=True)
    geometry = Column(geoalchemy2.Geometry('POINT', srid=4326))
    depth = Column(Float)

class Grid_1600(Base):
    __tablename__ = 'grid_1600'
    id = Column(Integer, primary_key=True)
    geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))

# class Grid_800(Base):
#     __tablename__ = 'grid_800'
#     id = Column(Integer, primary_key=True)
#     geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))
#     grid_1600_id = Column(Integer, ForeignKey('grid_1600.id'))

# class Grid_400(Base):
#     __tablename__ = 'grid_400'
#     id = Column(Integer, primary_key=True)
#     geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))
#     grid_800_id = Column(Integer, ForeignKey('grid_800.id'))

# class Grid_200(Base):
#     __tablename__ = 'grid_200'
#     id = Column(Integer, primary_key=True)
#     geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))
#     grid_400_id = Column(Integer, ForeignKey('grid_400.id'))

# class Grid_100(Base):
#     __tablename__ = 'grid_100'
#     id = Column(Integer, primary_key=True)
#     geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))
#     Grid_200_id = Column(Integer, ForeignKey('Grid_200.id'))

# class Grid_50(Base):
#     __tablename__ = 'grid_50'
#     id = Column(Integer, primary_key=True)
#     geometry = Column(geoalchemy2.Geometry('POLYGON', srid=4326))
#     grid_100_id = Column(Integer, ForeignKey('grid_100.id'))

# Create the tables in the database
Base.metadata.create_all(engine)


def extract_depth_map():
    processed_pixels = 0
    batch_size = 100000

    print("Starting depth map extraction")

    with rasterio.open(DEPTH_MAP, crs="3034") as dataset:
        msk = dataset.read_masks(1)
        transform = dataset.transform
        width = dataset.width
        height = dataset.height
        depth_values = dataset.read(1)
        transformer = Transformer.from_crs(CRS.from_epsg(3034), CRS.from_epsg(4326))

        points = []
        depths = []

        # Iterate through all pixels and print their corresponding latitudes and longitudes
        for row in range(height):
            for col in range(width):
                if msk[row, col] > 0:
                    # Calculate latitude and longitude for each pixel
                    lat, lon = transform * (col + 0.5, row + 0.5)
                    lon_trans, lat_trans = transformer.transform(lon, lat)

                    # Append to temporary lists
                    points.append(Point(lat_trans, lon_trans))
                    depths.append(depth_values[row, col])

                    processed_pixels += 1

                    # Check if the batch size is reached
                    if len(points) == batch_size:
                        # Convert to GeoDataFrame and insert
                        gdf = gpd.GeoDataFrame({'geometry': points, 'depth': depths}, crs="EPSG:4326")
                        insert_data_into_db_gdf(gdf)
                        
                        # Clear the lists for the next batch
                        points.clear()
                        depths.clear()

        # Insert any remaining points after looping through all pixels
        if points:
            gdf = gpd.GeoDataFrame({'geometry': points, 'depth': depths}, crs="EPSG:4326")
            insert_data_into_db_gdf(gdf)


def insert_data_into_db_gdf(gdf):
    Session = sessionmaker(bind=engine)
    session = Session()

    for index, row in gdf.iterrows():
        wkt_element = geoalchemy2.WKTElement(row['geometry'].wkt, srid=4326)
        depth_point = DepthPoint(geometry=wkt_element, depth=row['depth'])
        session.add(depth_point)
    
    session.commit()
    session.close()

    print(f"Inserted batch of {len(gdf)} points into the database.")


def create_and_insert_grid_into_db(cell_size_km):

    print("Creating grid layer")
    grid_gdf = create_grid_layer(cell_size_km)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        for _, row in grid_gdf.iterrows():
            # Convert the GeoDataFrame geometry to a GeoAlchemy shape
            geom = from_shape(row['geometry'], srid=4326)  # Ensure SRID matches your data
            
            # Create an instance of the Grid_1600 class for each row
            grid_entry = Grid_1600(geometry=geom)
            
            # Add the instance to the session
            session.add(grid_entry)
        
        # Commit the session to insert the records into the database
        session.commit()
        print("Grid data insertion was successful.")
    except Exception as e:
        print(f"An error occurred during grid data insertion: {e}")
        session.rollback()  # Roll back the session in case of error
    finally:
        session.close()







def create_child_grids_and_insert_grid_into_db(cell_size_km):

    print("Creating grid layer")
    grid_gdf = create_grid_layer(cell_size_km)

    Session = sessionmaker(bind=engine)
    session = Session()

    parent_grid_id = f"SELECT * FROM grid_1600"
    parent_grid_id_gdf = gpd.read_postgis(parent_grid_id, engine, geom_col='geometry')

    # Create a new column 'parent_grid_id' in grid_gdf and populate with parent_grid_id if it is contianed within it
    grid_gdf['grid_1600_id'] = None
    grid_gdf['grid_1600_id'] = grid_gdf.apply(lambda row: find_parent_grid(row['geometry'], parent_grid_id_gdf), axis=1)

    try:
        for _, row in grid_gdf.iterrows():
            # Convert the GeoDataFrame geometry to a GeoAlchemy shape
            geom = from_shape(row['geometry'], srid=4326)  # Ensure SRID matches your data
            
            # Create an instance of the Grid_800 class for each row
            grid_entry = Grid_800(geometry=geom, grid_1600_id=row['grid_1600_id'])
            
            # Add the instance to the session
            session.add(grid_entry)
        
        # Commit the session to insert the records into the database
        session.commit()
        print("Grid data insertion was successful.")
    except Exception as e:
        print(f"An error occurred during grid data insertion: {e}")
        session.rollback()  # Roll back the session in case of error
    finally:
        session.close()


def find_parent_grid(geometry, parent_grid):
    centroid = geometry.centroid
    intersecting_grid = parent_grid[parent_grid['geometry'].intersects(centroid)]
    
    if not intersecting_grid.empty:
        return intersecting_grid['id'].values[0]
    else:
        return None

def create_grid_layer(cell_size_km):
    min_lat, min_lon, max_lat, max_lon = 53.00, 7.00, 59.00, 17.00

    # Calculate the number of cells in the latitude and longitude directions
    n_cells_lat = int((max_lat - min_lat) / (cell_size_km / 111.32))  # Approximate conversion from km to degrees
    n_cells_lon = int((max_lon - min_lon) / (cell_size_km / (111.32 * np.cos(np.radians(np.mean([min_lat, max_lat]))))))

    lon_step = cell_size_km / (111.32 * np.cos(np.radians(np.mean([min_lat, max_lat]))))
    n_cells_lon = int((max_lon - min_lon) / lon_step)

    # Create grid cells
    grid_cells = []
    for i in range(n_cells_lat):
        for j in range(n_cells_lon):
            lat0 = min_lat + i * (cell_size_km / 111.32)  # Convert to degrees
            lon0 = min_lon + j * lon_step
            lat1 = lat0 + (cell_size_km / 111.32)  # Convert to degrees
            lon1 = lon0 + lon_step

            grid_cells.append(box(lon0, lat0, lon1, lat1))

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

    return grid_gdf

#extract_depth_map()
create_and_insert_grid_into_db(1.6)
