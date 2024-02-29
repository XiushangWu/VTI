import logging 
import pandas as pd
import geopandas as gpd
from enum import Enum
import math


logging.basicConfig(level=logging.INFO)

class HarborState(Enum):
    ENTERING = 0
    LEAVING = 1
    AT_ANCHOR = 2
    PASSING_THROUGH = 3
    NONE = 4

def create_trajectories_from_df(harbors_df:gpd.GeoDataFrame, trajectory_df:gpd.GeoDataFrame):
    trajectory_df = trajectory_df.reset_index(drop=True)
    trajectories_df = order_by_diff_vessels(trajectory_df)
    trajectories_df = trajectories_df.drop_duplicates(subset=['vessel_id', 'timestamp'], keep='first')

    trajectories_df = trajectories_df.reset_index(drop=True) # to ensure indexes are still fine
    sub_trajectories_df = make_subtrajectories_split_from_harbor(harbors_df, trajectories_df)
    
    return sub_trajectories_df

def make_subtrajectories_split_from_harbor(harbors_df: gpd.GeoDataFrame, trajectories_df: gpd.GeoDataFrame):
    trajectories_df = add_in_harbor_column(trajectories_df, harbors_df)
    
    sub_trajectories = []
    
    for _, positions_df in trajectories_df.groupby('vessel_id'):
        in_harbor_trajectory = []
        current_sub_trajectory = []
        prev_position = None
        from_sea = False
        from_harbor = False
        itt = 0
            
        for _, current_position in positions_df.iterrows():
                
            if (prev_position is None and current_position.in_harbor):
                from_harbor = True
                from_sea = False
            elif (prev_position is None and not current_position.in_harbor):
                from_harbor = False
                from_sea = True
            else:
                if prev_position.geometry.equals(current_position.geometry):
                    prev_position = current_position
                    continue
            if from_sea and current_position.in_harbor and current_sub_trajectory:
                in_harbor_trajectory.append(current_position)
                prev_position = current_position                
            elif from_sea and not current_position.in_harbor and in_harbor_trajectory and current_sub_trajectory:
                (state, in_harbor_trajectory, harbor_trajectory_positions) = update_harbor_positions_and_return_state(in_harbor_trajectory)     
                if (state == HarborState.PASSING_THROUGH):
                    current_sub_trajectory.extend(harbor_trajectory_positions)
                    current_sub_trajectory.append(current_position)
                    in_harbor_trajectory = []
                    prev_position = current_position
                    continue
                elif (state == HarborState.ENTERING):
                    current_sub_trajectory.extend(harbor_trajectory_positions)
                    sub_trajectories.append(current_sub_trajectory)
                    current_sub_trajectory = []
                    prev_position = None
                    
            elif from_harbor and current_position.in_harbor:
                in_harbor_trajectory.append(current_position)
                prev_position = current_position
            elif from_harbor and not current_position.in_harbor:
                (state, in_harbor_trajectory, harbor_trajectory_positions) = update_harbor_positions_and_return_state(in_harbor_trajectory) 
                
                if (state == HarborState.LEAVING):
                    current_sub_trajectory.extend(harbor_trajectory_positions)
                    current_sub_trajectory.append(current_position)
                    in_harbor_trajectory = []
                    prev_position = current_position
                    continue
                else:
                    if (state == HarborState.AT_ANCHOR): # don't do anything
                        print('at anchor. seems right?')
                        in_harbor_trajectory = []
                        prev_position = None
                        pass
                    
                    prev_position = None
            elif (from_sea and not current_sub_trajectory and not current_position.in_harbor and in_harbor_trajectory): # beginning trajectory with leftover harbor positions
                (state, in_harbor_trajectory, harbor_trajectory_positions) = update_harbor_positions_and_return_state(in_harbor_trajectory) 
                if state == HarborState.LEAVING:
                    current_sub_trajectory.extend(harbor_trajectory_positions)
                    current_sub_trajectory.append(current_position)
                    prev_position = current_position
                
                in_harbor_trajectory = []
                itt += 1
                continue
            else:
                current_sub_trajectory.append(current_position)
                prev_position = current_position
                
                
        if current_sub_trajectory:
            sub_trajectories.append(current_sub_trajectory)
    
    # Flatten the list of lists
    flattened_sub_trajectories = [
        {'sub_trajectory_id': i, **point} 
        for i, sub_trajectory in enumerate(sub_trajectories, start=1) 
        for point in sub_trajectory
    ]
        
    # Convert to DataFrame
    sub_trajectory_df = pd.DataFrame(flattened_sub_trajectories)
        
    return sub_trajectory_df


def update_harbor_positions_and_return_state(in_harbor_positions:list) -> tuple[HarborState, list, list]:
    state: HarborState
    
    if len(in_harbor_positions) > 2:
        if in_harbor_positions[-1].sog > in_harbor_positions[0].sog and in_harbor_positions[-1].sog > 0.2: # if last position is faster than current, assuming leaving
            leaving_positions = get_leaving_harbor_positions(in_harbor_positions)
            
            state = HarborState.LEAVING
            
            return (state, [], leaving_positions)

        if in_harbor_positions[0].sog > 0.2: # assuming that we are entering to moor or perhaps passing through
            (index, entering_positions) = get_entering_harbor_positions(in_harbor_positions)
            if len(entering_positions) == 0:
                is_passing_through = get_entering_harbor_positions(in_harbor_positions)
                
                if (is_passing_through):
                    state = HarborState.PASSING_THROUGH
                    return (state, [], in_harbor_positions)
            else:
                in_harbor_positions = in_harbor_positions[index + 1:]
                state = HarborState.ENTERING
            return (state, in_harbor_positions, entering_positions)

        return (HarborState.AT_ANCHOR, [], []) 
    else:
        return (HarborState.PASSING_THROUGH, [], in_harbor_positions) 
    
def get_leaving_harbor_positions(positions:list) -> list:   
    reversed_positions = list(reversed(positions))
    reversed_indices = range(len(reversed_positions), 0, -1)
    leaving_position_index = next((i for i, (prev_position, current_position) in zip(reversed_indices, zip(reversed_positions, reversed_positions[1:])) 
                                if calculate_speed(current_position, prev_position) < 0.2), None)
    
    if leaving_position_index is None:
        leaving_position_index = next((i for i, current_position in enumerate(reversed_positions) if current_position.sog < 0.2), None)
    
    if leaving_position_index is not None:
        if (leaving_position_index + 1) > (len(reversed_positions) - 1):
            leaving_positions = reversed_positions[:leaving_position_index]
        else:
            leaving_positions = reversed_positions[:leaving_position_index + 1]   
            leaving_positions = reversed(reversed_positions) 
    else:
        # If no position above 0.2 was found, return an empty list or handle as needed
        leaving_positions = []
    
    return leaving_positions

def get_entering_harbor_positions(positions:list) -> tuple[int, list]:
    stopped_position_index = next((i for i, (prev_position, current_position) in enumerate(zip(positions, positions[1:]), start=1) 
                                if calculate_speed(prev_position, current_position) <= 0.2), None)
    
    if (stopped_position_index is None):
        stopped_position_index = next((i for i, current_position in enumerate(positions)
                                            if current_position.sog <= 0.2), None)
    
    if stopped_position_index is not None:
        if (stopped_position_index + 1) > (len(positions) - 1):
            entering_positions = positions[:stopped_position_index ]
        else:
            entering_positions = positions[:stopped_position_index + 1]
    else:
        # If no position above 0.2 was found, return an empty list or handle as needed
        entering_positions = []
        stopped_position_index = 0
    
    return (stopped_position_index, entering_positions)

def is_passing_through(positions:list) -> bool:
    indexes = next((i for i, (prev_position, current_position) in enumerate(zip(positions, positions[1:]), start=1) 
                                if calculate_speed(prev_position, current_position) <= 0.5), None)      
    
    if indexes is None:
        indexes = next((i for i, current_position in enumerate(positions) if current_position.sog <= 0.5), None)  
    
    if indexes is None:
        return False
    
    return True 
     
def add_in_harbor_column(trajectory_df:gpd.GeoDataFrame, harbors_df: gpd.GeoDataFrame):
    """
    Perform a spatial join between trajectory points and harbor polygons,
    and add a new column indicating whether each point is within a harbor.

    Parameters:
    trajectory_df (GeoPandas GeoDataFrame): GeoDataFrame containing trajectory points.
    harbors_df (GeoPandas GeoDataFrame): GeoDataFrame containing harbor polygons.

    Returns:
    GeoPandas GeoDataFrame: New GeoDataFrame with an additional 'in_harbor' column.
    """
    # Perform a spatial join to identify points from trajectory_df that intersect harbors
    points_with_harbors = gpd.sjoin(trajectory_df, harbors_df, how="left", predicate="intersects", lsuffix='left', rsuffix='right')

    # Create a new column 'in_harbor' with boolean values indicating whether each point is within a harbor
    points_with_harbors['in_harbor'] = ~points_with_harbors['index_right'].isnull()

    # Optional: Drop unnecessary columns from the resulting DataFrame
    points_with_harbors.drop(columns=['index_right'], inplace=True)

    return points_with_harbors

def make_sub_trajectories_by_radius(sorted_locations_gdf: gpd.GeoDataFrame):
    if sorted_locations_gdf.empty:
        return sorted_locations_gdf        
    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True)
    sorted_locations_gdf = order_by_diff_vessels(sorted_locations_gdf)
    sorted_locations_gdf = sorted_locations_gdf.drop_duplicates()
    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True) # to ensure indexes are still fine
    radius_threshold = 3 # meters, diameter is 6
    sub_trajectories = []
    sub_trajectory_id = 1
    moving = True
    
    for _, locations_df in sorted_locations_gdf.groupby('vessel_id'):
        current_sub_trajectory = []
        consecutive_points_within_radius = []
        
        for index, row in locations_df[1:].iterrows():
            current_location = row
            last_location = locations_df.iloc[index - 1]
            distance = current_location.geometry.distance(last_location.geometry)

            if (consecutive_points_within_radius):
                distance = consecutive_points_within_radius[0].geometry.distance(current_location.geometry)

            if moving:  # If moving
                if (current_location.geometry == last_location.geometry and current_location.sog == 0.0):  # Not moving
                    continue

                elif distance < radius_threshold:  # Check if within radius threshold
                    consecutive_points_within_radius.append(current_location)

                    if len(consecutive_points_within_radius) == 5:  
                        moving = False
                else:
                    if consecutive_points_within_radius:  # If non-empty
                        if len(consecutive_points_within_radius) < 5:
                            current_sub_trajectory += (consecutive_points_within_radius)
                            
                    consecutive_points_within_radius = []  # Reset
                    current_sub_trajectory.append(current_location)
            else:  # If not moving
                if distance > radius_threshold:
                    # create sub trjaectory df
                    if current_sub_trajectory and len(current_sub_trajectory) > 50: 
                        sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
                        sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
                        sub_trajectories.append(sub_trajectory_df)
                        sub_trajectory_id += 1

                    # reset for next sub trajectory
                    current_sub_trajectory = []  # Reset
                    consecutive_points_within_radius = []  # Reset
                    current_sub_trajectory.append(current_location)
                    moving = True
                else:
                    continue

        if current_sub_trajectory and len(current_sub_trajectory) > 50:  # Append the last sub-trajectory if not empty
            sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
            sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
            sub_trajectories.append(sub_trajectory_df)
    
    if (sub_trajectories):
        result_sub_trajectory_df = pd.concat(sub_trajectories, ignore_index=True)
        return result_sub_trajectory_df
                      
    return pd.DataFrame(columns=sorted_locations_gdf.columns)

def make_sub_trajectories_by_speed(sorted_locations_gdf: gpd.GeoDataFrame):
    if sorted_locations_gdf.empty:
        return sorted_locations_gdf
       
    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True)
    sorted_locations_gdf = order_by_diff_vessels(sorted_locations_gdf)
    sorted_locations_gdf = sorted_locations_gdf.drop_duplicates()
    sorted_locations_gdf = sorted_locations_gdf.reset_index(drop=True) # to ensure indexes are still fine
       
    sub_trajectories = []
    sub_trajectory_id = 1
    threshold_speed = 0.1
    prev_location = None
    
    for _, locations_df in sorted_locations_gdf.groupby('vessel_id'):
        current_sub_trajectory = []
        stationary_locations = []
        
        for index, row in locations_df.iterrows():
            curr_location = row
            is_stationary = len(stationary_locations) > 10
            
            if prev_location is not None:
                speed = calculate_speed(prev_location, curr_location)
                if (speed <= threshold_speed):
                    stationary_locations.append(curr_location)
                    continue
            else:
                if curr_location.sog > threshold_speed: 
                    current_sub_trajectory.append(curr_location)
                    continue
            
            prev_location = locations_df.iloc[index - 1]

            if (not is_stationary and stationary_locations):
                current_sub_trajectory.extend(stationary_locations)
                stationary_locations = []
                
                current_sub_trajectory.append(curr_location)
            else:
                current_sub_trajectory.append(curr_location)
                stationary_locations = []
            
            if (is_stationary):
                current_sub_trajectory.append(curr_location)
                
                if len(current_sub_trajectory) > 50:
                    sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
                    sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
                    sub_trajectories.append(sub_trajectory_df)
                    sub_trajectory_id += 1
                
                current_sub_trajectory = []  # Reset

        if current_sub_trajectory:  # Append the last sub-trajectory if not empty
            if len(current_sub_trajectory) > 50:
                sub_trajectory_df = pd.DataFrame(current_sub_trajectory, columns=sorted_locations_gdf.columns)
                sub_trajectory_df['sub_trajectory_id'] = sub_trajectory_id
                sub_trajectories.append(sub_trajectory_df)
                      
    
    if (sub_trajectories):
        result_sub_trajectory_df = pd.concat(sub_trajectories, ignore_index=True)
        return result_sub_trajectory_df
                      
    return pd.DataFrame(columns=sorted_locations_gdf.columns)

def order_by_diff_vessels(sorted_locations_df: gpd.GeoDataFrame):
    sorted_locations_df.fillna({'imo': -1, 'ship_type': 'None', 'width': -1, 'length': -1}, inplace=True)
    sorted_locations_df['vessel_id'] = sorted_locations_df.groupby(['imo', 'ship_type', 'width', 'length']).ngroup() 
    return sorted_locations_df

def calculate_speed(p1:gpd.GeoDataFrame, p2:gpd.GeoDataFrame) -> float:
    distance_in_meters = p2.geometry.distance(p1.geometry)
    distance_in_kilometers = distance_in_meters/1000
    
    time_in_seconds = p2.timestamp - p1.timestamp 
    time_in_hours = time_in_seconds/60/60
     
    if time_in_seconds == 0:
        return max(p1.sog, p2.sog)

    kilometers_pr_hour = distance_in_kilometers/time_in_hours
    knot_pr_hour = kilometers_pr_hour/1.85
        
    return knot_pr_hour