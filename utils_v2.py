import numpy as np
import math

from utils import haversine_distance

def plot_tra(points, title):
    import matplotlib.pyplot as plt
    return
    # lats, lons = zip(*points)
    # plt.plot(lons, lats, marker='o', linestyle='', markersize=2)
    # #plt.scatter([start_lon, end_lon], [start_lat, end_lat], color='red')  # 起点和终点
    # #plt.scatter([control_point1[1], control_point2[1]], [control_point1[10], control_point2[10]], color='green')  # 控制点
    # plt.title(title)
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid()
    # plt.show()

def plot_tra_with_start_end(start, end, points, title):
    import matplotlib.pyplot as plt
    return
    # lats, lons = zip(*points)
    # plt.plot(lons, lats, marker='o', linestyle='', markersize=2)
    # plt.scatter([start[1], end[1]], [start[10], end[10]], color='red')  # 起点和终点
    # # plt.scatter([control_point1[1], control_point2[1]], [control_point1[10], control_point2[10]], color='green')  # 控制点
    # plt.title(title)
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.grid()
    # plt.show()

def adjust_edge_weights_for_draught_v2(G, start_trajectory_point, end_trajectory_point, tree, node_positions, start_draught, edge_dist_threshold, base_penalty=1000):
    start_props = start_trajectory_point["properties"]
    start_lat, start_lon = start_props["latitude"], start_props["longitude"]
    start_bearing = start_props["cog"]
    start_ts = start_props["timestamp"]
    start_speed = start_props["sog"]

    end_props = end_trajectory_point["properties"]
    end_lat, end_lon = end_props["latitude"], end_props["longitude"]
    end_bearing = end_props["cog"] - 180
    end_ts = start_props["timestamp"]
    end_speed = start_props["sog"]

    trajectory = pre_curve_generate(start_trajectory_point, end_trajectory_point)
    plot_tra_with_start_end([start_lat, start_lon], [end_lat, end_lon], trajectory, 'curve_generate')

    radius = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    radiusNew = 50

    valid_nodes = set()
    for p in trajectory:
        start_indices_within_radius = tree.query_ball_point([p[0], p[1]], edge_dist_threshold * 2)
        #end_indices_within_radius = tree.query_ball_point([end_point[10], end_point[1]], radiusNew)

        start_nodes_within = set([list(G.nodes)[i] for i in start_indices_within_radius])
        #end_nodes_within = set([list(G.nodes)[i] for i in end_indices_within_radius])

        #relevant_nodes = start_nodes_within.union(end_nodes_within)
        relevant_nodes = start_nodes_within
        for node in relevant_nodes:
            if G.nodes[node].get('avg_depth') is None and G.nodes[node].get('draught') is None:
                pass
            else:
                node_depth = abs(G.nodes[node].get('avg_depth', G.nodes[node].get('draught')))
                if node_depth >= start_draught * 1.2:
                    valid_nodes.add(node)

    valid_nodes.add((start_lat, start_lon))
    valid_nodes.add((end_lat, end_lon))

    subgraph = G.subgraph(valid_nodes).copy()

    return subgraph

def pre_curve_generate(start_point, end_point):
    start_props = start_point["properties"]
    start_lat, start_lon = start_props["latitude"], start_props["longitude"]
    start_bearing = start_props["cog"]
    start_ts = start_props["timestamp"]
    start_speed = start_props["sog"]

    end_props = end_point["properties"]
    end_lat, end_lon = end_props["latitude"], end_props["longitude"]
    end_bearing = end_props["cog"] - 180
    end_ts = end_props["timestamp"]
    end_speed = end_props["sog"]

    # 海里转m/s单位
    mean_speed = (start_speed + end_speed) / 2 * 0.514444
    total_distance = mean_speed * (end_ts - start_ts)

    control_point1 = offset_point(start_lat, start_lon, total_distance * 0.25, start_bearing)
    control_point2 = offset_point(end_lat, end_lon, total_distance * 0.25, end_bearing)
    trajectory = bezier_curve((start_lat, start_lon), control_point1, control_point2, (end_lat, end_lon), n_points=150)
    #trajectory = smooth_points_with_distance(trajectory)
    return trajectory

def bezier_curve(p0, p1, p2, p3, n_points=100):
    """
    生成贝塞尔曲线上的点。

    参数:
        p0 (tuple): 起点坐标 (lat, lon)
        p1 (tuple): 第一个控制点 (lat, lon)
        p2 (tuple): 第二个控制点 (lat, lon)
        p3 (tuple): 终点坐标 (lat, lon)
        n_points (int): 生成的轨迹点数量

    返回:
        List[Tuple[float, float]]: 贝塞尔曲线轨迹点列表
    """
    t_values = np.linspace(0, 1, n_points)
    curve = []
    for t in t_values:
        x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
        y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        curve.append((x, y))
    return curve

def smooth_points_with_distance(curve, max_distance=80):
    from geopy.distance import geodesic
    """
    根据目标距离平滑生成均匀分布的点。

    参数:
        points (List[Tuple[float, float]]): 输入轨迹点 (lat, lon)
        target_distance (float): 相邻点之间的目标距离（米）

    返回:
        List[Tuple[float, float]]: 平滑后的轨迹点
    """
    smoothed_curve = [curve[0]]  # 初始化平滑曲线，起点作为第一个点

    for i in range(1, len(curve)):
        prev_point = smoothed_curve[-1]
        current_point = curve[i]

        # 计算当前点与上一点之间的距离
        distance = geodesic(prev_point, current_point).meters

        if distance > max_distance:
            # 插值点数量
            num_intermediate_points = int(np.ceil(distance / max_distance))
            for j in range(1, num_intermediate_points + 1):
                lat = prev_point[0] + (current_point[0] - prev_point[0]) * j / num_intermediate_points
                lon = prev_point[1] + (current_point[1] - prev_point[1]) * j / num_intermediate_points
                smoothed_curve.append((lat, lon))
        else:
            smoothed_curve.append(current_point)

    return smoothed_curve

def offset_point(lat, lon, distance, bearing):
    """
    根据起始点、距离和方向计算新坐标点 (近似法)。

    参数:
        lat (float): 起始纬度
        lon (float): 起始经度
        distance (float): 距离 (单位：米)
        bearing (float): 方向角 (度)

    返回:
        (float, float): 新的坐标点 (lat, lon)
    """
    R = 6371000  # 地球半径 (米)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)

    lat2_rad = math.asin(math.sin(lat_rad) * math.cos(distance / R) +
                         math.cos(lat_rad) * math.sin(distance / R) * math.cos(bearing_rad))
    lon2_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / R) * math.cos(lat_rad),
                                    math.cos(distance / R) - math.sin(lat_rad) * math.sin(lat2_rad))

    return math.degrees(lat2_rad), math.degrees(lon2_rad)