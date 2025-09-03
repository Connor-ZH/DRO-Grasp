import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import os
import torch

def preload_object_pcs_from_folder(device):
    cache = {}
    object_folder_path = os.path.join(ROOT_DIR, f'data/pointcloud_normal')
    for fname in os.listdir(object_folder_path):
        if not fname.endswith(".pt"):
            continue
        object_name = os.path.splitext(fname)[0]
        object_path = os.path.join(object_folder_path, fname)
        obj = torch.load(object_path)['pcl'].to(device)  # (M, 6)
        cache[object_name] = (obj[:, :3], obj[:, 3:])
    return cache

def calculate_depth_new(robot_pc, object_names, cache):
    """
    Calculate the average penetration depth of predicted pc into the object.

    :param robot_pc: (B, N, 3)
    :param object_names: list<str>, len = B
    :return: scalar depth loss
    """
    object_pc_list, normals_list = [], []
    for object_name in object_names:
        pc, normal = cache[object_name]   # each (M, 3)
        object_pc_list.append(pc)
        normals_list.append(normal)

    object_pc = torch.stack(object_pc_list, dim=0)   # (B, M, 3)
    normals   = torch.stack(normals_list, dim=0)     # (B, M, 3)

    # nearest neighbor distance
    distance = torch.cdist(robot_pc, object_pc)      # (B, N, M)
    distance, index = distance.min(dim=-1)           # (B, N)

    index_exp = index.unsqueeze(-1).expand(-1, -1, 3)  # (B, N, 3)
    object_pc_nn = torch.gather(object_pc, 1, index_exp)  # (B, N, 3)
    normals_nn   = torch.gather(normals,   1, index_exp)  # (B, N, 3)

    # --- vectorized sign check (replaces vmap) ---
    diff = robot_pc - object_pc_nn                  # (B, N, 3)
    dot = (diff * normals_nn).sum(dim=-1)           # (B, N)
    sign = torch.where(dot >= 0, 1.0, -1.0)         # (B, N)

    signed_distance = distance * sign               # (B, N)
    signed_distance = torch.where(signed_distance > 0, 0.0, signed_distance)

    return -signed_distance.mean()

def calculate_depth(robot_pc, object_names, cache):
    """
    Calculate the average penetration depth of predicted pc into the object.

    :param robot_pc: (B, N, 3)
    :param object_name: list<str>, len = B
    :return: calculated depth, (B,)
    """
    object_pc_list = []
    normals_list = []
    for object_name in object_names:
        # object_path = os.path.join(ROOT_DIR, f'data/pointcloud_normal/{object_name}.pt')
        pc, normal = cache[object_name]
        object_pc_list.append(pc)
        normals_list.append(normal)


    object_pc = torch.stack(object_pc_list, dim=0)
    normals = torch.stack(normals_list, dim=0)

    distance = torch.cdist(robot_pc, object_pc)
    distance, index = torch.min(distance, dim=-1)
    index = index.unsqueeze(-1).repeat(1, 1, 3)
    object_pc_indexed = torch.gather(object_pc, dim=1, index=index)
    normals_indexed = torch.gather(normals, dim=1, index=index)
    get_sign = torch.vmap(torch.vmap(lambda x, y: torch.where(torch.dot(x, y) >= 0, 1, -1)))
    signed_distance = distance * get_sign(robot_pc - object_pc_indexed, normals_indexed)
    signed_distance[signed_distance > 0] = 0


    
    return -torch.mean(signed_distance)


def farthest_point_sampling(point_cloud, num_points=1024):
    """
    :param point_cloud: (N, 3) or (N, 4), point cloud (with link index)
    :param num_points: int, number of sampled points
    :return: ((N, 3) or (N, 4), list), sampled point cloud (numpy) & index
    """
    point_cloud_origin = point_cloud
    if point_cloud.shape[1] == 4:
        point_cloud = point_cloud[:, :3]

    selected_indices = [0]
    distances = torch.norm(point_cloud - point_cloud[selected_indices[-1]], dim=1)
    for _ in range(num_points - 1):
        farthest_point_idx = torch.argmax(distances)
        selected_indices.append(farthest_point_idx)
        new_distances = torch.norm(point_cloud - point_cloud[farthest_point_idx], dim=1)
        distances = torch.min(distances, new_distances)
    sampled_point_cloud = point_cloud_origin[selected_indices]

    return sampled_point_cloud, selected_indices

