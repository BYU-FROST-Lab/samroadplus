"""
SAMGraph Dataset: Unified data pipeline combining samroadplus's image/mask loading
with Sat2Graph's GTE (Graph Tensor Encoding) label generation.

Returns per sample:
    - rgb: [H, W, 3] satellite image patch (0-255 float)
    - keypoint_mask: [H, W] keypoint mask for SAM seg supervision (0-1)
    - road_mask: [H, W] road mask for SAM seg supervision (0-1)
    - target_prob: [H, W, 14] GTE vertex/edge probabilities
    - target_vector: [H, W, 12] GTE direction vectors
    - gt_seg: [H, W, 1] binary road segmentation for GTE seg loss (-0.5 to 0.5)
    - graph_points: [N_points, 2] candidate node coords (if TOPONET_ENABLED)
    - pairs: [N_samples, N_pairs, 2] node pair indices (if TOPONET_ENABLED)
    - connected: [N_samples, N_pairs] BFS ground truth (if TOPONET_ENABLED)
    - valid: [N_samples, N_pairs] padding mask (if TOPONET_ENABLED)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
import pickle
import os
import json
import scipy
import rtree
import graph_utils


# ============================================================
# Constants
# ============================================================
MAX_DEGREE = 6
VECTOR_NORM = 25.0


# ============================================================
# Image I/O
# ============================================================
def read_rgb_img(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ============================================================
# Data partitioning (from samroadplus)
# ============================================================
def cityscale_data_partition():
    indrange_train = []
    indrange_test = []
    indrange_validation = []
    for x in range(180):
        if x % 10 < 8:
            indrange_train.append(x)
        if x % 10 == 9:
            indrange_test.append(x)
        if x % 20 == 18:
            indrange_validation.append(x)
        if x % 20 == 8:
            indrange_test.append(x)
    return indrange_train, indrange_validation, indrange_test


def globalscale_data_partition():
    indrange_train = []
    indrange_test = []
    indrange_test_out_domain = []
    indrange_validation = []
    for x in range(2375):
        indrange_train.append(x)
    for x in range(2375, 2714):
        indrange_validation.append(x)
    for x in range(2714, 3338):
        indrange_test.append(x)
    for x in range(130):
        indrange_test_out_domain.append(x)
    return indrange_train, indrange_validation, indrange_test, indrange_test_out_domain


def spacenet_data_partition():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "spacenet")
    JSON_DATA = os.path.join(DATA_DIR, "data_split.json")
    with open(JSON_DATA, 'r') as jf:
        data_list = json.load(jf)
    return data_list['train'], data_list['validation'], data_list['test']


# ============================================================
# Sat2Graph neighbor utilities (from Sat2Graph dataloader)
# ============================================================
def neighbor_to_integer(n_in):
    """Convert neighbor dict keys/values to integer tuples."""
    n_out = {}
    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))
        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []
        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))
            if new_n_k not in nv:
                nv.append(new_n_k)
        n_out[nk] = nv
    return n_out


# ============================================================
# GTE label generation (ported from Sat2Graph dataloader)
# ============================================================
def generate_gte_labels(neighbors, patch_size, max_degree=6, vector_norm=25.0,
                        offset_x=0, offset_y=0, dataset_image_size=2048):
    """
    Generate GTE (Graph Tensor Encoding) ground truth labels from a neighbor graph.
    
    This is the core of Sat2Graph's dense graph representation — every pixel gets:
    - 2 channels for vertex presence (present/absent probabilities)
    - Per direction (6 directions): 2 channels for edge presence + 2 channels for direction vector
    
    Args:
        neighbors: dict mapping (row, col) → list of [(row, col)] neighbor nodes
        patch_size: size of the output patch (H = W = patch_size)
        max_degree: number of angular bins (default 6 → 60° each)
        vector_norm: normalization factor for direction vectors
        offset_x, offset_y: crop offset within the full tile
        dataset_image_size: full tile size
    
    Returns:
        target_prob: [patch_size, patch_size, 2*(max_degree+1)] = [H, W, 14]
        target_vector: [patch_size, patch_size, 2*max_degree] = [H, W, 12]
    """
    target_prob = np.zeros((patch_size, patch_size, 2 * (max_degree + 1)), dtype=np.float32)
    target_vector = np.zeros((patch_size, patch_size, 2 * max_degree), dtype=np.float32)
    
    # Default: all "absent" probabilities set to 1
    target_prob[:, :, 1::2] = 1.0
    
    r = 1  # radius for label spreading (3×3 neighborhood)
    
    for loc, n_locs in neighbors.items():
        # Convert from full-tile coordinates to patch coordinates
        px = loc[0] - offset_y  # row in patch
        py = loc[1] - offset_x  # col in patch
        
        # Skip nodes outside this patch (with border margin)
        if px < 0 or py < 0 or px >= patch_size or py >= patch_size:
            continue
        
        # Skip nodes near tile borders
        if loc[0] < 16 or loc[1] < 16 or loc[0] > dataset_image_size - 16 or loc[1] > dataset_image_size - 16:
            continue
        
        # Set vertex presence: channel 0 = present, channel 1 = absent
        for x in range(max(0, px - r), min(patch_size, px + r + 1)):
            for y in range(max(0, py - r), min(patch_size, py + r + 1)):
                target_prob[x, y, 0] = 1.0
                target_prob[x, y, 1] = 0.0
        
        # Process each neighbor edge
        for n_loc in n_locs:
            if n_loc[0] < 16 or n_loc[1] < 16 or n_loc[0] > dataset_image_size - 16 or n_loc[1] > dataset_image_size - 16:
                continue
            
            # Determine angular bin (6 bins of 60° each)
            d = math.atan2(n_loc[1] - loc[1], n_loc[0] - loc[0]) + math.pi
            j = int(d / (math.pi / 3.0)) % max_degree
            
            for x in range(max(0, px - r), min(patch_size, px + r + 1)):
                for y in range(max(0, py - r), min(patch_size, py + r + 1)):
                    # Edge presence for direction j
                    target_prob[x, y, 2 + 2*j] = 1.0
                    target_prob[x, y, 2 + 2*j + 1] = 0.0
                    
                    # Direction vector (normalized)
                    target_vector[x, y, 2*j] = (n_loc[0] - loc[0]) / vector_norm
                    target_vector[x, y, 2*j + 1] = (n_loc[1] - loc[1]) / vector_norm
    
    return target_prob, target_vector


# ============================================================
# GraphLabelGenerator: BFS-based edge ground truth for TopoNet
# (ported from samroadplus/dataset.py)
# ============================================================
class GraphLabelGenerator():
    """Generates graph-based training labels for TopoNet by running BFS on the
    ground truth graph to determine which candidate node pairs are connected."""
    
    def __init__(self, config, full_graph, coord_transform):
        self.config = config
        # convert to igraph for high performance
        self.full_graph_origin = graph_utils.igraph_from_adj_dict(full_graph, coord_transform)
        # find crossover points, we'll avoid predicting these as keypoints
        self.crossover_points = graph_utils.find_crossover_points(self.full_graph_origin)
        self.subdivide_resolution = 4
        self.full_graph_subdivide = graph_utils.subdivide_graph(self.full_graph_origin, self.subdivide_resolution)
        self.subdivide_points = np.array(self.full_graph_subdivide.vs['point'])
        # pre-build spatial index
        self.graph_rtee = rtree.index.Index()
        for i, v in enumerate(self.subdivide_points):
            x, y = v
            self.graph_rtee.insert(i, (x, y, x, y))
        self.graph_kdtree = scipy.spatial.KDTree(self.subdivide_points)

        # pre-exclude points near crossover points
        crossover_exclude_radius = 4
        exclude_indices = set()
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(p, crossover_exclude_radius)
            exclude_indices.update(nearby_indices)
        self.exclude_indices = exclude_indices

        # Find intersection points, these will always be kept in nms
        itsc_indices = set()
        point_num = len(self.full_graph_subdivide.vs)
        for i in range(point_num):
            if self.full_graph_subdivide.degree(i) != 2:
                itsc_indices.add(i)
        self.nms_score_override = np.zeros((point_num,), dtype=np.float32)
        if len(itsc_indices) > 0:
            self.nms_score_override[np.array(list(itsc_indices))] = 2.0

        # Points near crossover and intersections are sampled more frequently
        interesting_indices = set()
        interesting_radius = 32
        for i in itsc_indices:
            p = self.subdivide_points[i]
            nearby_indices = self.graph_kdtree.query_ball_point(p, interesting_radius)
            interesting_indices.update(nearby_indices)
        for p in self.crossover_points:
            nearby_indices = self.graph_kdtree.query_ball_point(np.array(p), interesting_radius)
            interesting_indices.update(nearby_indices)
        self.sample_weights = np.full((point_num,), 0.1, dtype=np.float32)
        if len(interesting_indices) > 0:
            self.sample_weights[list(interesting_indices)] = 0.9

    def sample_patch(self, patch, rot_index=0):
        (x0, y0), (x1, y1) = patch
        query_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        patch_indices_all = set(self.graph_rtee.intersection(query_box))
        patch_indices = patch_indices_all - self.exclude_indices
        patch_indices = np.array(list(patch_indices))
        sample_num = getattr(self.config, 'TOPO_SAMPLE_NUM', 512)
        max_nbr_queries = getattr(self.config, 'MAX_NEIGHBOR_QUERIES', 16)
        if len(patch_indices) == 0:
            fake_points = np.array([[0.0, 0.0]], dtype=np.float32)
            fake_sample = ([[0, 0]] * max_nbr_queries, [False] * max_nbr_queries, [False] * max_nbr_queries)
            return fake_points, [fake_sample] * sample_num
        patch_points = self.subdivide_points[patch_indices, :]
        nms_scores = np.random.uniform(low=0.9, high=1.0, size=patch_indices.shape[0])
        nms_score_override = self.nms_score_override[patch_indices]
        nms_scores = np.maximum(nms_scores, nms_score_override)
        nms_radius = getattr(self.config, 'ROAD_NMS_RADIUS', 16)
        nmsed_points, kept_indices = graph_utils.nms_points(patch_points, nms_scores, radius=nms_radius, return_indices=True)
        nmsed_indices = patch_indices[kept_indices]
        nmsed_point_num = nmsed_points.shape[0]

        sample_weights = self.sample_weights[nmsed_indices]
        sample_indices_in_nmsed = np.random.choice(
            np.arange(start=0, stop=nmsed_points.shape[0], dtype=np.int32),
            size=sample_num, replace=True, p=sample_weights / np.sum(sample_weights))
        sample_indices = nmsed_indices[sample_indices_in_nmsed]

        radius = getattr(self.config, 'NEIGHBOR_RADIUS', 64)
        nmsed_kdtree = scipy.spatial.KDTree(nmsed_points)
        sampled_points = self.subdivide_points[sample_indices, :]
        knn_d, knn_idx = nmsed_kdtree.query(sampled_points, k=max_nbr_queries + 1, distance_upper_bound=radius)

        samples = []
        for i in range(sample_num):
            source_node = sample_indices[i]
            valid_nbr_indices = knn_idx[i, knn_idx[i, :] < nmsed_point_num]
            valid_nbr_indices = valid_nbr_indices[1:]  # remove self
            target_nodes = [nmsed_indices[ni] for ni in valid_nbr_indices]
            reached_nodes = graph_utils.bfs_with_conditions(
                self.full_graph_subdivide, source_node, set(target_nodes),
                radius // self.subdivide_resolution)
            shall_connect = [t in reached_nodes for t in target_nodes]
            pairs = []
            valid = []
            source_nmsed_idx = sample_indices_in_nmsed[i]
            for target_nmsed_idx in valid_nbr_indices:
                pairs.append((source_nmsed_idx, target_nmsed_idx))
                valid.append(True)
            for j in range(len(pairs), max_nbr_queries):
                pairs.append((source_nmsed_idx, source_nmsed_idx))
                shall_connect.append(False)
                valid.append(False)
            samples.append((pairs, shall_connect, valid))
        # Transform points into patch-local coordinates
        nmsed_points -= np.array([x0, y0])[np.newaxis, :]
        # Apply rotation
        nmsed_points = np.concatenate([nmsed_points, np.ones((nmsed_point_num, 1), dtype=nmsed_points.dtype)], axis=1)
        patch_size = getattr(self.config, 'PATCH_SIZE', 512)
        trans = np.array([
            [1, 0, -0.5 * patch_size],
            [0, 1, -0.5 * patch_size],
            [0, 0, 1],
        ], dtype=np.float32)
        rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        nmsed_points = nmsed_points @ trans.T @ np.linalg.matrix_power(rot.T, rot_index) @ np.linalg.inv(trans.T)
        nmsed_points = nmsed_points[:, :2]
        return nmsed_points, samples


# ============================================================
# Patch info for evaluation
# ============================================================
def get_patch_info_one_img(image_index, image_size, sample_margin, patch_size, patches_per_edge):
    patch_info = []
    sample_min = sample_margin
    sample_max = image_size - (patch_size + sample_margin)
    eval_samples = np.linspace(start=sample_min, stop=sample_max, num=patches_per_edge)
    eval_samples = [round(x) for x in eval_samples]
    for x in eval_samples:
        for y in eval_samples:
            patch_info.append(
                (image_index, (x, y), (x + patch_size, y + patch_size))
            )
    return patch_info


# ============================================================
# Collate function (simpler than samroadplus — no variable-length graph points)
# ============================================================
def samgraph_collate_fn(batch):
    """Collate with zero-padding for variable-length graph_points."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key == 'graph_points':
            # Variable-length: pad to max length in batch
            tensors = [item[key] for item in batch]
            max_point_num = max([x.shape[0] for x in tensors])
            padded = []
            for x in tensors:
                pad_num = max_point_num - x.shape[0]
                padded_x = torch.concat([x, torch.zeros(pad_num, 2)], dim=0)
                padded.append(padded_x)
            collated[key] = torch.stack(padded, dim=0)
        else:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated


# ============================================================
# Main Dataset
# ============================================================
class SatMapDataset(Dataset):
    """
    Unified dataset producing both SAM seg targets and GTE graph targets.
    
    Supports cityscale and spacenet datasets (matching samroadplus's SatMapDataset)
    with added GTE label generation from the same _refine_gt_graph.p files.
    """
    
    def __init__(self, config, is_train, dev_run=False):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        self.config = config
        self.is_train = is_train
        self.max_degree = getattr(config, 'MAX_DEGREE', MAX_DEGREE)
        self.vector_norm = getattr(config, 'VECTOR_NORM', VECTOR_NORM)
        
        assert self.config.DATASET in {'cityscale', 'spacenet', 'globalscale'}
        
        # Initialize memory structures
        self.rgbs = []
        self.keypoint_masks = []
        self.road_masks = []
        self.gt_segs = []
        self.neighbor_graphs = []
        self.graph_label_generators = []
        self.samples = [] # Used for GlobalScale lazy loading
        
        self.toponet_enabled = getattr(config, 'TOPONET_ENABLED', False)
        self.coord_transform = lambda v: v[:, ::-1]
        
        if self.config.DATASET == 'cityscale':
            DIR_CS = os.path.join(BASE_DIR, self.config.DATASET)
            DATA_DIR_CS = os.path.join(DIR_CS, "20cities")
            PROCESSED_DIR = os.path.join(DIR_CS, "processed")
            
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            self.rgb_pattern = os.path.join(DATA_DIR_CS, 'region_{}_sat.png')
            self.keypoint_mask_pattern = os.path.join(PROCESSED_DIR, 'keypoint_mask_{}.png')
            self.road_mask_pattern = os.path.join(PROCESSED_DIR, 'road_mask_{}.png')
            self.gt_graph_pattern = os.path.join(DATA_DIR_CS, 'region_{}_refine_gt_graph.p')
            self.gt_seg_pattern = os.path.join(DATA_DIR_CS, 'region_{}_gt.png')
            
            train, val, test = cityscale_data_partition()
            train_split = train + val
            test_split = test
            self.tile_indices = train_split if self.is_train else test_split
            self.trainnum = train
            
        elif self.config.DATASET == 'spacenet':
            DIR_SN = os.path.join(BASE_DIR, self.config.DATASET)
            DATA_DIR_SN = os.path.join(DIR_SN, "RGB_1.0_meter")
            PROCESSED_DIR = os.path.join(DIR_SN, "processed")
            
            self.IMAGE_SIZE = 400
            self.SAMPLE_MARGIN = 0
            self.rgb_pattern = os.path.join(DATA_DIR_SN, '{}__rgb.png')
            self.keypoint_mask_pattern = os.path.join(PROCESSED_DIR, 'keypoint_mask_{}.png')
            self.road_mask_pattern = os.path.join(PROCESSED_DIR, 'road_mask_{}.png')
            self.gt_graph_pattern = os.path.join(DATA_DIR_SN, '{}__gt_graph.p')
            self.gt_seg_pattern = None  # Spacenet may not have separate gt seg
            
            train, val, test = spacenet_data_partition()
            train_split = train + val
            test_split = test
            self.tile_indices = train_split if self.is_train else test_split
            self.trainnum = train
            
        elif self.config.DATASET == 'globalscale':
            DIR_GS = os.path.join(BASE_DIR, '..', 'samroadplus', 'globalscale')
            DATA_DIR = os.path.join(DIR_GS, "Global-Scale")
            PROCESSED_DIR = os.path.join(DIR_GS, "processed")
            
            self.IMAGE_SIZE = 2048
            self.SAMPLE_MARGIN = 64
            
            GLOBALSCALE_DIRS = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
            patterns = {}
            for DIR in GLOBALSCALE_DIRS:
                DATA_DIR_REC = os.path.join(DATA_DIR, DIR)
                PROCESSED_DIR_REC = os.path.join(PROCESSED_DIR, DIR)
                patterns[DIR] = {
                    "rgb_pattern": os.path.join(DATA_DIR_REC, 'region_{}_sat.png'),
                    "keypoint_mask_pattern": os.path.join(PROCESSED_DIR_REC, 'keypoint_mask_{}.png'),
                    "road_mask_pattern": os.path.join(PROCESSED_DIR_REC, 'road_mask_{}.png'),
                    "gt_graph_pattern": os.path.join(DATA_DIR_REC, 'region_{}_refine_gt_graph.p')
                }
            
            train_split = ["train", "val"]
            test_split = ["in-domain-test"]
            folders = train_split if self.is_train else test_split
            self.trainnum = train_split
            
            if dev_run:
                # Limit folders for dev run to speed up __init__
                folders = [folders[0]]

            for folder in folders:
                DATA_DIR_REC = os.path.join(DATA_DIR, folder)
                files = sorted(
                    [f for f in os.listdir(DATA_DIR_REC) if f.endswith("_refine_gt_graph.p")],
                    key=lambda x: int(x.split("_")[1])
                )
                if dev_run:
                    files = files[:4]
                
                for fname in files:
                    tile_idx = fname.split("_")[1]
                    gt_graph_path = patterns[folder]["gt_graph_pattern"].format(tile_idx)
                    
                    gt_graph_adj = pickle.load(open(gt_graph_path, 'rb'))
                    if len(gt_graph_adj) == 0:
                        print(f'===== skipped empty tile {tile_idx} =====')
                        continue
                        
                    self.samples.append({
                        "tile_idx": tile_idx,
                        "folder": folder,
                        "rgb": patterns[folder]["rgb_pattern"].format(tile_idx),
                        "keypoint_mask": patterns[folder]["keypoint_mask_pattern"].format(tile_idx),
                        "road_mask": patterns[folder]["road_mask_pattern"].format(tile_idx),
                        "gt_seg": patterns[folder]["road_mask_pattern"].format(tile_idx), # Fallback to road mask
                        "gt_graph_path": gt_graph_path
                    })
                        
            # Set tile_indices to empty since globalscale uses self.samples
            self.tile_indices = []
            
        if self.config.DATASET != 'globalscale':
            if dev_run:
                self.tile_indices = self.tile_indices[:4]
        
        for tile_idx in self.tile_indices:
            print(f'loading tile {tile_idx}')
            
            rgb_path = self.rgb_pattern.format(tile_idx)
            road_mask_path = self.road_mask_pattern.format(tile_idx)
            keypoint_mask_path = self.keypoint_mask_pattern.format(tile_idx)
            gt_graph_path = self.gt_graph_pattern.format(tile_idx)
            
            # Load graph and check if valid
            gt_graph_adj = pickle.load(open(gt_graph_path, 'rb'))
            if len(gt_graph_adj) == 0:
                print(f'===== skipped empty tile {tile_idx} =====')
                continue
            
            self.rgbs.append(read_rgb_img(rgb_path))
            self.road_masks.append(cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE))
            self.keypoint_masks.append(cv2.imread(keypoint_mask_path, cv2.IMREAD_GRAYSCALE))
            
            # Store neighbor graph for GTE label generation
            neighbors = neighbor_to_integer(gt_graph_adj)
            self.neighbor_graphs.append(neighbors)
            
            # Build GraphLabelGenerator for TopoNet (if enabled)
            if self.toponet_enabled and len(gt_graph_adj) > 0:
                glg = GraphLabelGenerator(config, gt_graph_adj, self.coord_transform)
                self.graph_label_generators.append(glg)
            else:
                self.graph_label_generators.append(None)
            
            # Load ground truth segmentation for GTE seg loss
            if self.gt_seg_pattern is not None:
                gt_seg_path = self.gt_seg_pattern.format(tile_idx)
                if os.path.exists(gt_seg_path):
                    self.gt_segs.append(cv2.imread(gt_seg_path, cv2.IMREAD_GRAYSCALE))
                else:
                    # Fallback: use road_mask as seg GT
                    self.gt_segs.append(self.road_masks[-1].copy())
            else:
                # Fallback: use road_mask as seg GT
                self.gt_segs.append(self.road_masks[-1].copy())
        
        self.sample_min = self.SAMPLE_MARGIN
        self.sample_max = self.IMAGE_SIZE - (self.config.PATCH_SIZE + self.SAMPLE_MARGIN)
        
        if not self.is_train:
            eval_patches_per_edge = math.ceil(
                (self.IMAGE_SIZE - 2 * self.SAMPLE_MARGIN) / self.config.PATCH_SIZE
            )
            self.eval_patches = []
            num_images = len(self.samples) if self.config.DATASET == 'globalscale' else len(self.rgbs)
            for i in range(num_images):
                self.eval_patches += get_patch_info_one_img(
                    i, self.IMAGE_SIZE, self.SAMPLE_MARGIN,
                    self.config.PATCH_SIZE, eval_patches_per_edge
                )
    
    def __len__(self):
        if self.is_train:
            num_patches_per_image = max(1, int(self.IMAGE_SIZE / self.config.PATCH_SIZE)) ** 2
            num_images = len(self.samples) if self.config.DATASET == 'globalscale' else len(self.trainnum)
            return num_images * num_patches_per_image
        else:
            return len(self.eval_patches)
    
    def __getitem__(self, idx):
        if self.is_train:
            num_images = len(self.samples) if self.config.DATASET == 'globalscale' else len(self.rgbs)
            img_idx = np.random.randint(low=0, high=num_images)
            begin_x = np.random.randint(low=self.sample_min, high=max(self.sample_min + 1, self.sample_max + 1))
            begin_y = np.random.randint(low=self.sample_min, high=max(self.sample_min + 1, self.sample_max + 1))
            end_x = begin_x + self.config.PATCH_SIZE
            end_y = begin_y + self.config.PATCH_SIZE
        else:
            img_idx, (begin_x, begin_y), (end_x, end_y) = self.eval_patches[idx]
        
        # Crop patches
        if self.config.DATASET == 'globalscale':
            sample = self.samples[img_idx]
            rgb_full = read_rgb_img(sample['rgb'])
            keypoint_full = cv2.imread(sample['keypoint_mask'], cv2.IMREAD_GRAYSCALE)
            road_full = cv2.imread(sample['road_mask'], cv2.IMREAD_GRAYSCALE)
            gt_seg_full = road_full.copy() # Fallback for now
            
            rgb_patch = rgb_full[begin_y:end_y, begin_x:end_x, :]
            keypoint_mask_patch = keypoint_full[begin_y:end_y, begin_x:end_x]
            road_mask_patch = road_full[begin_y:end_y, begin_x:end_x]
            gt_seg_patch = gt_seg_full[begin_y:end_y, begin_x:end_x]
        else:
            rgb_patch = self.rgbs[img_idx][begin_y:end_y, begin_x:end_x, :]
            keypoint_mask_patch = self.keypoint_masks[img_idx][begin_y:end_y, begin_x:end_x]
            road_mask_patch = self.road_masks[img_idx][begin_y:end_y, begin_x:end_x]
            gt_seg_patch = self.gt_segs[img_idx][begin_y:end_y, begin_x:end_x]
        
        # Augmentation: random 90° rotation
        rot_index = 0
        if self.is_train:
            rot_index = np.random.randint(0, 4)
            if rot_index > 0:
                rgb_patch = np.rot90(rgb_patch, rot_index, [0, 1]).copy()
                keypoint_mask_patch = np.rot90(keypoint_mask_patch, rot_index, [0, 1]).copy()
                road_mask_patch = np.rot90(road_mask_patch, rot_index, [0, 1]).copy()
                gt_seg_patch = np.rot90(gt_seg_patch, rot_index, [0, 1]).copy()
        
        # Generate GTE labels from neighbor graph
        if self.config.DATASET == 'globalscale':
            gt_graph_path = self.samples[img_idx]["gt_graph_path"]
            gt_graph_adj = pickle.load(open(gt_graph_path, 'rb'))
            neighbors = neighbor_to_integer(gt_graph_adj)
        else:
            neighbors = self.neighbor_graphs[img_idx]
            
        target_prob, target_vector = generate_gte_labels(
            neighbors,
            patch_size=self.config.PATCH_SIZE,
            max_degree=self.max_degree,
            vector_norm=self.vector_norm,
            offset_x=begin_x,
            offset_y=begin_y,
            dataset_image_size=self.IMAGE_SIZE,
        )
        
        # Apply same rotation to GTE labels
        if rot_index > 0:
            target_prob = np.rot90(target_prob, rot_index, [0, 1]).copy()
            target_vector = np.rot90(target_vector, rot_index, [0, 1]).copy()
            # TODO: When rotating GTE labels, the direction bin assignments
            # should also be shifted. For exact correctness, bins should be
            # cyclically shifted by rot_index * (max_degree / 4). This is a
            # simplification that may need refinement for best accuracy.
        
        # Convert gt_seg to Sat2Graph's [-0.5, 0.5] range
        gt_seg_normalized = (gt_seg_patch.astype(np.float32) / 255.0) - 0.5  # [H, W]
        
        result = {
            'rgb': torch.tensor(rgb_patch, dtype=torch.float32),
            'keypoint_mask': torch.tensor(keypoint_mask_patch, dtype=torch.float32) / 255.0,
            'road_mask': torch.tensor(road_mask_patch, dtype=torch.float32) / 255.0,
            'target_prob': torch.tensor(target_prob, dtype=torch.float32),
            'target_vector': torch.tensor(target_vector, dtype=torch.float32),
            'gt_seg': torch.tensor(gt_seg_normalized, dtype=torch.float32).unsqueeze(-1),  # [H, W, 1]
        }
        
        # Generate TopoNet labels (if enabled)
        if self.toponet_enabled:
            glg = None
            if self.config.DATASET == 'globalscale':
                glg = GraphLabelGenerator(self.config, gt_graph_adj, self.coord_transform)
            else:
                glg = self.graph_label_generators[img_idx]
                
            if glg is not None:
                patch = ((begin_x, begin_y), (begin_x + self.config.PATCH_SIZE, begin_y + self.config.PATCH_SIZE))
                graph_points, topo_samples = glg.sample_patch(patch, rot_index)
                pairs, connected, valid = zip(*topo_samples)
                result['graph_points'] = torch.tensor(graph_points, dtype=torch.float32)
                result['pairs'] = torch.tensor(pairs, dtype=torch.int32)
                result['connected'] = torch.tensor(connected, dtype=torch.bool)
                result['valid'] = torch.tensor(valid, dtype=torch.bool)
            else:
                # Fallback for empty tiles: produce zero-filled topo tensors
                sample_num = getattr(self.config, 'TOPO_SAMPLE_NUM', 512)
                max_nbr = getattr(self.config, 'MAX_NEIGHBOR_QUERIES', 16)
                result['graph_points'] = torch.zeros((1, 2), dtype=torch.float32)
                result['pairs'] = torch.zeros((sample_num, max_nbr, 2), dtype=torch.int32)
                result['connected'] = torch.zeros((sample_num, max_nbr), dtype=torch.bool)
                result['valid'] = torch.zeros((sample_num, max_nbr), dtype=torch.bool)
        
        return result
