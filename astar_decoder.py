import numpy as np
import networkx as nx
import graph_extraction
from graph_utils import convert_to_sat2graph_format

class SamroadConfig:
    def __init__(self, v_thr=0.08, e_thr=0.09, snap_dist=30.0):
        self.ITSC_THRESHOLD = v_thr
        self.ITSC_NMS_RADIUS = 4.0
        self.ROAD_THRESHOLD = e_thr
        self.ROAD_NMS_RADIUS = 4.0
        self.NEIGHBOR_RADIUS = snap_dist

def DecodeAstar(fused_gte, v_thr=0.08, e_thr=0.09, snap_dist=200.0):
    """
    Bypasses the physics engine entirely and traces roads directly using A* over
    the raw probability tensor.
    
    Args:
        fused_gte: [H, W, 50] probability tensor from the neural network.
        v_thr: threshold for vertex probabilities
        e_thr: threshold for edge probabilities
        snap_dist: maximum pathfinding radius for A*
    
    Returns:
        dict_graph: Sat2Graph compatible dictionary graph format.
    """
    config = SamroadConfig(v_thr=v_thr, e_thr=e_thr, snap_dist=snap_dist)
    
    # Extract the vertexness mask (channel 0)
    keypoint_mask = (fused_gte[:, :, 0] * 255).astype(np.uint8)
    
    # Extract the edgeness mask by pooling the 6 directional channels
    # Channels 2, 6, 10, 14, 18, 22 are the outgoing edge probabilities
    edge_prob = np.zeros_like(fused_gte[:, :, 0])
    for j in range(6):
        edge_prob = np.maximum(edge_prob, fused_gte[:, :, 2 + 4 * j])
        
    road_mask = (np.maximum(fused_gte[:, :, 0], edge_prob) * 255).astype(np.uint8)
    
    # Pathfind the graph using samroadplus's SOTA algorithm!
    # Returns a NetworkX graph with node format (x, y)
    nx_graph = graph_extraction.extract_graph_astar(keypoint_mask, road_mask, config)
    
    # Convert NetworkX (x, y) to nodes array [N, 2] in (row, col) and edges array [E, 2]
    nodes = []
    node_to_idx = {}
    for node in nx_graph.nodes():
        node_to_idx[node] = len(node_to_idx)
        nodes.append((node[1], node[0])) # Convert (x, y) to (r, c)
        
    nodes = np.array(nodes)
    
    edges = []
    for u, v in nx_graph.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))
    edges = np.array(edges)
    
    if len(nodes) == 0:
        return {}
        
    # Convert back to Sat2Graph dictionary format so the APLS evaluator still works!
    dict_graph = convert_to_sat2graph_format(nodes, edges)
    return dict_graph
