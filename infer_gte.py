"""
SAMGraph Inference Script — Cityscale

Runs sliding-window inference on cityscale test tiles, decodes the GTE output
into road graphs, and saves them in Sat2Graph pickle format for APLS/TOPO
evaluation.

Usage:
    python infer_samgraph.py \
        --config config/samgraph_cityscale.yaml \
        --checkpoint samgraph/my9a8of2/checkpoints/epoch=9-step=5760.ckpt \
        --output_dir output_cityscale
"""

import os
import sys
import pickle
import time
import numpy as np
import cv2
import torch
from argparse import ArgumentParser

from utils import load_config
from dataset_gte import read_rgb_img, cityscale_data_partition, get_patch_info_one_img
from model_gte import SAMGraph, gte_softmax_output

from gte_decoder import DecodeAndVis

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument("--config", required=True, help="YAML config file")
parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt path")
parser.add_argument("--output_dir", default="output_cityscale",
                    help="Directory for output graphs and visualizations")
parser.add_argument("--device", default="cuda")
parser.add_argument("--v_thr", type=float, default=0.08,
                    help="Vertex threshold for DecodeAndVis")
parser.add_argument("--e_thr", type=float, default=0.09,
                    help="Edge threshold for DecodeAndVis")
parser.add_argument("--snap_dist", type=float, default=30.0,
                    help="Magnetic snap distance limit for decoding")
parser.add_argument("--angle_weight", type=float, default=12.0,
                    help="Penalty weight applied to curved traces")
parser.add_argument("--fast_dev_run", action="store_true",
                    help="Process only the first test tile for quick testing")
parser.add_argument("--decoder", type=str, default="sat2graph",
                    choices=["sat2graph", "samroad"],
                    help="Graph extraction algorithm to use")
parser.add_argument("--bridge_mode", type=str, default="none",
                    choices=["none", "vanilla", "geometric", "astar"],
                    help="Algorithm to use for stitching disconnected components")
parser.add_argument("--bridge_dist", type=float, default=200.0,
                    help="Max distance to bridge")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def crop_patch(img, x0, y0, x1, y1):
    """Crop a patch from an image.  img: [H, W, C]."""
    return img[y0:y1, x0:x1, :]


def infer_one_img(net, img, config, device):
    """
    Run sliding-window inference on a single 2048×2048 satellite image.

    Returns:
        fused_gte: np.ndarray [H, W, C_gte]  (channels-last, after softmax)
    """
    image_size = img.shape[0]  # assume square
    patch_size = config.PATCH_SIZE
    sample_margin = config.SAMPLE_MARGIN
    batch_size = config.INFER_BATCH_SIZE
    patches_per_edge = getattr(config, 'INFER_PATCHES_PER_EDGE', 16)

    # All patch coordinates: list of (idx, (x0,y0), (x1,y1))
    all_patch_info = get_patch_info_one_img(
        0, image_size, sample_margin, patch_size, patches_per_edge)
    patch_num = len(all_patch_info)
    batch_num = (patch_num + batch_size - 1) // batch_size

    # GTE output has 26 raw channels -> gte_softmax_output -> 50 channels
    gte_raw_ch = 2 + 4 * config.MAX_DEGREE + (2 if config.JOINT_WITH_SEG else 0)
    # After softmax conversion the channel count changes
    gte_soft_ch = 2 + 4 * config.MAX_DEGREE + (2 if config.JOINT_WITH_SEG else 0)

    fused_gte = np.zeros((image_size, image_size, gte_soft_ch), dtype=np.float32)
    pixel_count = np.zeros((image_size, image_size, 1), dtype=np.float32)

    for batch_idx in range(batch_num):
        offset = batch_idx * batch_size
        batch_info = all_patch_info[offset: offset + batch_size]

        # Build batch tensor [B, H, W, C]
        patches = []
        for _, (x0, y0), (x1, y1) in batch_info:
            patch = crop_patch(img, x0, y0, x1, y1)
            patches.append(torch.tensor(patch, dtype=torch.float32))
        batch_tensor = torch.stack(patches, 0).to(device, non_blocking=True)

        with torch.no_grad():
            # gte_output: [B, 26, H, W]
            gte_output, _mask_logits, _mask_scores = net(batch_tensor)
            # Convert to probabilities: [B, gte_soft_ch, H, W]
            gte_probs = gte_softmax_output(
                gte_output, max_degree=config.MAX_DEGREE,
                joint_with_seg=config.JOINT_WITH_SEG)
            # -> channels last [B, H, W, C]
            gte_probs = gte_probs.permute(0, 2, 3, 1).cpu().numpy()

        # Accumulate
        for pi, (_, (x0, y0), (x1, y1)) in enumerate(batch_info):
            fused_gte[y0:y1, x0:x1, :] += gte_probs[pi]
            pixel_count[y0:y1, x0:x1, :] += 1.0

    # Average overlapping regions
    pixel_count = np.maximum(pixel_count, 1.0)
    fused_gte /= pixel_count

    return fused_gte


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ------ Load model ------
    net = SAMGraph(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        net.load_state_dict(checkpoint, strict=True)
    net.eval()
    net.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ------ Data paths ------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if config.DATASET == 'globalscale':
        DATA_DIR = os.path.join(BASE_DIR, "..", "samroadplus", "globalscale", "Global-Scale", "in-domain-test")
        test_indices = sorted(
            [f.split("_")[1] for f in os.listdir(DATA_DIR) if f.endswith("_refine_gt_graph.p")],
            key=lambda x: int(x)
        )
    else:
        DATA_DIR = os.path.join(BASE_DIR, "cityscale", "20cities")
        _, _, test_indices = cityscale_data_partition()

    if args.fast_dev_run:
        test_indices = test_indices[:1]
        print(f"[fast_dev_run] Processing only tile {test_indices[0]}")

    # ------ Output dirs ------
    graph_dir = os.path.join(args.output_dir, "graph")
    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # ------ Inference loop ------
    total_time = 0.0
    vector_norm = getattr(config, 'VECTOR_NORM', 25.0)

    for tile_id in test_indices:
        rgb_path = os.path.join(DATA_DIR, f"region_{tile_id}_sat.png")
        if not os.path.exists(rgb_path):
            print(f"WARNING: {rgb_path} not found, skipping tile {tile_id}")
            continue

        print(f"Processing tile {tile_id} ...")
        img = read_rgb_img(rgb_path)  # [H, W, 3] uint8 RGB
        img_float = img.astype(np.float32)

        t0 = time.time()
        fused_gte = infer_one_img(net, img_float, config, device)
        infer_time = time.time() - t0
        total_time += infer_time
        print(f"  Inference: {infer_time:.1f}s")

        # ------ Decode graph ------
        image_size = img.shape[0]
        viz_prefix = os.path.join(viz_dir, str(tile_id))

        # Set vector_norm in decoder module
        import gte_decoder as dec_module
        dec_module.vector_norm = vector_norm

        t0 = time.time()
        if args.decoder == "samroad":
            import astar_decoder
            graph = astar_decoder.DecodeAstar(
                fused_gte, 
                v_thr=args.v_thr, 
                e_thr=args.e_thr, 
                snap_dist=args.snap_dist
            )
        else:
            graph = DecodeAndVis(
                fused_gte, viz_prefix,
                imagesize=image_size,
                max_degree=config.MAX_DEGREE,
                thr=args.v_thr,
                edge_thr=args.e_thr,
                snap=True,
                testing=True,
                angledistance_weight=args.angle_weight,
                snap_dist=args.snap_dist,
            )
        decode_time = time.time() - t0
        print(f"  Decode: {decode_time:.1f}s  |  nodes={len(graph)}")
        
        if args.bridge_mode != "none":
            import bridge_graphs
            t1 = time.time()
            if args.bridge_mode == "vanilla":
                graph = bridge_graphs.bridge_vanilla(graph, max_bridge_dist=args.bridge_dist)
            elif args.bridge_mode == "geometric":
                graph = bridge_graphs.bridge_geometric(graph, max_bridge_dist=args.bridge_dist)
            elif args.bridge_mode == "astar":
                graph = bridge_graphs.bridge_astar(graph, fused_gte, max_bridge_dist=args.bridge_dist)
            print(f"  Bridge: {time.time() - t1:.1f}s")

        # ------ Save graph pickle ------
        graph_path = os.path.join(graph_dir, f"{tile_id}.p")
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)

        # ------ Satellite overlay visualization ------
        overlay = img.copy()
        for k, v in graph.items():
            for n2 in v:
                cv2.line(overlay, (k[1], k[0]), (n2[1], n2[0]), (255, 255, 0), 3)
        for k in graph.keys():
            cv2.circle(overlay, (k[1], k[0]), 4, (255, 0, 0), -1)
        overlay_path = os.path.join(viz_dir, f"{tile_id}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"  Saved → {graph_path}")
        print(f"  Overlay → {overlay_path}")

    print(f"\nDone. Total inference time: {total_time:.1f}s for {len(test_indices)} tiles.")
    time_path = os.path.join(args.output_dir, "inference_time.txt")
    with open(time_path, "w") as f:
        f.write(f"Inference completed in {total_time:.1f}s for {len(test_indices)} tiles.\n")
