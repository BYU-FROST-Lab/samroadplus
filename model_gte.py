import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
from functools import partial
import lightning.pytorch as pl
from sam.segment_anything.modeling.image_encoder import ImageEncoderViT
from sam.segment_anything.modeling.mask_decoder import MaskDecoder
from sam.segment_anything.modeling.prompt_encoder import PromptEncoder
from sam.segment_anything.modeling.transformer import TwoWayTransformer
from sam.segment_anything.modeling.common import LayerNorm2d
import wandb
import pprint
import torchvision
import numpy as np
# vitdet is imported lazily inside SAMGraph.__init__ when NO_SAM=True


# ============================================================
# GTE Constants
# ============================================================
MAX_DEGREE = 6
VECTOR_NORM = 25.0



# ============================================================
# TopoNet: Learned Edge Classifier (ported from samroadplus)
# ============================================================
def find_highest_mask_point(x, y, mask, device='cuda'):
    """Find highest-scoring point within a small radius on the mask."""
    H, W, D = mask.shape
    x = torch.clamp(x, 0, W)
    y = torch.clamp(y, 0, D)
    x = int(x)
    y = int(y)
    radius = torch.tensor(2)
    x_min = max(0, x - radius)
    x_max = min(W, x + radius)
    y_min = max(0, y - radius)
    y_max = min(D, y + radius)

    mask_region = mask[:, x_min:x_max, y_min:y_max].to(device)

    x_coords = torch.arange(x_min, x_max, device=device).view(-1, 1).expand(x_max - x_min, y_max - y_min)
    y_coords = torch.arange(y_min, y_max, device=device).view(1, -1).expand(x_max - x_min, y_max - y_min)

    distances = torch.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)

    within_radius = (distances <= radius).to(device)
    mask_scores = mask_region[1] * within_radius + mask_region[0] * within_radius

    if mask_scores.numel() > 0:
        mask_max = torch.max(mask_scores)
        max_pos = torch.nonzero(mask_scores == mask_max)
        if len(max_pos) > 0:
            x_final = max_pos[0][0] + x_min
            y_final = max_pos[0][1] + y_min
        else:
            x_final, y_final = x, y
    else:
        x_final, y_final = x, y

    return x_final, y_final


def extract_point(x1, y1, x2, y2, image, num_points):
    """Uniformly sample points between two endpoints for line feature extraction."""
    H, W = image.shape[-2:]
    x_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)
    y_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)

    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values

    x_interp = torch.clamp(x_interp.long(), min=0, max=W - 1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H - 1)

    x_plus_1 = torch.clamp(x_interp + 1, max=W - 1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H - 1)

    x_final = torch.cat([x_interp, x_interp, x_plus_1], dim=-1)
    y_final = torch.cat([y_interp, y_plus_1, y_interp], dim=-1)

    return (x_final, y_final)


def extendline(points1, points2, image):
    """Sample features along extended line between two points in batch."""
    B, N, _ = points1.shape
    H, W = image.shape[-2:]
    extend_length = 8
    batch_A = points1
    batch_B = points2
    directions = batch_B - batch_A
    lengths = torch.norm(directions, dim=2, keepdim=True)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths
    extended_A = batch_A - directions_norm * extend_length
    extended_B = batch_B + directions_norm * extend_length
    extended_A = torch.round(extended_A).long()
    extended_B = torch.round(extended_B).long()
    extended_A[..., 0] = extended_A[..., 0].clamp(0, W - 1)
    extended_A[..., 1] = extended_A[..., 1].clamp(0, H - 1)
    extended_B[..., 0] = extended_B[..., 0].clamp(0, W - 1)
    extended_B[..., 1] = extended_B[..., 1].clamp(0, H - 1)

    extend_x1, extend_y1 = extended_A[..., 0], extended_A[..., 1]
    extend_x2, extend_y2 = extended_B[..., 0], extended_B[..., 1]
    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    x_final_1, y_final_1 = extract_point(extend_x1, extend_y1, x1, y1, image, num_points=15)
    x_final, y_final = extract_point(x1, y1, x2, y2, image, num_points=20)
    x_final_2, y_final_2 = extract_point(extend_x2, extend_y2, x2, y2, image, num_points=15)

    features1 = image[np.arange(B)[:, None, None], x_final_1, y_final_1]
    features = image[np.arange(B)[:, None, None], x_final, y_final]
    features2 = image[np.arange(B)[:, None, None], x_final_2, y_final_2]
    features = torch.concat([features1, features, features2], dim=2)
    return features


class BilinearSampler(nn.Module):
    """Sample backbone features at graph point locations via bilinear interpolation."""
    def __init__(self, config):
        super(BilinearSampler, self).__init__()
        self.config = config

    def forward(self, feature_maps, sample_points, mask_scores):
        B, D, H, W = feature_maps.shape
        batch_size, N_points, _ = sample_points.shape

        target_new_points = torch.zeros_like(sample_points).cuda()
        for batch_index in range(batch_size):
            for point_index in range(N_points):
                x, y = sample_points[batch_index, point_index]
                if (x.item(), y.item()) == (0, 0):
                    target_new_points[batch_index, point_index] = torch.tensor([x, y])
                else:
                    current_mask = mask_scores[batch_index]
                    x_new, y_new = find_highest_mask_point(x, y, current_mask)
                    target_new_points[batch_index, point_index] = torch.tensor([x_new, y_new], dtype=torch.float32)
        point = target_new_points

        target_new_points = (target_new_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        target_new_points = target_new_points.unsqueeze(2)
        sampled_features = F.grid_sample(feature_maps, target_new_points, mode='bilinear', align_corners=False)
        sampled_features_target = sampled_features.squeeze(dim=-1).permute(0, 2, 1)

        sample_points = (sample_points / self.config.PATCH_SIZE) * 2.0 - 1.0
        sample_points = sample_points.unsqueeze(2)
        sampled_features_o = F.grid_sample(feature_maps, sample_points, mode='bilinear', align_corners=False)
        sampled_features_source = sampled_features_o.squeeze(dim=-1).permute(0, 2, 1)

        return sampled_features_target, point, sampled_features_source


class TopoNet(nn.Module):
    """Learned edge classifier: given two candidate nodes, predict connection probability."""
    def __init__(self, config, feature_dim):
        super(TopoNet, self).__init__()
        self.config = config
        self.hidden_dim = 128
        self.heads = 4
        self.num_attn_layers = 3
        self.feature_proj = nn.Linear(feature_dim, self.hidden_dim)
        self.pair_proj = nn.Linear(2 * self.hidden_dim + 152, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        toponet_version = getattr(config, 'TOPONET_VERSION', 'normal')
        if toponet_version != 'no_transformer':
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.output_proj = nn.Linear(self.hidden_dim, 1)

    def forward(self, points, point_features, graph_points, point_features_o, pairs, pairs_valid, mask_scores):
        B, _, H, W = mask_scores.shape
        point_features = F.relu(self.feature_proj(point_features))
        point_features_o = F.relu(self.feature_proj(point_features_o))
        batch_size, n_samples, n_pairs, _ = pairs.shape
        pairs = pairs.view(batch_size, -1, 2)
        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n_samples * n_pairs)
        src_features = point_features_o[batch_indices, pairs[:, :, 0]]
        tgt_features = point_features[batch_indices, pairs[:, :, 1]]
        src_points = graph_points[batch_indices, pairs[:, :, 0]]
        tgt_points = points[batch_indices, pairs[:, :, 1]]
        _, N, _ = tgt_points.shape
        mask_road_dim = mask_scores[:, 1, :, :]
        line_features = extendline(src_points, tgt_points, mask_road_dim)
        offset_x = tgt_points - src_points
        pair_features = torch.concat([src_features, tgt_features, line_features, offset_x], dim=2)
        pair_features = F.relu(self.pair_proj(pair_features))
        pair_features = pair_features.view(batch_size * n_samples, n_pairs, -1)
        pairs_valid = pairs_valid.view(batch_size * n_samples, n_pairs)
        all_invalid_pair_mask = torch.eq(torch.sum(pairs_valid, dim=-1), 0).unsqueeze(-1)
        pairs_valid = torch.logical_or(pairs_valid, all_invalid_pair_mask)
        padding_mask = ~pairs_valid
        toponet_version = getattr(self.config, 'TOPONET_VERSION', 'normal')
        if toponet_version != 'no_transformer':
            pair_features = self.transformer_encoder(pair_features, src_key_padding_mask=padding_mask)
        _, n_pairs, _ = pair_features.shape
        pair_features = pair_features.view(batch_size, n_samples, n_pairs, -1)
        logits = self.output_proj(pair_features)
        scores = torch.sigmoid(logits)
        return logits, scores


# ============================================================
# GTE Decoder: feature map → 26-channel GTE output
# ============================================================
class GTEDecoder(nn.Module):
    """
    Decodes SAM encoder features into Sat2Graph's Graph Tensor Encoding (GTE).
    
    Output channels (with joint_with_seg=True, max_degree=6):
        - 2 channels: vertex probability (present / absent)
        - 6 × 4 = 24 channels: per-direction (2 edge prob + 2 direction vector)
        - 2 channels: segmentation (road / non-road)
        Total: 28 channels (or 26 without seg)
    """
    def __init__(self, encoder_dim=256, max_degree=6, joint_with_seg=True):
        super().__init__()
        self.max_degree = max_degree
        self.joint_with_seg = joint_with_seg
        # 2 (vertex) + max_degree * 4 (edge prob + direction vec) + optional 2 (seg)
        output_ch = 2 + max_degree * 4 + (2 if joint_with_seg else 0)
        
        activation = nn.GELU
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, output_ch, kernel_size=2, stride=2),
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, encoder_dim, h, w] from SAM encoder
        Returns:
            gte_output: [B, output_ch, H, W] raw logits for GTE
        """
        return self.decoder(features)


# ============================================================
# CBAM: Convolutional Block Attention Module (Woo et al., 2018)
# ============================================================
class ChannelAttention(nn.Module):
    """Learns which feature channels carry road-relevant information."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.mlp(x.mean(dim=[2, 3]))  # [B, C]
        max_out = self.mlp(x.amax(dim=[2, 3]))  # [B, C]
        weights = torch.sigmoid(avg_out + max_out)  # [B, C]
        return x * weights.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Learns which pixel locations are road-relevant."""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)   # [B, 1, H, W]
        spatial_weights = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_weights


class CBAMBlock(nn.Module):
    """Sequential Channel + Spatial attention."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ============================================================
# GTEDecoderV2: Wider bottleneck + CBAM attention at each scale
# ============================================================
class GTEDecoderV2(nn.Module):
    """
    Upgraded GTE decoder with:
    - Wider channel pipeline: 256 → 256 → 128 → 64 → output_ch
    - CBAM attention after each upsampling stage
    - LayerNorm at every level for stable training
    """
    def __init__(self, encoder_dim=256, max_degree=6, joint_with_seg=True):
        super().__init__()
        self.max_degree = max_degree
        self.joint_with_seg = joint_with_seg
        output_ch = 2 + max_degree * 4 + (2 if joint_with_seg else 0)

        activation = nn.GELU

        # Stage 1: 32×32 → 64×64, keep full 256 channels
        self.up1 = nn.ConvTranspose2d(encoder_dim, 256, kernel_size=2, stride=2)
        self.norm1 = LayerNorm2d(256)
        self.act1 = activation()
        self.attn1 = CBAMBlock(256, reduction=16)

        # Stage 2: 64×64 → 128×128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(128)
        self.act2 = activation()
        self.attn2 = CBAMBlock(128, reduction=16)

        # Stage 3: 128×128 → 256×256
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.norm3 = LayerNorm2d(64)
        self.act3 = activation()
        self.attn3 = CBAMBlock(64, reduction=8)

        # Stage 4: 256×256 → 512×512, project to output channels
        self.up4 = nn.ConvTranspose2d(64, output_ch, kernel_size=2, stride=2)

    def forward(self, features):
        """
        Args:
            features: [B, encoder_dim, h, w] from SAM encoder
        Returns:
            gte_output: [B, output_ch, H, W] raw logits for GTE
        """
        x = self.attn1(self.act1(self.norm1(self.up1(features))))  # [B, 256, 2h, 2w]
        x = self.attn2(self.act2(self.norm2(self.up2(x))))         # [B, 128, 4h, 4w]
        x = self.attn3(self.act3(self.norm3(self.up3(x))))         # [B, 64, 8h, 8w]
        x = self.up4(x)                                            # [B, out, 16h, 16w]
        return x


# ============================================================
# GTE Loss: ported from Sat2Graph's SupervisedLoss (TF → PyTorch)
# ============================================================
class GTELoss(nn.Module):
    """
    Computes Sat2Graph's supervised loss for Graph Tensor Encoding.
    
    Expects:
        gte_output: [B, C, H, W] — raw logits from GTEDecoder (channels-first)
        target_prob: [B, H, W, 2*(MAX_DEGREE+1)] — ground truth probabilities
        target_vector: [B, H, W, 2*MAX_DEGREE] — ground truth direction vectors
        gt_seg: [B, H, W, 1] — binary road segmentation GT (-0.5 to 0.5 range)
    """
    def __init__(self, max_degree=6, joint_with_seg=True,
                 keypoint_weight=1.0, direction_prob_weight=10.0,
                 direction_vector_weight=1000.0, seg_weight=0.1):
        super().__init__()
        self.max_degree = max_degree
        self.joint_with_seg = joint_with_seg
        self.keypoint_weight = keypoint_weight
        self.direction_prob_weight = direction_prob_weight
        self.direction_vector_weight = direction_vector_weight
        self.seg_weight = seg_weight
    
    def forward(self, gte_output, target_prob, target_vector, gt_seg=None):
        """
        Args:
            gte_output: [B, C, H, W] raw logits
            target_prob: [B, H, W, 2*(MAX_DEGREE+1)] = [B, H, W, 14]
            target_vector: [B, H, W, 2*MAX_DEGREE] = [B, H, W, 12]
            gt_seg: [B, H, W, 1] values in [-0.5, 0.5]
        Returns:
            dict of individual losses and total
        """
        B, C, H, W = gte_output.shape
        
        # Convert gte_output to channels-last [B, H, W, C] to match Sat2Graph convention
        gte = gte_output.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Unstack the channels
        gte_channels = [gte[..., i:i+1] for i in range(C)]
        prob_channels = [target_prob[..., i:i+1] for i in range(target_prob.shape[-1])]
        vec_channels = [target_vector[..., i:i+1] for i in range(target_vector.shape[-1])]
        
        # Soft mask: probability of vertex presence, used to weight directional losses
        # Only compute losses at keypoint locations (where vertices exist)
        soft_mask = torch.clamp(prob_channels[0] - 0.01, 0.0, 0.99) + 0.01  # [B, H, W, 1]
        soft_mask_squeezed = soft_mask.squeeze(-1)  # [B, H, W]
        
        # ========== Keypoint probability loss ==========
        # Channels [0:2] are vertex present/absent logits
        kp_output = torch.cat(gte_channels[0:2], dim=-1)   # [B, H, W, 2]
        kp_target = torch.cat(prob_channels[0:2], dim=-1)   # [B, H, W, 2]
        # Softmax cross-entropy: target is soft labels
        kp_log_probs = F.log_softmax(kp_output, dim=-1)
        keypoint_prob_loss = -torch.mean(torch.sum(kp_target * kp_log_probs, dim=-1))
        
        # ========== Direction probability loss ==========
        direction_prob_loss = torch.tensor(0.0, device=gte_output.device)
        
        for i in range(self.max_degree):
            # GTE output channels for direction i: [2+i*4 : 2+i*4+2] are edge prob logits
            dir_output = torch.cat(gte_channels[2 + i*4 : 2 + i*4 + 2], dim=-1)  # [B, H, W, 2]
            # Target prob channels for direction i: [2+i*2 : 2+i*2+2]
            dir_target = torch.cat(prob_channels[2 + i*2 : 2 + i*2 + 2], dim=-1)  # [B, H, W, 2]
            
            # Softmax CE, masked at keypoints only
            dir_log_probs = F.log_softmax(dir_output, dim=-1)
            per_pixel_loss = -torch.sum(dir_target * dir_log_probs, dim=-1)  # [B, H, W]
            direction_prob_loss = direction_prob_loss + torch.mean(soft_mask_squeezed * per_pixel_loss)
        
        direction_prob_loss = direction_prob_loss / self.max_degree
        
        # ========== Direction vector loss ==========
        direction_vector_loss = torch.tensor(0.0, device=gte_output.device)
        
        for i in range(self.max_degree):
            # GTE output channels for direction i: [2+i*4+2 : 2+i*4+4] are direction vectors
            vec_output = torch.cat(gte_channels[2 + i*4 + 2 : 2 + i*4 + 4], dim=-1)  # [B, H, W, 2]
            # Target vector channels: [i*2 : i*2+2]
            vec_target = torch.cat(vec_channels[i*2 : i*2 + 2], dim=-1)  # [B, H, W, 2]
            
            # MSE masked at keypoints
            squared_error = torch.square(vec_output - vec_target)  # [B, H, W, 2]
            direction_vector_loss = direction_vector_loss + torch.mean(soft_mask * squared_error)
        
        direction_vector_loss = direction_vector_loss / self.max_degree
        
        # ========== Segmentation loss ==========
        seg_loss = torch.tensor(0.0, device=gte_output.device)
        if self.joint_with_seg and gt_seg is not None:
            # Last 2 GTE channels are seg logits
            seg_output = torch.cat(
                [gte_channels[2 + self.max_degree * 4],
                 gte_channels[2 + self.max_degree * 4 + 1]], dim=-1
            )  # [B, H, W, 2]
            # Target: convert gt_seg from [-0.5, 0.5] to soft labels [road_prob, non_road_prob]
            seg_gt_target = torch.cat([gt_seg + 0.5, 0.5 - gt_seg], dim=-1)  # [B, H, W, 2]
            seg_log_probs = F.log_softmax(seg_output, dim=-1)
            seg_loss = -torch.mean(torch.sum(seg_gt_target * seg_log_probs, dim=-1))
        
        # Apply weights
        weighted_losses = {
            'gte_keypoint_loss': keypoint_prob_loss * self.keypoint_weight,
            'gte_dir_prob_loss': direction_prob_loss * self.direction_prob_weight,
            'gte_dir_vec_loss': direction_vector_loss * self.direction_vector_weight,
            'gte_seg_loss': seg_loss * self.seg_weight,
        }
        weighted_losses['gte_total'] = sum(weighted_losses.values())
        
        return weighted_losses


# ============================================================
# GTE Output Decoding (for inference)
# ============================================================
def gte_softmax_output(gte_output, max_degree=6, joint_with_seg=True):
    """
    Converts raw GTE logits to probabilities and normalized vectors.
    Port of Sat2Graph's SoftmaxOutput.
    
    Args:
        gte_output: [B, C, H, W] raw logits (channels-first)
    Returns:
        [B, C_out, H, W] with sigmoid probabilities and raw vectors
    """
    # Convert to channels-last for easier indexing
    gte = gte_output.permute(0, 2, 3, 1)  # [B, H, W, C]
    channels = [gte[..., i:i+1] for i in range(gte.shape[-1])]
    
    new_outputs = []
    
    # Vertex probability: sigmoid(ch0 - ch1)
    vertex_prob = torch.sigmoid(channels[0] - channels[1])
    new_outputs.append(vertex_prob)
    new_outputs.append(1.0 - vertex_prob)
    
    # Per-direction: edge probability + direction vector
    for i in range(max_degree):
        edge_prob = torch.sigmoid(channels[2 + i*4] - channels[2 + i*4 + 1])
        new_outputs.append(edge_prob)
        new_outputs.append(1.0 - edge_prob)
        new_outputs.append(torch.cat(channels[2 + i*4 + 2 : 2 + i*4 + 4], dim=-1))
    
    # Segmentation probability
    if joint_with_seg:
        seg_prob = torch.sigmoid(channels[2 + 4*max_degree] - channels[2 + 4*max_degree + 1])
        new_outputs.append(seg_prob)
        new_outputs.append(1.0 - seg_prob)
    
    result = torch.cat(new_outputs, dim=-1)
    return result.permute(0, 3, 1, 2)  # back to [B, C_out, H, W]


class HieraGTEDecoderV3(nn.Module):
    """
    Attention-Guided FPN Decoder for SAM 2's Hierarchical Architecture.
    Takes the 32x32 bottleneck and explicitly concatenates the raw high-resolution 
    64x64 and 128x128 feature pyramids physically extracted from SAM 2.
    It passes the combinations through CBAM Attention blocks to filter the geometry.
    """
    def __init__(self, encoder_dim=256, max_degree=6, joint_with_seg=True):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.max_degree = max_degree
        self.joint_with_seg = joint_with_seg
        out_channels = 2 + max_degree * 4
        if joint_with_seg: 
            out_channels += 2
            
        # Up 1: 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(encoder_dim, 128, kernel_size=2, stride=2)
        # Fuse with native 64x64 (which comes as 256 dims). Total = 128 + 256 = 384
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128 + encoder_dim, 128, kernel_size=3, padding=1),
            LayerNorm2d(128),
            nn.GELU(),
            CBAMBlock(128)
        )
        
        # Up 2: 64x64 -> 128x128
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Fuse with native 128x128. Total = 64 + 256 = 320
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + encoder_dim, 64, kernel_size=3, padding=1),
            LayerNorm2d(64),
            nn.GELU(),
            CBAMBlock(64)
        )
        
        # Up 3: 128x128 -> 256x256
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.fuse3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            LayerNorm2d(32),
            nn.GELU(),
            CBAMBlock(32)
        )
        
        # Up 4: 256x256 -> 512x512
        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x, fpn_features=None):
        # x is [B, 256, 32, 32] (lowest-res bottleneck)
        # SAM 2 inherently returns high-res tensors inside `fpn_features`
        f_64 = None
        f_128 = None
        if fpn_features is not None:
            for feat in fpn_features:
                if feat.shape[-1] == 64: f_64 = feat
                if feat.shape[-1] == 128: f_128 = feat
        
        # Up 1
        x = self.up1(x)  # -> 64x64
        if f_64 is not None:
            x = torch.cat([x, f_64], dim=1)
            x = self.fuse1(x)
        else:
            dummy = torch.zeros(x.shape[0], self.encoder_dim, x.shape[2], x.shape[3], device=x.device)
            x = self.fuse1(torch.cat([x, dummy], dim=1))
            
        # Up 2
        x = self.up2(x)  # -> 128x128
        if f_128 is not None:
            x = torch.cat([x, f_128], dim=1)
            x = self.fuse2(x)
        else:
            dummy = torch.zeros(x.shape[0], self.encoder_dim, x.shape[2], x.shape[3], device=x.device)
            x = self.fuse2(torch.cat([x, dummy], dim=1))
            
        # Up 3 & 4 (Standard FPN ending to 512x512)
        x = self.up3(x)
        x = self.fuse3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        
        return x


# ============================================================
# LoRA for SAM (from samroadplus)
# ============================================================
class _LoRA_qkv(nn.Module):
    """LoRA adaptation for SAM's QKV linear layers."""
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.weight = qkv.weight
        self.bias = qkv.bias
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = F.linear(x, self.weight, self.bias)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


# ============================================================
# PyTorch Port of Sat2Graph Deep Layer Aggregation (DLA)
# ============================================================
class DLAResBlock(nn.Module):
    def __init__(self, in_channels, channels, downsample=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.downsample = None
        if downsample or in_channels != channels:
            self.downsample = nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False)
            
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        
        if self.downsample is not None:
            identity = self.downsample(out)
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity
        return out

class DLAReduceBlock(nn.Module):
    def __init__(self, in_ch, out_ch, resnet_step=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        layers = []
        if resnet_step > 0:
            for _ in range(resnet_step):
                layers.append(DLAResBlock(out_ch, out_ch))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.resnet = nn.Sequential(*layers) if layers else nn.Identity()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.resnet(x)

class DLAAggregateBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, batchnorm=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch2, in_ch2, 3, stride=2, padding=1, output_padding=1, bias=True if not batchnorm else False)
        self.bn_deconv = nn.BatchNorm2d(in_ch2) if batchnorm else nn.Identity()
        
        cat_ch = in_ch1 + in_ch2
        self.conv1 = nn.Conv2d(cat_ch, cat_ch, 3, padding=1, bias=True if not batchnorm else False)
        self.bn1 = nn.BatchNorm2d(cat_ch) if batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(cat_ch, out_ch, 3, padding=1, bias=True if not batchnorm else False)
        self.bn2 = nn.BatchNorm2d(out_ch) if batchnorm else nn.Identity()
        
    def forward(self, x1, x2):
        x2 = F.relu(self.bn_deconv(self.deconv(x2)))
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class Sat2GraphDLA(nn.Module):
    """Exact PyTorch port of Sat2Graph's TensorFlow Deep Layer Aggregation architecture"""
    def __init__(self, in_channels=3, output_ch=26, ch=24, resnet_step=8):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, ch, 5, padding=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch*2, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(inplace=True)
        )
        
        self.x_4s = DLAReduceBlock(ch*2, ch*4, resnet_step // 8)
        self.x_8s = DLAReduceBlock(ch*4, ch*8, resnet_step // 4)
        self.x_16s = DLAReduceBlock(ch*8, ch*16, resnet_step // 2)
        self.x_32s = DLAReduceBlock(ch*16, ch*32, resnet_step)
        
        self.a1_2s = DLAAggregateBlock(ch*2, ch*4, ch*4)
        self.a1_4s = DLAAggregateBlock(ch*4, ch*8, ch*8)
        self.a1_8s = DLAAggregateBlock(ch*8, ch*16, ch*16)
        self.a1_16s = DLAAggregateBlock(ch*16, ch*32, ch*32)
        self.a1_16s_res = nn.Sequential(*[DLAResBlock(ch*32, ch*32) for _ in range(resnet_step // 2)], nn.BatchNorm2d(ch*32), nn.ReLU(inplace=True)) if resnet_step//2 > 0 else nn.Identity()
        
        self.a2_2s = DLAAggregateBlock(ch*4, ch*8, ch*4)
        self.a2_4s = DLAAggregateBlock(ch*8, ch*16, ch*8)
        self.a2_8s = DLAAggregateBlock(ch*16, ch*32, ch*16)
        self.a2_8s_res = nn.Sequential(*[DLAResBlock(ch*16, ch*16) for _ in range(resnet_step // 4)], nn.BatchNorm2d(ch*16), nn.ReLU(inplace=True)) if resnet_step//4 > 0 else nn.Identity()
        
        self.a3_2s = DLAAggregateBlock(ch*4, ch*8, ch*4)
        self.a3_4s = DLAAggregateBlock(ch*8, ch*16, ch*8)
        self.a3_4s_res = nn.Sequential(*[DLAResBlock(ch*8, ch*8) for _ in range(resnet_step // 8)], nn.BatchNorm2d(ch*8), nn.ReLU(inplace=True)) if resnet_step//8 > 0 else nn.Identity()
        
        self.a4_2s = DLAAggregateBlock(ch*4, ch*8, ch*8)
        
        self.a5_2s = nn.Sequential(
            nn.Conv2d(ch*8, ch*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch*4),
            nn.ReLU(inplace=True)
        )
        
        self.a_out = DLAAggregateBlock(ch, ch*4, ch*4, batchnorm=False)
        self.out = nn.Conv2d(ch*4, output_ch, 3, padding=1)
        
    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = self.conv2(c1)
        
        x_4s = self.x_4s(c2)
        x_8s = self.x_8s(x_4s)
        x_16s = self.x_16s(x_8s)
        x_32s = self.x_32s(x_16s)
        
        a1_2s = self.a1_2s(c2, x_4s)
        a1_4s = self.a1_4s(x_4s, x_8s)
        a1_8s = self.a1_8s(x_8s, x_16s)
        a1_16s = self.a1_16s_res(self.a1_16s(x_16s, x_32s))
        
        a2_2s = self.a2_2s(a1_2s, a1_4s)
        a2_4s = self.a2_4s(a1_4s, a1_8s)
        a2_8s = self.a2_8s_res(self.a2_8s(a1_8s, a1_16s))
        
        a3_2s = self.a3_2s(a2_2s, a2_4s)
        a3_4s = self.a3_4s_res(self.a3_4s(a2_4s, a2_8s))
        
        a4_2s = self.a4_2s(a3_2s, a3_4s)
        a5_2s = self.a5_2s(a4_2s)
        
        a_out = self.a_out(c1, a5_2s)
        
        return self.out(a_out)


# ============================================================
# SAMGraph: Main Model (Lightning Module)
# ============================================================
class SAMGraph(pl.LightningModule):
    """
    SAMGraph combines SAM's ViT backbone with Sat2Graph's GTE output head
    and a secondary segmentation decoder for dual-supervision training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_degree = getattr(config, 'MAX_DEGREE', MAX_DEGREE)
        self.joint_with_seg = getattr(config, 'JOINT_WITH_SEG', True)
        
        # ---- SAM Backbone ----
        self.is_sam2 = getattr(config, 'SAM_VERSION', '') == 'sam2_hiera_b+'
        self.is_dinov3 = hasattr(config, 'BACKBONE') and 'dinov3' in config.BACKBONE
        self.is_radio = hasattr(config, 'BACKBONE') and 'radio' in config.BACKBONE
        self.is_resnet = hasattr(config, 'BACKBONE') and 'resnet' in config.BACKBONE
        self.is_dinov2 = hasattr(config, 'BACKBONE') and config.BACKBONE == 'dinov2'
        
        if getattr(config, 'NO_VFM', False):
            # Sat2Graph Baseline (DLA)
            output_ch = 2 + self.max_degree * 4 + (2 if self.joint_with_seg else 0)
            self.sat2graph_dla = Sat2GraphDLA(in_channels=3, output_ch=output_ch, ch=24, resnet_step=8)
            self.matched_param_names = []
        else:
            if not (self.is_dinov3 or self.is_radio or self.is_resnet or self.is_dinov2):
                assert config.SAM_VERSION in {'vit_b', 'vit_l', 'vit_h', 'sam2_hiera_b+'}
            if config.SAM_VERSION == 'vit_b':
                encoder_embed_dim = 768
                encoder_depth = 12
                encoder_num_heads = 12
                encoder_global_attn_indexes = [2, 5, 8, 11]
            elif config.SAM_VERSION == 'vit_l':
                encoder_embed_dim = 1024
                encoder_depth = 24
                encoder_num_heads = 16
                encoder_global_attn_indexes = [5, 11, 17, 23]
            elif config.SAM_VERSION == 'vit_h':
                encoder_embed_dim = 1280
                encoder_depth = 32
                encoder_num_heads = 16
                encoder_global_attn_indexes = [7, 15, 23, 31]
            
            prompt_embed_dim = 256
            image_size = config.PATCH_SIZE
            self.image_size = image_size
            vit_patch_size = 16
            encoder_output_dim = prompt_embed_dim
            
            self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
            self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)
            
            if self.is_dinov2:
                self.image_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                self.dinov2_proj = nn.Conv2d(768, encoder_output_dim, kernel_size=1)
                self.matched_param_names = [f"image_encoder.{k}" for k, _ in self.image_encoder.named_parameters()]
            elif self.is_resnet:
                resnet = torchvision.models.resnet50(pretrained=True)
                self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])
                self.resnet_proj = nn.Conv2d(2048, encoder_output_dim, kernel_size=1)
                self.matched_param_names = [f"image_encoder.{k}" for k, _ in self.image_encoder.named_parameters()]
            elif self.is_dinov3:
                from transformers import AutoModel
                self.image_encoder = AutoModel.from_pretrained('facebook/dinov2-base')
                self.matched_param_names = [f"image_encoder.{k}" for k, _ in self.image_encoder.named_parameters()]
                encoder_output_dim = 768
            elif self.is_radio:
                self.image_encoder = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-b', progress=True, skip_validation=True)
                self.matched_param_names = [f"image_encoder.{k}" for k, _ in self.image_encoder.named_parameters()]
                encoder_output_dim = 1024
            elif self.is_sam2:
                from sam2.build_sam import build_sam2
                print(f"Building SAM 2 from {config.SAM_CKPT_PATH}...")
                sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_b+.yaml", config.SAM_CKPT_PATH)
                self.image_encoder = sam2_model.image_encoder
            else:
                self.image_encoder = ImageEncoderViT(
                    depth=encoder_depth,
                    embed_dim=encoder_embed_dim,
                    img_size=image_size,
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=prompt_embed_dim,
                )
        
        # ---- SAM Segmentation Decoder (2-ch: keypoint + road) ----
        activation = nn.GELU
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_dim, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            activation(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            activation(),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2),
        )
        
        if not getattr(self.config, 'NO_VFM', False):
            # ---- GTE Decoder (26-ch GTE output) ----
            gte_version = getattr(config, 'GTE_DECODER_VERSION', 'v1')
            if gte_version == 'v3':
                self.gte_decoder = HieraGTEDecoderV3(
                    encoder_dim=encoder_output_dim,
                    max_degree=self.max_degree,
                    joint_with_seg=self.joint_with_seg,
                )
                print(f"Using HieraGTEDecoderV3 (Attention-Guided Feature Pyramid Network for SAM 2)")
            elif gte_version == 'v2':
                self.gte_decoder = GTEDecoderV2(
                    encoder_dim=encoder_output_dim,
                    max_degree=self.max_degree,
                    joint_with_seg=self.joint_with_seg,
                )
                print(f"Using GTEDecoderV2 (wider bottleneck + CBAM attention)")
            else:
                self.gte_decoder = GTEDecoder(
                    encoder_dim=encoder_output_dim,
                    max_degree=self.max_degree,
                    joint_with_seg=self.joint_with_seg,
                )
            
            # ---- LoRA ----
            if getattr(config, 'ENCODER_LORA', False):
                r = self.config.LORA_RANK
                assert r > 0
            
            if getattr(self.config, 'SAM_VERSION', '') == 'sam2_hiera_b+':
                # Dynamic LoRA injection for SAM 2's Hiera hierarchy
                self.w_As = []
                self.w_Bs = []
                # Freeze all parameters
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                    
                # Crawl the PyTorch module tree looking for any Module that owns a .qkv Linear layer
                injected_count = 0
                for module in self.image_encoder.modules():
                    if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                        w_qkv_linear = module.qkv
                        dim = w_qkv_linear.in_features
                        w_a_linear_q = nn.Linear(dim, r, bias=False)
                        w_b_linear_q = nn.Linear(r, dim, bias=False)
                        w_a_linear_v = nn.Linear(dim, r, bias=False)
                        w_b_linear_v = nn.Linear(r, dim, bias=False)
                        
                        self.w_As.extend([w_a_linear_q, w_a_linear_v])
                        self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
                        
                        module.qkv = _LoRA_qkv(
                            w_qkv_linear,
                            w_a_linear_q, w_b_linear_q,
                            w_a_linear_v, w_b_linear_v,
                        )
                        injected_count += 1
                        
                for w_A in self.w_As:
                    nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                for w_B in self.w_Bs:
                    nn.init.zeros_(w_B.weight)
                print(f"Injected LoRA into {injected_count} SAM 2 Hiera Attention blocks.")
            else:
                self.lora_layer_selection = list(range(len(self.image_encoder.blocks)))
                self.w_As = []
                self.w_Bs = []
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                for t_layer_i, blk in enumerate(self.image_encoder.blocks):
                    if t_layer_i not in self.lora_layer_selection:
                        continue
                    w_qkv_linear = blk.attn.qkv
                    dim = w_qkv_linear.in_features
                    w_a_linear_q = nn.Linear(dim, r, bias=False)
                    w_b_linear_q = nn.Linear(r, dim, bias=False)
                    w_a_linear_v = nn.Linear(dim, r, bias=False)
                    w_b_linear_v = nn.Linear(r, dim, bias=False)
                    self.w_As.append(w_a_linear_q)
                    self.w_Bs.append(w_b_linear_q)
                    self.w_As.append(w_a_linear_v)
                    self.w_Bs.append(w_b_linear_v)
                    blk.attn.qkv = _LoRA_qkv(
                        w_qkv_linear,
                        w_a_linear_q, w_b_linear_q,
                        w_a_linear_v, w_b_linear_v,
                    )
                for w_A in self.w_As:
                    nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
                for w_B in self.w_Bs:
                    nn.init.zeros_(w_B.weight)
        
        # ---- Losses ----
        # GTE loss (ported from Sat2Graph)
        self.gte_loss_fn = GTELoss(
            max_degree=self.max_degree,
            joint_with_seg=self.joint_with_seg,
            keypoint_weight=getattr(config, 'GTE_KEYPOINT_WEIGHT', 1.0),
            direction_prob_weight=getattr(config, 'GTE_DIRECTION_PROB_WEIGHT', 10.0),
            direction_vector_weight=getattr(config, 'GTE_DIRECTION_VECTOR_WEIGHT', 1000.0),
            seg_weight=getattr(config, 'GTE_SEG_WEIGHT', 0.1),
        )
        # SAM segmentation loss
        if self.config.FOCAL_LOSS:
            self.mask_criterion = partial(torchvision.ops.sigmoid_focal_loss, reduction='mean')
        else:
            self.mask_criterion = torch.nn.BCEWithLogitsLoss()
        
        self.sam_mask_weight = getattr(config, 'SAM_MASK_WEIGHT', 1.0)
        
        # ---- TopoNet Head ----
        self.toponet_enabled = getattr(config, 'TOPONET_ENABLED', False)
        if self.toponet_enabled:
            self.bilinear_sampler = BilinearSampler(config)
            self.topo_net = TopoNet(config, 256)
            self.topo_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.topo_loss_weight = getattr(config, 'TOPO_LOSS_WEIGHT', 1.0)
            print(f"TopoNet enabled (version={getattr(config, 'TOPONET_VERSION', 'normal')})")
        
        # ---- Load SAM checkpoint ----
        if self.config.NO_SAM:
            return
            
        if getattr(self.config, 'SAM_VERSION', '') == 'sam2_hiera_b+':
            # SAM 2 weights are already loaded via build_sam2
            self.matched_param_names = set(k for k, _ in self.image_encoder.named_parameters())
            return
            
        with open(config.SAM_CKPT_PATH, "rb") as f:
            ckpt_state_dict = torch.load(f)
            
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(
                    ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes
                )
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            print("###### Matched params ######")
            pprint.pprint(matched_names)
            print("###### Mismatched params ######")
            pprint.pprint(mismatch_names)
            
            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)
    
    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        """Resize SAM positional embeddings for non-1024 image sizes."""
        new_state_dict = {k: v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict
    
    # ================================================================
    # Forward
    # ================================================================
    def forward(self, rgb, graph_points=None, pairs=None, valid=None):
        """
        Args:
            rgb: [B, H, W, 3] satellite image (0-255 float)
            graph_points: [B, N_points, 2] candidate node coords (optional, for TopoNet)
            pairs: [B, N_samples, N_pairs, 2] node pair indices (optional)
            valid: [B, N_samples, N_pairs] padding mask (optional)
        Returns:
            gte_output: [B, 26, H, W] raw GTE logits
            mask_logits: [B, 2, H, W] raw seg logits (keypoint, road)
            mask_scores: [B, 2, H, W] sigmoid seg scores
            topo_logits: [B, N_samples, N_pairs, 1] edge logits (if TopoNet enabled)
            topo_scores: [B, N_samples, N_pairs, 1] edge probabilities (if TopoNet enabled)
        """
        # [B, 3, H, W]
        x = rgb.permute(0, 3, 1, 2)
        x = (x - self.pixel_mean) / self.pixel_std
        if getattr(self.config, 'NO_VFM', False):
            # Pure PyTorch port of Sat2Graph DLA
            gte_output = self.sat2graph_dla(x)
            
            # Since NO_VFM skips the SAM encoder and mask decoder, 
            # we simulate mask_logits and mask_scores with a zero tensor, 
            # or pull them out of gte_output if joint_with_seg is true.
            B, C, H, W = gte_output.shape
            mask_logits = torch.zeros((B, 2, H, W), device=x.device)
            if self.joint_with_seg:
                mask_logits = gte_output[:, -2:, :, :]
            mask_scores = torch.sigmoid(mask_logits)
            
            topo_logits, topo_scores = None, None
            return gte_output, mask_logits, mask_scores, topo_logits, topo_scores
            
        fpn_features = None
        if self.is_dinov2:
            x_dinov2 = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
            feats = self.image_encoder.get_intermediate_layers(x_dinov2, n=1, reshape=True)[0]
            image_embeddings = self.dinov2_proj(feats)
            image_embeddings = F.interpolate(image_embeddings, size=(32, 32), mode="bilinear", align_corners=False)
        elif self.is_resnet:
            feats = self.image_encoder(x)
            image_embeddings = self.resnet_proj(feats)
            image_embeddings = F.interpolate(image_embeddings, size=(32, 32), mode="bilinear", align_corners=False)
        elif self.is_dinov3:
            outputs = self.image_encoder(pixel_values=x, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state') and len(outputs.last_hidden_state.shape) == 3:
                from einops import rearrange
                image_embeddings = outputs.last_hidden_state
                h, w = x.shape[2]//16, x.shape[3]//16
                num_spatial = h * w
                if image_embeddings.shape[1] > num_spatial:
                    image_embeddings = image_embeddings[:, -num_spatial:, :]
                image_embeddings = rearrange(image_embeddings, 'b (h w) d -> b d h w', h=h, w=w)
            else:
                image_embeddings = outputs.hidden_states[4]
            image_embeddings = F.interpolate(image_embeddings, size=(32, 32), mode="bilinear", align_corners=False)
        elif self.is_radio:
            from einops import rearrange
            summary, features = self.image_encoder(x)
            image_embeddings = rearrange(features, 'b (h w) d -> b d h w', h=32, w=32)
        else:
            encoder_output = self.image_encoder(x)
            if isinstance(encoder_output, dict):
                image_embeddings = encoder_output.get("vision_features")
                fpn_features = encoder_output.get("backbone_fpn", None)
            else:
                image_embeddings = encoder_output
            
        # GTE head: [B, 26, H, W]
        if isinstance(self.gte_decoder, HieraGTEDecoderV3):
            gte_output = self.gte_decoder(image_embeddings, fpn_features=fpn_features)
        else:
            gte_output = self.gte_decoder(image_embeddings)
        
        # SAM seg head: [B, 2, H, W]
        mask_logits = self.map_decoder(image_embeddings)
        mask_scores = torch.sigmoid(mask_logits)
        
        # TopoNet head (optional)
        topo_logits, topo_scores = None, None
        if self.toponet_enabled and graph_points is not None:
            point_features, newpoint, point_features_o = self.bilinear_sampler(
                image_embeddings, graph_points, mask_scores
            )
            topo_logits, topo_scores = self.topo_net(
                newpoint, point_features, graph_points, point_features_o,
                pairs, valid, mask_scores
            )
        
        return gte_output, mask_logits, mask_scores, topo_logits, topo_scores
    
    # ================================================================
    # Training Step
    # ================================================================
    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        target_prob = batch['target_prob']
        target_vector = batch['target_vector']
        gt_seg = batch['gt_seg']
        keypoint_mask = batch['keypoint_mask']
        road_mask = batch['road_mask']
        
        # TopoNet data (may be None if not enabled)
        graph_points = batch.get('graph_points', None)
        pairs = batch.get('pairs', None)
        valid = batch.get('valid', None)
        
        gte_output, mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, graph_points=graph_points, pairs=pairs, valid=valid
        )
        
        # GTE losses
        gte_losses = self.gte_loss_fn(gte_output, target_prob, target_vector, gt_seg)
        
        # SAM segmentation loss
        mask_logits_hwc = mask_logits.permute(0, 2, 3, 1)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        sam_mask_loss = self.mask_criterion(mask_logits_hwc, gt_masks) * self.sam_mask_weight
        
        # Total loss
        total_loss = gte_losses['gte_total'] + sam_mask_loss
        
        # TopoNet loss
        topo_loss = torch.tensor(0.0, device=rgb.device)
        if self.toponet_enabled and topo_logits is not None:
            topo_gt = batch['connected'].to(torch.int32)
            topo_loss_mask = valid.to(torch.float32)
            topo_loss_raw = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
            topo_loss_raw *= topo_loss_mask.unsqueeze(-1)
            mask_sum = topo_loss_mask.sum()
            if mask_sum > 0:
                topo_loss = topo_loss_raw.sum() / mask_sum
            topo_loss = topo_loss * self.topo_loss_weight
            total_loss = total_loss + topo_loss
        
        # Handle NaN
        if torch.any(torch.isnan(total_loss)):
            print("NaN detected in loss. Using default loss value.")
            total_loss = torch.tensor(0.0, device=total_loss.device)
        
        # Logging
        self.log('train_gte_kp_loss', gte_losses['gte_keypoint_loss'], on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_gte_dir_prob', gte_losses['gte_dir_prob_loss'], on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_gte_dir_vec', gte_losses['gte_dir_vec_loss'], on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_gte_seg', gte_losses['gte_seg_loss'], on_step=True, on_epoch=False, prog_bar=False)
        self.log('train_sam_mask', sam_mask_loss, on_step=True, on_epoch=False, prog_bar=True)
        if self.toponet_enabled:
            self.log('train_topo_loss', topo_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_loss', total_loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return total_loss
    
    # ================================================================
    # Validation Step
    # ================================================================
    def validation_step(self, batch, batch_idx):
        rgb = batch['rgb']
        target_prob = batch['target_prob']
        target_vector = batch['target_vector']
        gt_seg = batch['gt_seg']
        keypoint_mask = batch['keypoint_mask']
        road_mask = batch['road_mask']
        
        graph_points = batch.get('graph_points', None)
        pairs = batch.get('pairs', None)
        valid = batch.get('valid', None)
        
        gte_output, mask_logits, mask_scores, topo_logits, topo_scores = self(
            rgb, graph_points=graph_points, pairs=pairs, valid=valid
        )
        
        # GTE losses
        gte_losses = self.gte_loss_fn(gte_output, target_prob, target_vector, gt_seg)
        
        # SAM seg loss
        mask_logits_hwc = mask_logits.permute(0, 2, 3, 1)
        gt_masks = torch.stack([keypoint_mask, road_mask], dim=3)
        sam_mask_loss = self.mask_criterion(mask_logits_hwc, gt_masks) * self.sam_mask_weight
        
        total_loss = gte_losses['gte_total'] + sam_mask_loss
        
        # TopoNet loss
        topo_loss = torch.tensor(0.0, device=rgb.device)
        if self.toponet_enabled and topo_logits is not None:
            topo_gt = batch['connected'].to(torch.int32)
            topo_loss_mask = valid.to(torch.float32)
            topo_loss_raw = self.topo_criterion(topo_logits, topo_gt.unsqueeze(-1).to(torch.float32))
            topo_loss_raw *= topo_loss_mask.unsqueeze(-1)
            mask_sum = topo_loss_mask.sum()
            if mask_sum > 0:
                topo_loss = topo_loss_raw.sum() / mask_sum
            topo_loss = topo_loss * self.topo_loss_weight
            total_loss = total_loss + topo_loss
        
        self.log('val_gte_kp_loss', gte_losses['gte_keypoint_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_gte_dir_prob', gte_losses['gte_dir_prob_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_gte_dir_vec', gte_losses['gte_dir_vec_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_gte_seg', gte_losses['gte_seg_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_sam_mask', sam_mask_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.toponet_enabled:
            self.log('val_topo_loss', topo_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log visualization images on first batch
        if batch_idx == 0:
            max_viz_num = 4
            viz_rgb = rgb[:max_viz_num]
            viz_pred_road = mask_scores[:max_viz_num, 1, :, :]  # road channel
            viz_pred_kp = mask_scores[:max_viz_num, 0, :, :]    # keypoint channel
            viz_gt_kp = keypoint_mask[:max_viz_num]
            viz_gt_road = road_mask[:max_viz_num]
            
            # GTE vertex probability
            gte_probs = gte_softmax_output(gte_output[:max_viz_num], self.max_degree, self.joint_with_seg)
            viz_gte_vertex = gte_probs[:, 0, :, :]  # vertex presence probability
            
            columns = ['rgb', 'gt_keypoint', 'gt_road', 'pred_keypoint', 'pred_road', 'gte_vertex']
            data = [
                [wandb.Image(x.cpu().numpy()) for x in row]
                for row in zip(viz_rgb, viz_gt_kp, viz_gt_road, viz_pred_kp, viz_pred_road, viz_gte_vertex)
            ]
            self.logger.log_table(key='viz_table', columns=columns, data=data)
    
    # ================================================================
    # Optimizer
    # ================================================================
    def configure_optimizers(self):
        param_dicts = []
        
        if not self.config.FREEZE_ENCODER and not self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters()
                           if 'image_encoder.' + k in self.matched_param_names],
                'lr': self.config.BASE_LR * self.config.ENCODER_LR_FACTOR,
            }
            param_dicts.append(encoder_params)
        
        if self.config.ENCODER_LORA:
            encoder_params = {
                'params': [p for k, p in self.image_encoder.named_parameters() if 'qkv.linear_' in k],
                'lr': self.config.BASE_LR,
            }
            param_dicts.append(encoder_params)
        
        # Map decoder params
        decoder_params = [{
            'params': list(self.map_decoder.parameters()),
            'lr': self.config.BASE_LR,
        }]
        param_dicts += decoder_params
        
        # GTE decoder params
        gte_params = [{
            'params': list(self.gte_decoder.parameters()),
            'lr': self.config.BASE_LR,
        }]
        param_dicts += gte_params
        
        # TopoNet params
        if self.toponet_enabled:
            topo_params = [{
                'params': list(self.topo_net.parameters()) + list(self.bilinear_sampler.parameters()),
                'lr': self.config.BASE_LR,
            }]
            param_dicts += topo_params
        
        for i, param_dict in enumerate(param_dicts):
            param_num = sum([int(p.numel()) for p in param_dict['params']])
            print(f'optim param dict {i} params num: {param_num}')
        
        optimizer = torch.optim.AdamW(param_dicts, lr=self.config.BASE_LR, betas=(0.9, 0.999), weight_decay=0.1)
        step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9], gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': step_lr}
