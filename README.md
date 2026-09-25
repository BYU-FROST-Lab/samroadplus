<div align="center">

<h1>What Transfers to Road Topology? A Controlled Study of Foundation-Model Representations Across Architectures and Geographic Domains</h1>

## Abstract
> *Road network graph extraction from satellite imagery is critical for autonomous navigation, urban planning, and disaster response. While vision foundation models have demonstrated strong performance in pixel-level segmentation, it remains unclear which representational properties transfer effectively to graph-level topology extraction—a task requiring not only accurate road detection but also correct long-range connectivity. In this work, we systematically evaluate four modern foundation-model backbones—SAM, SAM2.1, DINOv3, and C-RADIOv3—within a unified topology extraction framework. We further investigate whether common architectural augmentations, including Feature Pyramid Networks and attention mechanisms, can improve topological connectivity. Our results reveal two key findings. First, backbone selection has a substantially larger impact on graph connectivity (APLS) than any architectural modification. SAM 2.1 achieves a significant APLS improvement over the prior SAM baseline on the CityScale dataset. This result is consistent with the hypothesis that pre-training objectives emphasizing spatial boundary localization transfer more effectively to topological routing than semantic or multi-teacher distilled objectives. Second, architectural augmentations consistently fail to improve topology metrics, with APLS declining across the evaluated augmentations. We further demonstrate the scalability of the selected configuration on the GlobalScale dataset, showing that the observed gains extend beyond a single benchmark. Our work identifies SAM2.1 as the strongest-performing backbone among the evaluated models for road topology extraction, and our findings suggest that backbone representation plays a larger role than architectural augmentation in determining graph-level performance.*

<img src="https://https://github.com/BYU-FROST-Lab/samroadplus/blob/main/img/main.pdf" width="100%"/>
Overview of our arbitrary backbone encoder-decoder architecture and graph extraction pipeline for comparison purposes. Built upon original - [SAM-Road++](https://github.com/earth-insights/samroadplus) architecture.

</div>

## Installation
Following the cloning of the repo, follow these steps to get your environment set up:

```bash
conda env create -f environment.yml
conda activate samroadplus
conda install -y -c conda-forge go
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### External Dependencies
Clone the following repositories into the root directory:

**1. Segment Anything Model (SAM)**
```bash
mkdir sam && cd sam
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything && pip install -e .
mv segment_anything ../
cd ../../
```

**2. Detectron2**
```bash
git clone https://github.com/facebookresearch/detectron2.git
pip install -e . --no-build-isolation
```

**3. SAM 2**
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```
*(Note: DINOv3 and RADIO dependencies are purely Python-based and are handled automatically by the `requirements.txt` via `timm` and `torch.hub`)*

## Data & Checkpoint Preparation

### Model Checkpoints
- **Vanilla SAM:** Download the ["vit_b" SAM model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file) and place it in the `sam_ckpts/` directory.
- **SAM 2:** Download the `sam2.1_hiera_base_plus.pt` checkpoint and add it to the `sam_ckpts/` folder.
- **DINOv3 & RADIO:** No manual checkpoint downloads are required; they are automatically downloaded and cached by `timm` and `torch.hub` upon first run.
  - **Important for DINOv3:** You will need a [Hugging Face account](https://huggingface.co/) to access the weights. Please ensure you have accepted the model terms on the Hugging Face website for the specific DINOv3 model, and log in via your terminal using `huggingface-cli login` before running the code.

### Datasets
Download the datasets and place them in the root directory:
- **SpaceNet:** [RGB_1.0_meter_full.zip](https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W)
- **CityScale:** [20cities](https://drive.google.com/drive/folders/1FlMcO3Jr8W4qboZUwxgRn6AlYc-AuxQ2)

## Unified Foundation Model Architecture

The architecture has been unified to dynamically support multiple foundation models directly from the `main` branch. You can seamlessly switch between **SAM 1**, **SAM 2**, **DINOv3**, **RADIO**, and **ResNet50** simply by specifying the corresponding configuration file.

**Supported Models & Configurations (Examples):**
- **SAM 1 (Baseline)**: `--config config/cityscale/toponet_vitb_512_cityscale.yaml`
- **SAM 2**: `--config config/cityscale/toponet_sam2_512_cityscale.yaml`
- **DINOv3**: `--config config/cityscale/toponet_dinov3_512_cityscale.yaml`
- **NVIDIA RADIO**: `--config config/cityscale/toponet_radio_512_cityscale.yaml`

The codebase uses a Factory Pattern in `model.py` and `modelinfer.py` to route backbone initialization and feature extraction dynamically based on the configuration file (via the `BACKBONE` or `SAM_VERSION` keys). This removes the need to checkout separate branches for each model.

## Evaluation Pipeline (J-STARS Benchmarking)

To train and evaluate any of the foundation models (e.g., DINOv3 on GlobalScale), follow this 4-step pipeline:

### 1. Train
*Note: The training script (`train.py`) now supports resuming from a checkpoint using the `--resume` flag.*
```bash
python train.py --config config/globalscale/toponet_dinov3_512_globalscale.yaml
```

### 2. Hyperparameter Optimization (Threshold Sweeping)
Extract the optimal thresholds for keypoint, road, and topology extraction on the validation set. Update your config file with these thresholds.
```bash
python test.py --config config/globalscale/toponet_dinov3_512_globalscale.yaml --checkpoint path_to_ckpt
```

### 3. Inference
Generate the predicted graphs.
```bash
python inferencer.py --config config/globalscale/toponet_dinov3_512_globalscale.yaml --checkpoint path_to_ckpt
```

### 4. Metrics Evaluation (APLS & TOPO)
We have implemented a self-contained benchmarking script (`benchmark_eval.py`) that entirely removes the need to use the external Sat2Graph repository for evaluation.
```bash
python benchmark_eval.py --dataset globalscale --graph_dir save/output_dir/graph
```

### Reproducing the Complexity Table
To evaluate the parameter count, FLOPs, latency, and VRAM for all foundation models, run:
```bash
python benchmark_models.py
```

## Acknowledgement
We sincerely appreciate the authors of the following codebases which made this project possible:
- [SAM-Road++](https://github.com/earth-insights/samroadplus) (The core architecture this project builds upon)
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)  
- [SAM_Road](https://github.com/htcr/sam_road) 
- [Sat2Graph](https://github.com/songtaohe/Sat2Graph)
- [SAMed](https://github.com/hitachinsk/SAMed)  
- [Detectron2](https://github.com/facebookresearch/detectron2)  
