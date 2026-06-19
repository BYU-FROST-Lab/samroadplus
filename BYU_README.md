## Installation

Following the cloning of the repo follow these steps to get your environment set up and ready to begin recreating the repo:


```bash
conda env create -f environment.yml
conda activate samroadplus
conda install -y -c conda-forge go
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Download the following (__check official README for where to place files__):

- SAM model checkpoint
    - Vanilla SAM: ["vit_b" SAM model under 'Model Checkpoints'](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file)
    - SAM 2: Download the `sam2.1_hiera_base_plus.pt` checkpoint and add it to the `sam_ckpts` folder (just as you did for vanilla SAM).
    - DINOv3 & RADIO: No manual checkpoint downloads are required! They will be automatically downloaded and cached by `timm` and `torch.hub` upon first run.
- spacenet dataset
    - [RGB_1.0_meter_full.zip](https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W)
- cityscale dataset
    - [20cities](https://drive.google.com/drive/folders/1FlMcO3Jr8W4qboZUwxgRn6AlYc-AuxQ2)
        - Download 'data.zip,' look for and keep '20cities' folder

Clone the following repositories:
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
    - clone inside of 'sam' folder 
        ```bash
        cd sam
        git clone git@github.com:facebookresearch/segment-anything.git
        cd segment-anything; pip install -e .
        ```
    - move 'segment_anything' folder out of 'segment-anything' and into 'sam' for proper functionality
- [Detectron2](https://github.com/facebookresearch/detectron2)
    ```bash
    git clone https://github.com/facebookresearch/detectron2.git
    pip install -e . --no-build-isolation
    ```
- [SAM 2](https://github.com/facebookresearch/sam2)
    ```bash
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .
    cd ..
    ```
    *(Note: DINOv3 and RADIO dependencies are purely Python-based and are handled automatically by the `requirements.txt` via `timm` and `torch.hub`)*

Clone [sam_road](https://github.com/htcr/sam_road) repository but only keep 'config' folder. This folder will be needed when running 'train.py' as described in main README.

## Recent Improvements

### 1. Training & Hyperparameters
- **Checkpoint Resuming**: The training script (`train.py`) now supports resuming from a checkpoint. You can use the `--resume` flag (e.g., `--resume path/to/checkpoint.ckpt`).
- **Early Stopping Disabled**: The restrictive 15-epoch early stopping constraint was removed. The model will now consistently train for the full 250 epochs defined in the configurations.
- **Hardware Optimization**: The `toponet_vitb_512_cityscale.yaml` config has been optimized for high-memory GPUs (e.g. RTX 6000). The batch size was increased from 8 to 32, and the learning rate was linearly scaled from 0.001 to 0.002 to ensure optimal convergence.

### 2. Standalone Benchmarking
We have implemented a self-contained benchmarking script (`benchmark_eval.py`) that entirely removes the need to use the external Sat2Graph repository for evaluation.
- It features a native Python implementation of the TOPO metric (Precision, Recall, F1).
- It runs the APLS metric natively via a Go subprocess (which is why `go` is included in the installation steps).

**How to run evaluation:**
1. First, run the inference script on your checkpoint to generate the graph files:
    ```bash
    python inferencer.py --config config/toponet_vitb_512_cityscale.yaml --checkpoint path/to/model.ckpt --output_dir save/output_my_run
    ```
2. Then, run the standalone benchmark script on the generated graphs:
    ```bash
    python benchmark_eval.py --graph_dir save/save/output_my_run/graph --label my_run
    ```

### 3. Unified Foundation Model Architecture
The architecture has been unified to dynamically support multiple foundation models directly from the `main` branch. You can seamlessly switch between **SAM 1**, **SAM 2**, **DINOv3**, and **RADIO** simply by specifying the corresponding configuration file.

**Supported Models & Configurations:**
- **SAM 1 (Baseline)**: `--config config/toponet_vitb_512_cityscale.yaml`
- **SAM 2**: `--config config/toponet_sam2_512_cityscale.yaml`
- **DINOv3**: `--config config/toponet_dinov3_512_cityscale.yaml`
- **NVIDIA RADIO**: `--config config/toponet_radio_512_cityscale.yaml`

**Additional Dependencies:**
To run the extended foundation models, ensure you have installed the newly appended dependencies in `requirements.txt` (which are already included in the standard install steps):
```bash
pip install timm einops transformers open_clip_torch
```

**Architecture Details:**
The codebase uses a Factory Pattern in `model.py` and `modelinfer.py` to route backbone initialization and feature extraction dynamically based on the configuration file (via the `BACKBONE` or `SAM_VERSION` keys). This removes the need to checkout separate branches for each model.
