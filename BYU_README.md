## Installation

Following the cloning of the repo follow these steps to get your environment set up and ready to begin recreating the repo:


```bash
conda env create -f environment.yml
conda activate samroadplus
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Download the following (__check official README for where to place files__):

- SAM model checkpoint
    - ["vit_b" SAM model under 'Model Checkpoints'](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file)
- spacenet dataset
    - [RGB_1.0_meter_full.zip](https://drive.google.com/uc?id=1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W)
- cityscale dataset
    - [20cities](https://drive.google.com/drive/folders/1FlMcO3Jr8W4qboZUwxgRn6AlYc-AuxQ2)
        - Download 'data.zip,' look for and keep '20cities' folder

