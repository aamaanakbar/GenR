import tyro
from dataclasses import dataclass
from typing import * 

import sys 
sys.path.append("stylegan3")


@dataclass
class Config:
    name: str = f"restored_samples"
    "A name used to group log files."

    pkl_path: str = "/home/akbar.ali/personal/GencleanAI/pretrained_networks/ffhq_stylegan3.pkl"
    "The location of the pretrained StyleGAN."

    dataset_path: str = "/home/akbar.ali/personal/GencleanAI/datasets/ffhq_resize35"
    "The location of the images to process."
    
    resolution: int = 1024
    "The resolution of your images. Images which are smaller or larger will be resized."

    global_lr_scale: float = 1.0
    "A global factor which scales up and down all learning rates. This may need adjustment for datasets other than faces."

    tasks: Literal["all", "single", "composed", "custom"] = "all"
    "Selects which tasks to run."


def parse_config() -> Config:
    return tyro.cli(Config)



