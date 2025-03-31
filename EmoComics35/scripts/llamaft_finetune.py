# ***************** Fine-Tuning LLMs on Comics dataset *********************** #

# ********** Libraries and GPU *************

import os
import ast
import sys
import json
import torch
import pickle
import subprocess

sys.path.append('../')

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from sklearn.metrics import classification_report
#from utils.post_processing import post_process

try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")


CURRENT_DIR = Path.cwd()
EC35_DIR = CURRENT_DIR.parent
DATASET_DIR = EC35_DIR / "json_datasets"

MMER_DIR = EC35_DIR.parent
LLAMA_FACTORY_DIR = MMER_DIR / "LLaMA-Factory"

BASE_MODEL = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
#LOGGING_DIR = FT_DIR / "training_logs"
OUTPUT_DIR = EC35_DIR / "model_outputs" / f"""ec35_mm_{BASE_MODEL.split("/")[1]}"""
#OUTPUT_DIR = OUTPUT_DIR.as_posix()

print(CURRENT_DIR, EC35_DIR, DATASET_DIR, LLAMA_FACTORY_DIR, BASE_MODEL, OUTPUT_DIR, sep="\n")