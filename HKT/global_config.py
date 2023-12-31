import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

acoustic_features_list=list(range(0,74))
visual_features_list=list(range(0,46))
hcf_features_list = list(range(0,4))

ACOUSTIC_DIM = len(acoustic_features_list) 
VISUAL_DIM = len(visual_features_list)
HCF_DIM=len(hcf_features_list)
LANGUAGE_DIM=768

VISUAL_DIM_ALL = 47
ACOUSTIC_DIM_ALL =74
HCF_DIM_ALL=17

H_MERGE_SENT = 768
DATASET_LOCATION = "./dataset/"
SEP_TOKEN_ID = 3