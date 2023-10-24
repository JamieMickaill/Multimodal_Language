import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

visual_features_list=list(range(0,72))
acoustic_features_list=list(range(0,44))
hcf_features_list = list(range(0,4))

ACOUSTIC_DIM = 74
VISUAL_DIM = 47
HCF_DIM=len(hcf_features_list)
LANGUAGE_DIM=768

VISUAL_DIM_ALL = 74
ACOUSTIC_DIM_ALL =47
HCF_DIM_ALL=17

H_MERGE_SENT = 768
DATASET_LOCATION = "./dataset/"
SEP_TOKEN_ID = 3