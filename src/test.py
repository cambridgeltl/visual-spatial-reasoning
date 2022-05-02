
from utils import *
from transformers import BertTokenizer, VisualBertModel
import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy

model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("What is the man eating?", return_tensors="pt")

cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
cfg = load_config_and_model_weights(cfg_path)
visual_model = get_model(cfg)



img1 = plt.imread(f'data/val2014/COCO_val2014_000000441814.jpg')
img2 = plt.imread(f'data/val2014/COCO_val2014_000000113113.jpg')

# Detectron expects BGR images
img_bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img_bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
images = [img_bgr1, img_bgr2]

visual_embeds = extract_visual_features(cfg, visual_model, images)
print (len(visual_embeds))
print (visual_embeds[0].shape)
