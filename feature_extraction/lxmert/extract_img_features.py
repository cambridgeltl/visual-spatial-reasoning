
from IPython.display import clear_output, Image, display
import PIL.Image
import io
import glob
import json
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
import wget
import pickle
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='compute LXMERT image rois & features')
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=True)

    args = parser.parse_args()
    
    # load models and model components
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    frcnn_cfg.MODEL.DEVICE='cuda'
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg).cuda()
    
    image_preprocess = Preprocess(frcnn_cfg)
    
    # load images
    #print (f"load images...")
    all_img_paths = glob.glob(args.img_folder + "/*.jpg")
    batch_size = 128
    names, boxes_all, visual_features_all = [], [], []
    for iii in tqdm(np.arange(0, len(all_img_paths), batch_size)):
        all_images, all_sizes, all_scales_yx = [], [], []
        for i, path in enumerate(all_img_paths[iii:iii+batch_size]):
            images, sizes, scales_yx = image_preprocess(path)
            all_images.append(images)
            all_sizes.append(sizes)
            all_scales_yx.append(scales_yx)
            name = path.split("/")[-1]
            names.append(name)

        # run frcnn
        #print (f"run frcnn...")
        for i in range(len(all_images)):
            output_dict = frcnn(
                all_images[i].cuda(), 
                all_sizes[i].cuda(),
                scales_yx=all_scales_yx[i].cuda(),
                padding="max_detections",
                max_detections=frcnn_cfg.max_detections,
                return_tensors="pt",)
        
            #Very important that the boxes are normalized
            normalized_boxes = output_dict.get("normalized_boxes")
            features = output_dict.get("roi_features")

            boxes_all.append(normalized_boxes.detach().cpu())
            visual_features_all.append(features.detach().cpu())

    boxes_all = torch.cat(boxes_all, dim=0)
    visual_features_all = torch.cat(visual_features_all, dim=0)
    print (f"box shape: {boxes_all.shape}, visual feature shape: {visual_features_all.shape}, # names: {len(names)}")
    assert len(all_img_paths) == visual_features_all.shape[0]
    
    # write out
    if not os.path.isdir(args.output_folder_path):
        os.mkdir(args.output_folder_path)
    output_path_txt = os.path.join(args.output_folder_path, "names.txt")
    output_path_feature = os.path.join(args.output_folder_path, "features.pt")
    output_path_box = os.path.join(args.output_folder_path, "boxes.pt")
    # write names to txt
    with open(output_path_txt, "w") as f:
        for name in names:
            f.write(name + "\n")

    # write features to pth
    torch.save(visual_features_all, output_path_feature)
    torch.save(boxes_all, output_path_box)

