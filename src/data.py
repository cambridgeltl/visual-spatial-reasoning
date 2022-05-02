
import os
import cv2
import glob
import json
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_feature_path, json_path, model_type="visualbert", vilt_processor=None): 
        
        
        self.model_type = model_type

        if self.model_type in ["visualbert", "lxmert"]:
            self.img_features = torch.load(os.path.join(img_feature_path, "features.pt"))
            with open(os.path.join(img_feature_path, "names.txt"), "r") as f:
                lines = f.readlines()
            self.img_names = [line.strip() for line in lines]
            self.img_name2index = {}
            for i, name in enumerate(self.img_names):
                self.img_name2index[name] = i # the i-th vector in img_features
        elif self.model_type == "vilt":
            self.imgs = {}
            img_paths = glob.glob(img_feature_path+"/*.jpg")
            print (f"load images...")
            for img_path in tqdm(img_paths):
                img_name = img_path.split("/")[-1]
                #tmp = Image.open(img_path)
                #keep = tmp.copy()
                #tmp.close()
                self.imgs[img_name] = cv2.imread(img_path)

        if self.model_type == "lxmert":
            self.boxes = torch.load(os.path.join(img_feature_path, "boxes.pt"))

        self.vilt_processor = vilt_processor
       
        self.data_json = []
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data_json.append(j_line)
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        if self.model_type == "visualbert":
            img_index = self.img_name2index[data_point["image"]]
            return data_point["caption"], self.img_features[img_index], data_point["label"]
        elif self.model_type == "lxmert":
            img_index = self.img_name2index[data_point["image"]]
            return data_point["caption"], self.boxes[img_index], self.img_features[img_index], data_point["label"]
        elif self.model_type == "vilt":
            """
            try:
                inputs = self.vilt_processor(images=self.imgs[data_point["image"]], text=data_point["caption"], 
                        max_length=32, return_tensors="pt", padding='max_length', truncation=True, 
                        add_special_tokens=True)
            except:
                print (data_point)
                exit()
            """
            return self.imgs[data_point["image"]], data_point["caption"], data_point["label"]
            #return inputs, data_point["label"]

    def __len__(self):
        return len(self.data_json)


