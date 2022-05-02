<br />
<p align="center">
  <h1 align="center">VSR: Visual Spatial Reasoning</h1>
  <h3 align="center">A probing benchmark for spatial undersranding of vision-language models.</h3>
  
  <p align="center">  
    <a href="...">arxiv</a>
    ·
    <a href="...">dataset</a>
  </p>
</p>

### 1 Overview

The Visual Spatial Reasoning (VSR) corpus is a collection of caption-image pairs with true/false labels. Each caption describes the spatial relation of two individual objects in the image, and a vision-language model (VLM) needs to juedge whether the caption is correctly describing the image (True) or not (False). Below are a few examples.

_The cat is behind the laptop_.  (True)   |  _The cow is ahead of the person._ (False) | _The cake is at the edge of the dining table._ (True) | _The horse is left of the person._ (False)
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](http://images.cocodataset.org/train2017/000000119360.jpg)  |  ![](http://images.cocodataset.org/train2017/000000080336.jpg) |   ![](http://images.cocodataset.org/train2017/000000261511.jpg) | ![](http://images.cocodataset.org/train2017/000000057550.jpg) 

#### 1.1 Why VSR?
Understanding spatial relations is fundamental to achieve intelligence. Existing vision-language reasoning datasets are great but they compose multiple types of challenges and can thus conflate different sources of error.
The VSR corpus focuses specifically on spatial relations so we can have accurate diagnosis and maximum interpretability.

#### 1.2 What have we found?
Below are baselines' by-relation performances on VSR (random split). 
![](data_annotation/performance_by_relation_random_split_v2.png)
**_More data != better performance._** The relations are sorted by frequencies from left to right. The VLMs' by-relation performances have little correlation with relation frequency, meaning that more training data do not necessarily lead to better performance.

<img align="right" width="320"  src="data_annotation/performance_by_meta_cat_random_split_v2.png"> 

**_Understanding object orientation is hard._** After classifying spatial relations with meta-categories, we can clearly see that all models are at chance level for "orientation"-related relations (such as "facing", "facing away from", "parallel to", etc.).

For more findings and takeways including zero-shot split performance. check out our paper!

### 2 The VSR dataset: Splits, statistics, and meta-data

The VSR corpus, after validation, containing 10,119 data points with high agreement. On top of these, we create two splits (1) random split and (2) zero-shot split. (1) randomly splits all data points into train, development, and test sets. (2) makes sure that tran, development and test sets have no overlap of concepts (i.e., if *dog* is in test set, it is not used for training and development). 


split   |  train | dev | test | total
:------|:--------:|:--------:|:--------:|:--------:
random | 7,083 | 1,012 | 2,024 | 10,119 
zero-shot | 5,440 | 259 | 731 | 6,430

### 3 Baselines: Performance

We test three baselines, all supported in huggingface. They are VisualBERT [(Li et al. 2019)](https://arxiv.org/abs/1908.03557), LXMERT [(Tan and Bansal, 2019)](https://arxiv.org/abs/1908.07490) and ViLT [(Kim et al. 2021)](https://arxiv.org/abs/2102.03334).

model   |  random split | zero-shot
:-------------|:-------------:|:-------------:
*human* | *95.4* | *95.4* 
VisualBERT | 57.4 | 54.0
LXMERT | **72.5** | **63.2**
ViLT | 71.0 | 62.4


### 4 Baselines: How to run?

#### Download images
See `data/` folder's readme.

#### Extract visual embeddings
For VisualBERT and LXMERT, we need to first extract visual embeddings using pre-trained object detectors.
```bash
cd feature_extraction/lxmert
python extract_img_features.py \                                                                                                  
	--img_folder ../../data/images \
	--output_folder_path ../../data/lxmert_features/
```
VisualBERT feature extraction is done similarly by `cd` into `feature_extraction/visualbert`. The feature extraction codes are modified from huggingface examples [here](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing) (for VisualBERT) and [here](https://colab.research.google.com/drive/18TyuMfZYlgQ_nXo-tr8LCnzUaoX0KS-h?usp=sharing) (for LXMERT).

#### Train

VisualBERT:
```bash
CUDA_VISIBLE_DEVICES=2 python train.py \      
		--img_feature_path data/vsr_data_trial/img_features \
		--train_json_path data/vsr_data_trial/test.json \
		--amp \
		--output_dir "tmp/test" \
		--checkpoint_step 10 \
		--epoch 100 \
		--batch_size 32 \
		--learning_rate 2e-5
```
LXMERT:

```
```

ViLT:
```
```


#### Evaluation
```bash
CUDA_VISIBLE_DEVICES=2 python eval.py \ 
		--checkpoint_path tmp/test/checkpoint_iter_140 \
		--img_feature_path data/vsr_data_trial/img_features \
		--test_json_path data/vsr_data_trial/test.json
```

### License
This project is licensed under the Apache-2.0 License.
