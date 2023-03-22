# Data 

### Download images
We use a subset of COCO-2017's train and development images. The following script download COCO-2017's train and val sets images then put them into a single fodler `trainval2017/`.

```bash
cd data/ # enter this folder 
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip
mv train2017 trainval2017 && mv val2017/* trainval2017 && rm -r val2017
```
Copy only relevant images to `images/`.
```bash
mkdir images
python select_only_revlevant_images.py data_files/all_vsr_validated_data.jsonl/  trainval2017/ images/
```

Alternatively, you could also download the used images from dropbox [here](https://www.dropbox.com/s/0s3bj25s62crjh2/vsr_images.zip?dl=0) and put them under the `images/` folder.

### Splits
An overview of the [`splits/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits) folder:
```
splits
├── create_random_splits.py
├── create_sample_efficiency_train_splits.py
├── create_zeroshot_splits.py
├── random
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── train.jsonl
└── zeroshot
    ├── dev.jsonl
    ├── test.jsonl
    └── train.jsonl
```

[`splits/random`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits/random) and [`splits/zeroshot`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits/zeroshot) contain the train and zeroshot splits data in `jsonl` format. [`splits/create_random_splits.py`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits/create_random_splits.py) and [`splits/create_zeroshot_splits.py`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits/create_zeroshot_splits.py) are the scripts used to create them. [`splits/create_sample_efficiency_train_splits.py`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/splits/create_sample_efficiency_train_splits.py) is for creating sample efficiency training files (100-shot, 10\%, 25\%, 50\%, 75\% of all training data). Read the code and you will see that they are quite self-explanatory.

You could also access the jsonl file through huggingface datasets  [[🤗vsr_random]](https://huggingface.co/datasets/cambridgeltl/vsr_random) & [[🤗vsr_zeroshot]](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot):
```python
from datasets import load_dataset

data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
vsr_dataset_random = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
vsr_dataset_zeroshot= load_dataset("cambridgeltl/vsr_zeroshot", data_files=data_files)
```

### Format of the data
Each `jsonl` file is of the following format:
```json
{"image": "000000050403.jpg", "image_link": "http://images.cocodataset.org/train2017/000000050403.jpg", "caption": "The teddy bear is in front of the person.", "label": 1, "relation": "in front of", "annotator_id": 31, "vote_true_validator_id": [2, 6], "vote_false_validator_id": []}
{"image": "000000401552.jpg", "image_link": "http://images.cocodataset.org/train2017/000000401552.jpg", "caption": "The umbrella is far away from the motorcycle.", "label": 0, "relation": "far away from", "annotator_id": 2, "vote_true_validator_id": [], "vote_false_validator_id": [2, 9, 1]}
{"..."}
```
Each line is an individual data point.
`image` denotes name of the image in COCO and `image_link` points to the image on the COCO server (so you can also access directly). `caption` is self-explanatory. `label` being `0` and `1` corresponds to False and True respectively. `relation` records the spatial relation used. `annotator_id` points to the annotator who originally wrote the caption. `vote_true_validator_id` and `vote_false_validator_id` are annotators who voted True or False in the second phase validation.

### Other data files
[`data_files/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files) contain the major data collected for creating VSR. [`data_files/all_vsr_raw_data.jsonl`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/all_vsr_raw_data.jsonl) contains all 12,809 raw data points and [`data_files/all_vsr_validated_data.jsonl`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/all_vsr_validated_data.jsonl) contains the 10,119 data points that passed the second-round validation (and is used for creating the random and zeroshot splits). [`data_files/meta_data.csv`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data/data_files/meta_data.jsonl) contains meta data of annotators.

