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
```
python select_only_revlevant_images.py data_files/all_vsr_validated_data.jsonl/  trainval2017/ images/
```

Alternatively, you could also download the used images from dropbox [here](https://www.dropbox.com/s/0s3bj25s62crjh2/vsr_images.zip?dl=0) and put them under the `images/` folder.

### Splits
`splits/random` and `splits/zeroshot` contain the train and zeroshot splits data in `jsonl` format. `splits/create_random_splits.py` and `splits/create_zeroshot_splits.py` are the scripts used to create them. `splits/create_sample_efficiency_train_splits.py` is for creating sample efficiency training files (100-shot, 10\%, 25\%, 50\%, 75\% of all training data). Read the code and you will see that they are quite self-explanatory.

### Data files
`data_files/` contain the major data collected for creating VSR. `data_files/all_vsr_raw_data.jsonl` contains all 12,809 raw data points and `data_files/all_vsr_validated_data.jsonl` contains the 10,119 data points that passed the second-round validation (and is used for creating the random and zeroshot splits). 

