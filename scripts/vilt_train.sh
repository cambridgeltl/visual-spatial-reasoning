CUDA_VISIBLE_DEVICES=$1 python src/train.py \
	--img_feature_path data/images/ \
	--train_json_path data/splits/random/train.jsonl \
	--amp \
	--output_dir "tmp/vilt_random_split" \
	--checkpoint_step 9999999 \
	--epoch 30 \
	--batch_size 12 \
	--learning_rate 1e-5 \
	--model_path dandelin/vilt-b32-mlm \
	--model_type vilt \
	--val_json_path data/splits/random/dev.jsonl \
	--eval_step 100
