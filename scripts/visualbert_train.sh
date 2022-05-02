CUDA_VISIBLE_DEVICES=$1 python src/train.py \
	--img_feature_path data/features/visualbert/ \
	--train_json_path data/splits/random/train.jsonl \
	--amp \
	--output_dir "tmp/visualbert_random_split" \
	--checkpoint_step 9999999 \
	--epoch 100 \
	--batch_size 32 \
	--learning_rate 2e-6 \
	--model_path uclanlp/visualbert-nlvr2-coco-pre	\
	--model_type visualbert \
	--val_json_path data/splits/random/dev.jsonl	\
	--eval_step 100

#	--learning_rate 2e-6 \
#	--train_json_path data/label_studio_march_3rd_1420_add_more.json \
		
