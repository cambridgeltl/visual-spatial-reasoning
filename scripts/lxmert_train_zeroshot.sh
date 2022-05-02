CUDA_VISIBLE_DEVICES=$1 python src/train.py \
	--img_feature_path data/features/lxmert/ \
	--train_json_path data/splits/zeroshot/train.jsonl \
	--amp \
	--output_dir "tmp/lxmert_zeroshot_split" \
	--checkpoint_step 9999999 \
	--epoch 100 \
	--batch_size 32 \
	--learning_rate 1e-5 \
	--model_path unc-nlp/lxmert-base-uncased \
	--model_type lxmert \
	--val_json_path data/splits/zeroshot/dev.jsonl \
	--eval_step 100

