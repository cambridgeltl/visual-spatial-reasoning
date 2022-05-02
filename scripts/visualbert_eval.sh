CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
	--checkpoint_path tmp/visualbert_random_split/best_checkpoint \
	--img_feature_path data/features/visualbert/ \
	--test_json_path data/splits/random/test.jsonl \
	--model_type visualbert \
	--output_preds
