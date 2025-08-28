CUDA_VISIBLE_DEVICES=0,1 python test/test_wllava.py \
--pretrained_model_name_or_path="preset/models/stable-diffusion-3.5-medium" \
--transformer_model_name_or_path="preset/models/dit4sr_q" \
--image_path preset/datasets/dataset_testqq/input_x12/input_x1203 \
--output_dir results/ \