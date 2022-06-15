data_path=""
output_path=""

python main.py \
    --data_path $data_path \
    --output_path $output_path \
    --checkpoint \
    --model su_vit_tiny_patch16_224 \
    --dataset mnist \
    --base_keep_rate 0.7 \
    --uncertainty_keep_rate 0.8 \
    --mse \
    --epoch 20 \
    --batch_size 256 \
    --input_size 224 \

echo "Output path: $output_path"