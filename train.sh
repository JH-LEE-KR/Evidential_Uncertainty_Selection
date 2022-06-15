data_path="/local_datasets/"
output_path="./output/"

python main.py \
    --data_path $data_path \
    --output_path $output_path \
    --model su_vit_tiny_patch16_224 \
    --dataset cifar10 \
    --base_keep_rate 0.7 \
    --uncertainty_keep_rate 0.8 \
    --mse \
    --epoch 1 \
    --batch_size 256 \
    --input_size 224 \

echo "Output path: $output_path"