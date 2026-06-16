dataset=unc
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --data_root ./ln_data/ \
 --batch_size 2 --lr 0.0001 --num_workers=12 \
 --output_dir ./outputs/$dataset \
 --dataset $dataset --max_query_len 20 \
 --aug_crop --aug_scale --aug_translate \
 --lr_drop 60