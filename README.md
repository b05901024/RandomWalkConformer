# ZINC
## train
```
python3 zinc.py --hidden_dim 44 --ffn_dim 44 --edge_dim 11 --n_heads 4 \
    --warmup_steps 40000 --total_steps 400000 --max_epoch 10000 \
    --walk_len_tt 150 --data_root $YOUR_DATA_ROOT --batch_size 256 \
    --gradient_clip_val 5  --precision 16 --num_workers 16 \
    --default_root_dir $YOUR_ROOT_DIR --gpus 1 --accelerator ddp \
    --peak_lr 2e-4
```
