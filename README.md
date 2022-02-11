# ZINC
## train
```
python3 zinc.py --hidden_dim 44 --ffn_dim 44 --edge_dim 11 --n_heads 4 \
    --warmup_steps 40000 --total_steps 400000 --max_epoch 10000 \
    --walk_len_tt 150 --data_root $YOUR_DATA_ROOT --batch_size 256 \
    --gradient_clip_val 5  --precision 16 --num_workers 16 \
    --default_root_dir $YOUR_ROOT_DIR --gpus 1 --accelerator ddp \
    --peak_lr 1e-3 --weight_decay 0.01
```
## validation
```
python3 zinc.py --hidden_dim 44 --ffn_dim 44 --edge_dim 11 --n_heads 4 \
    --walk_len_tt 150 --data_root $YOUR_DATA_ROOT --batch_size 256 \
    --precision 16 --num_workers 16 --default_root_dir $YOUR_ROOT_DIR \
    --gpus 1 --accelerator ddp --val
```
## test
```
python3 zinc.py --hidden_dim 44 --ffn_dim 44 --edge_dim 11 --n_heads 4 \
    --walk_len_tt 150 --data_root $YOUR_DATA_ROOT --batch_size 256 \
    --precision 16 --num_workers 16 --default_root_dir $YOUR_ROOT_DIR \
    --gpus 1 --accelerator ddp --test
```

# PCQM4Mv2
## train
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --warmup_steps 60000 --total_steps 1000000 --max_epoch 300 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --accumulate_grad_batches 2 --gradient_clip_val 5  --precision 16 \
    --num_workers 16 --default_root_dir $YOUR_ROOT_DIR --gpus 1 \
    --accelerator ddp --peak_lr 1e-3 --weight_decay 0
```
## validation
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --precision 16 --num_workers 16 --default_root_dir $YOUR_ROOT_DIR \
    --gpus 1 --accelerator ddp --val
```
## test
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --precision 16 --num_workers 16 --default_root_dir $YOUR_ROOT_DIR \
    --gpus 1 --accelerator ddp --test
```
