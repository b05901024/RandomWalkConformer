# Install
```
conda create --name rwc python=3.8
conda activate rwc
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install pytorch-lightning==1.5
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install tensorboard
pip install ogb
pip install rdkit-pypi
```

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
    --precision 16 --num_workers 16 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --val
```
## test
```
python3 zinc.py --hidden_dim 44 --ffn_dim 44 --edge_dim 11 --n_heads 4 \
    --walk_len_tt 150 --data_root $YOUR_DATA_ROOT --batch_size 256 \
    --precision 16 --num_workers 16 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --test
```

# PCQM4Mv2
## train
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --warmup_steps 60000 --total_steps 1000000 --max_epoch 300 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --accumulate_grad_batches 2 --gradient_clip_val 5  --precision 16 \
    --num_workers 16 --default_root_dir $YOUR_ROOT_DIR --gpus 1 \
    --accelerator ddp --peak_lr 1e-3 --weight_decay 0.01
```
## validation
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --precision 16 --num_workers 16 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --val
```
## test
```
python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 512 \
    --precision 16 --num_workers 16 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --test
```

# OGBG-MOLHIV
## train
```
python3 molhiv.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --warmup_steps 6000 --total_steps 100000 --max_epoch 300 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --accumulate_grad_batches 16 --gradient_clip_val 5  --precision 16 \
    --num_workers 8 --default_root_dir $YOUR_ROOT_DIR --gpus 1 \
    --accelerator ddp --peak_lr 1e-3 --weight_decay 0.01 \
    --num_sanity_val_steps 10
```
## validation
```
python3 molhiv.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --precision 16 --num_workers 8 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --num_sanity_val_steps 0 --val
```
## test
```
python3 molhiv.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --precision 16 --num_workers 8 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --num_sanity_val_steps 0 --test
```

# OGBG-MOLPCBA
## train
```
python3 pcba.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --warmup_steps 6000 --total_steps 100000 --max_epoch 300 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --accumulate_grad_batches 32 --gradient_clip_val 5  --precision 16 \
    --num_workers 8 --default_root_dir $YOUR_ROOT_DIR --gpus 1 \
    --accelerator ddp --peak_lr 1e-3 --weight_decay 0.01
```
## validation
```
python3 pcba.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --precision 16 --num_workers 8 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \
    --val
```
## test
```
python3 pcba.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --walk_len_tt 100 --data_root $YOUR_DATA_ROOT --batch_size 8 \
    --precision 16 --num_workers 8 \
    --checkpoint $YOUR_ROOT_DIR/$YOUR_CHECKPOINT --gpus 1 --accelerator ddp \ --test
```