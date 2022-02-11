python3 PCQM4Mv2.py --hidden_dim 256 --ffn_dim 256 --edge_dim 64 --n_heads 8 \
    --warmup_steps 60000 --total_steps 1000000 --max_epoch 300 \
    --walk_len_tt 100 --data_root /media/disk1/peikaiyeh --batch_size 256 \
    --accumulate_grad_batches 4 --gradient_clip_val 5  --precision 16 \
    --num_workers 16 --default_root_dir  /media/disk1/peikaiyeh/log/PCQM4M-LSCv2/rwc_id_con_directed --gpus 1 --accelerator ddp \
    --peak_lr 2e-4 --weight_decay 0
