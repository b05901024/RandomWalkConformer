from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from rwc import RandomWalkConformer
from data.data_module import myDataModule

class Solver:
    def __init__(self, args):        
        pl.seed_everything(args.seed)
        dm = myDataModule.from_argparse_args(args)

        if args.checkpoint == "":
            model = RandomWalkConformer(
                n_layers=args.n_layers,
                hidden_dim=args.hidden_dim,
                ffn_dim=args.ffn_dim,
                edge_dim=args.edge_dim,
                n_heads=args.n_heads,
                ffn_dropout=args.ffn_dropout,
                attn_dropout=args.attn_dropout,
                conv_dropout=args.conv_dropout,
                peak_lr=args.peak_lr,
                end_lr=args.end_lr,
                warmup_steps=args.warmup_steps,
                total_steps=args.total_steps,
                weight_decay=args.weight_decay,
                evaluator=args.evaluator,
                metric=args.metric,
                loss_fn=args.loss_fn,
                num_class=args.num_class,
                max_hop=args.max_hop,
                win_size=args.win_size,
                kernel_size=args.kernel_size,
                walk_len_tr=args.walk_len_tr,
                walk_len_tt=args.walk_len_tt,
                node_emb_dim=args.node_emb_dim,
                edge_emb_dim=args.edge_emb_dim,
                edge_dis_emb_dim=args.edge_dis_emb_dim,
                degree_emb_dim=args.degree_emb_dim,
                test_outfile=args.test_outfile,
                directed=args.directed,
                feat_emb=args.feat_emb,
            )
        else:
            model = RandomWalkConformer.load_from_checkpoint(
                args.checkpoint,
                n_layers=args.n_layers,
                hidden_dim=args.hidden_dim,
                ffn_dim=args.ffn_dim,
                edge_dim=args.edge_dim,
                n_heads=args.n_heads,
                ffn_dropout=args.ffn_dropout,
                attn_dropout=args.attn_dropout,
                conv_dropout=args.conv_dropout,
                peak_lr=args.peak_lr,
                end_lr=args.end_lr,
                warmup_steps=args.warmup_steps,
                total_steps=args.total_steps,
                weight_decay=args.weight_decay,
                evaluator=args.evaluator,
                metric=args.metric,
                loss_fn=args.loss_fn,
                num_class=args.num_class,
                max_hop=args.max_hop,
                win_size=args.win_size,
                kernel_size=args.kernel_size,
                walk_len_tr=args.walk_len_tr,
                walk_len_tt=args.walk_len_tt,
                node_emb_dim=args.node_emb_dim,
                edge_emb_dim=args.edge_emb_dim,
                edge_dis_emb_dim=args.edge_dis_emb_dim,
                degree_emb_dim=args.degree_emb_dim,
                test_outfile=args.test_outfile,
                directed=args.directed,
                feat_emb=args.feat_emb,
            )
        print("total params:", sum(p.numel() for p in model.parameters()))

        metric = "val_" + args.metric
        dirpath = args.default_root_dir
        checkpoint_callback = ModelCheckpoint(
            dirpath,
            "{epoch:03d}-{" + metric + ":.4f}",
            metric,
            save_last=True,
            mode=args.metric_mode
        )
        if not args.val and not args.test and os.path.exists(
            dirpath + "/last.ckpt"):
            args.resume_from_checkpoint = dirpath + "/last.ckpt"
            print("resume from:", args.resume_from_checkpoint)
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.callbacks.append(checkpoint_callback)
        trainer.callbacks.append(LearningRateMonitor("step"))

        if args.test:
            pprint(trainer.test(model, datamodule=dm))
        elif args.val:
            pprint(trainer.validate(model, datamodule=dm))
        else:
            trainer.fit(model, datamodule=dm)
            