from argparse import ArgumentParser
import pytorch_lightning as pl
from rwc import RandomWalkConformer
from data.data_module import myDataModule

def parse_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = RandomWalkConformer.add_argparse_args(parser)
    parser = myDataModule.add_argparse_args(parser)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    return args
