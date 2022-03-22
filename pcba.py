from torch.nn import functional as F
import ogb
from data.ogb import myGraphPropPredDataset
from solver import Solver
from parser import parse_args
import torch
import numpy as np

args = parse_args()
args.num_class= 128
args.loss_fn = F.binary_cross_entropy_with_logits
args.metric = "ap"
args.metric_mode = "max"
args.evaluator = ogb.graphproppred.Evaluator('ogbg-molpcba')
dataset = myGraphPropPredDataset('ogbg-molpcba', root=args.data_root)
split_idx = dataset.get_idx_split()
args.tr_set = dataset[split_idx["train"]]
args.val_set = dataset[split_idx["valid"]]
args.tt_set = dataset[split_idx["test"]]
args.node_emb_dim = 512 * 9 + 1
args.edge_emb_dim = 512 * 3 + 1
args.edge_dis_emb_dim = 128
args.degree_emb_dim = 512
args.test_outfile = False
args.feat_emb = True

solver = Solver(args)
