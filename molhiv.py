from torch.nn import functional as F
import ogb
from data.ogb import myGraphPropPredDataset
from solver import Solver
from parser import parse_args
import torch
import numpy as np

args = parse_args()
args.num_class= 1
args.loss_fn = F.binary_cross_entropy_with_logits
args.metric = "rocauc"
args.metric_mode = "max"
args.evaluator = ogb.graphproppred.Evaluator('ogbg-molhiv')
args.fp = torch.from_numpy(np.load(args.fp_path)) if args.fp_path != "" \
    else None
dataset = myGraphPropPredDataset('ogbg-molhiv', root=args.data_root, 
                                 fp=args.fp)
split_idx = dataset.get_idx_split()
args.tr_set = dataset[split_idx["train"]]
args.val_set = dataset[split_idx["valid"]]
args.tt_set = dataset[split_idx["test"]]
args.node_emb_dim = 512 * 9 + 1
args.edge_emb_dim = 512 * 3 + 1
args.edge_dis_emb_dim = 128
args.degree_emb_dim = 512
args.test_outfile = False

solver = Solver(args)
