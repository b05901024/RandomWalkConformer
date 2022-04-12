from torch.nn import functional as F
from ogb.graphproppred import Evaluator
from data.tuDataset import myTUDataset
from solver import Solver
from parser import parse_args
from sklearn.model_selection import StratifiedKFold
import numpy as np

args = parse_args()
args.num_class= 2
args.loss_fn = F.cross_entropy
args.metric = "acc"
args.metric_mode = "max"
args.max_node = 128
args.evaluator = Evaluator('ogbg-ppa') # acc
dataset = myTUDataset(name="NCI1", root=args.data_root)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

args.node_emb_dim = 37
args.edge_emb_dim = 0
args.edge_dis_emb_dim = 0
args.degree_emb_dim = 64
args.test_outfile = False
args.feat_emb = False

ys = [data.y.item() for data in dataset]
idx_list = []
for idx in skf.split(np.zeros(len(ys)), ys):
    idx_list.append(idx)
default_root_dir = args.default_root_dir
for fold in range(10):
    args.default_root_dir = default_root_dir + str(fold)
    tr_idx, val_idx = idx_list[fold]
    args.tr_set = dataset[tr_idx]
    args.val_set = dataset[val_idx]
    solver = Solver(args)
