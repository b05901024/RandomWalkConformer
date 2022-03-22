from torch.nn import functional as F
from ogb.graphproppred import Evaluator
from data.benchmarkGNN import myGNNBenchmarkDataset
from solver import Solver
from parser import parse_args

args = parse_args()
args.num_class= 10
args.loss_fn = F.cross_entropy
args.metric = "acc"
args.metric_mode = "max"
args.max_node = 150
args.evaluator = Evaluator('ogbg-ppa') # acc
args.tr_set = myGNNBenchmarkDataset(name="CIFAR10", root=args.data_root, 
                                    split='train')
args.val_set = myGNNBenchmarkDataset(name="CIFAR10", root=args.data_root, 
                                     split='val')
args.tt_set = myGNNBenchmarkDataset(name="CIFAR10", root=args.data_root, 
                                    split='test')
args.node_emb_dim = 3
args.edge_emb_dim = 1
args.edge_dis_emb_dim = 40
args.degree_emb_dim = 64
args.test_outfile = False
args.feat_emb = False

solver = Solver(args)
