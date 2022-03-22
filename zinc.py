from torch.nn import functional as F
from ogb.lsc import PCQM4Mv2Evaluator
from data.zinc import myZINCDataset
from solver import Solver
from parser import parse_args

args = parse_args()
args.num_class= 1
args.loss_fn = F.l1_loss
args.metric = "mae"
args.metric_mode = "min"
args.evaluator = PCQM4Mv2Evaluator() # mae
args.tr_set = myZINCDataset(subset=True, root=args.data_root, split='train')
args.val_set = myZINCDataset(subset=True, root=args.data_root, split='val')
args.tt_set = myZINCDataset(subset=True, root=args.data_root, split='test')
args.node_emb_dim = 64
args.edge_emb_dim = 64
args.edge_dis_emb_dim = 40
args.degree_emb_dim = 64
args.test_outfile = False
args.feat_emb = True

solver = Solver(args)
