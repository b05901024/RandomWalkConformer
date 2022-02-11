from torch.nn import functional as F
from ogb.lsc import PCQM4Mv2Evaluator
from data.ogb import myPygPCQM4Mv2Dataset
from solver import Solver
from parser import parse_args

args = parse_args()
args.num_class= 1
args.loss_fn = F.l1_loss
args.metric = "mae"
args.metric_mode = "min"
args.evaluator = PCQM4Mv2Evaluator() # mae

dataset = myPygPCQM4Mv2Dataset(root=args.data_root)
split_idx = dataset.get_idx_split()
args.tr_set = dataset[split_idx["train"]]
args.val_set = dataset[split_idx["valid"]]
args.tt_set = dataset[split_idx["test-dev"]]
# args.tt_set = dataset[split_idx["test-challenge"]]
args.node_emb_dim = 512 * 9 + 1
args.edge_emb_dim = 512 * 3 + 1
args.edge_dis_emb_dim = 128
args.degree_emb_dim = 512
args.test_outfile = True

solver = Solver(args)
