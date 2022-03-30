import torch
import math
import torch.nn as nn
import pytorch_lightning as pl
from walker import genWalk
from scheduler import LinearWarmupLR
from layers import ConformerBlock

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

class RandomWalkConformer(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        hidden_dim,
        ffn_dim,
        edge_dim,
        n_heads,
        ffn_dropout,
        attn_dropout,
        conv_dropout,
        peak_lr,
        end_lr,
        warmup_steps,
        total_steps,
        weight_decay,
        evaluator,
        metric,
        loss_fn,
        num_class=1,
        max_hop=5,
        win_size=8,
        kernel_size=9,
        walk_len_tr=50,
        walk_len_tt=100,
        node_emb_dim=512 * 9 + 1,
        edge_emb_dim=512 * 3 + 1,
        edge_dis_emb_dim=128,
        degree_emb_dim=512,
        test_outfile=False,
        directed=False,
        feat_emb=True,
    ):
        assert hidden_dim % n_heads == 0, \
               "hidden_dim must be divisible by n_heads"
        super().__init__()
        self.save_hyperparameters()

        if feat_emb:
            self.node_encoder = nn.Embedding(
                    node_emb_dim, hidden_dim, padding_idx=0)
            self.edge_encoder_attn = nn.Embedding(
                edge_emb_dim, n_heads, padding_idx=0)
            self.edge_encoder_conv = nn.Embedding(
                edge_emb_dim, edge_dim, padding_idx=0)
        else:
            self.node_encoder = nn.Linear(node_emb_dim, hidden_dim)
            self.edge_encoder_attn = nn.Linear(edge_emb_dim, n_heads)
            self.edge_encoder_conv = nn.Linear(edge_emb_dim, edge_dim)
        self.edge_dis_encoder = nn.Embedding(
            edge_dis_emb_dim * n_heads * n_heads, 1)
        self.in_degree_encoder = nn.Embedding(
            degree_emb_dim, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            degree_emb_dim, hidden_dim, padding_idx=0)
        self.spatial_encoder_attn = nn.Embedding(
            max_hop + 2, n_heads, padding_idx=0) # max_hop + 1 unreachable
        self.spatial_encoder_conv = nn.Embedding(
            max_hop + 2, 1, padding_idx=0)
        self.vn_encoder = nn.Embedding(1, hidden_dim)
        self.vn_pos_encoder_out = nn.Embedding(1, n_heads)
        if directed:
            self.vn_pos_encoder_in = nn.Embedding(1, n_heads)
        
        self.layers = nn.ModuleList(
            [ConformerBlock(hidden_dim, ffn_dim, edge_dim, n_heads, win_size,
                            kernel_size, ffn_dropout, attn_dropout, 
                            conv_dropout) for i in range(n_layers)]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_class)
        
        self.n_layers       = n_layers
        self.hidden_dim     = hidden_dim
        self.n_heads        = n_heads
        self.win_size       = win_size
        self.max_hop        = max_hop
        self.walk_len       = walk_len_tr
        self.walk_len_tr    = walk_len_tr
        self.walk_len_tt    = walk_len_tt
        self.peak_lr        = peak_lr
        self.end_lr         = end_lr
        self.warmup_steps   = warmup_steps
        self.total_steps    = total_steps
        self.weight_decay   = weight_decay
        self.evaluator      = evaluator
        self.metric         = metric
        self.loss_fn        = loss_fn
        self.test_outfile   = test_outfile
        self.directed       = directed
        self.feat_emb       = feat_emb

        self.apply(lambda module: init_params(module, n_layers))

    def forward(self, batched_data):
        n_nodes     = batched_data.n_nodes
        x           = batched_data.x 
        edge_index  = batched_data.edge_index
        edge_attr   = batched_data.edge_attr
        adj         = batched_data.adj
        in_degree   = batched_data.in_degree
        out_degree  = batched_data.out_degree
        adj_offset  = batched_data.adj_offset
        attn_bias   = batched_data.attn_bias

        # walk_nodes, walk_edges, id_enc, con_enc, s_enc, spatial_pos, \
        #     edge_input = genWalk(
        #         self.walk_len, self.win_size, edge_index, edge_attr, n_nodes,
        #         adj, adj_offset, out_degree, self.n_layers)
        walk_nodes, walk_edges, id_enc, con_enc, s_enc, spatial_pos, \
            edge_input = genWalk(self.walk_len, self.max_hop, self.win_size, 
                    edge_index, edge_attr, n_nodes, adj, adj_offset, 
                    out_degree, self.n_layers, self.directed)
        n_graphs, max_node_num = x.size()[:2]

        # embedding node/ edge features
        if self.feat_emb:
            node_feat = self.node_encoder(x).sum(-2)
            edge_feat = self.edge_encoder_conv(edge_attr).sum(-2)
            edge_input = self.edge_encoder_attn(edge_input).mean(-2)
        else:
            node_feat = self.node_encoder(x)
            edge_feat = self.edge_encoder_conv(edge_attr)
            edge_input = self.edge_encoder_attn(edge_input)

        # attention bias
        attn_bias = attn_bias.repeat(self.n_layers, 1, 1)
        attn_bias[:, 1:, 1:][spatial_pos == self.max_hop + 1] \
            = float('-inf')
        attn_bias_ = attn_bias.clone()
        attn_bias_ = attn_bias_.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1) # b, h, n + 1, n + 1
        spatial_pos_bias = self.spatial_encoder_attn(
            spatial_pos).permute(0, 3, 1, 2) # b, n, n, h -> b, h, n, n
        attn_bias_[:, :, 1:, 1:] = attn_bias_[:, :, 1:, 1:] \
            + spatial_pos_bias
        # virtual node
        theta = self.vn_pos_encoder_out.weight.view(1, self.n_heads, 1)
        attn_bias_[:, :, 0, :] = attn_bias_[:, :, 0, :] + theta
        if self.directed:
            theta = self.vn_pos_encoder_in.weight.view(1, self.n_heads, 1)
        attn_bias_[:, :, 1:, 0] = attn_bias_[:, :, 1:, 0] + theta
        
        # edge
        spatial_pos_ = spatial_pos.clone().half()
        # x > 1 to x - 1
        spatial_pos_ = torch.where(
            spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
        # b, n, n, dis, h
        edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
            self.max_hop, -1, self.n_heads)
        edge_input_flat = torch.bmm(
            edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.n_heads, self.n_heads)[:self.max_hop, :, :])
        edge_input = edge_input_flat.reshape(
                self.max_hop, -1, max_node_num, max_node_num, self.n_heads
            ).permute(1, 2, 3, 0, 4)
        # b, h, n, n
        edge_input = (edge_input.sum(-2) 
                   / spatial_pos_.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn_bias_[:, :, 1:, 1:] = attn_bias_[:, :, 1:, 1:] + edge_input
        attn_bias_ = attn_bias_ + attn_bias.unsqueeze(1) # reset unreachable

        # convolution module encoding
        s_enc = self.spatial_encoder_conv(s_enc).squeeze(-1)
        encodings = torch.cat([id_enc, con_enc, s_enc], 1)
        # encodings = torch.cat([id_enc, con_enc], 1)

        # node feature
        # b, n, d
        node_feat = node_feat \
                  + self.in_degree_encoder(in_degree) \
                  + self.out_degree_encoder(out_degree)
        vn_feat = self.vn_encoder.weight.unsqueeze(0).repeat(n_graphs, 1, 1)
        node_feat = torch.cat([vn_feat, node_feat], 1) # vn at index 0

        for i, layer in enumerate(self.layers):
            node_feat = layer(
                node_feat, attn_bias_[i * n_graphs:(i + 1) * n_graphs], 
                edge_feat, walk_nodes[i * n_graphs:(i + 1) * n_graphs], 
                walk_edges[i * n_graphs:(i + 1) * n_graphs], 
                encodings[i * n_graphs:(i + 1) * n_graphs])
        out = self.ln(node_feat)
        # only use virtual node to represent the graph
        out = self.out(out[:, 0, :])
        return out
        
    def training_step(self, batched_data, batch_idx):
        pred = self(batched_data)
        if self.metric != "acc":
            pred = pred.view(-1)
        gt = batched_data.y.view(-1)
        if self.metric == "rocauc":
            gt = gt.float()
        mask = ~torch.isnan(gt)
        loss = self.loss_fn(pred[mask], gt[mask])
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # set walk length for training
        self.walk_len = self.walk_len_tr
    
    def validation_step(self, batched_data, batch_idx):
        pred = self(batched_data)
        gt = batched_data.y
        return { "pred": pred, "gt": gt }
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        # set walk length for validation
        self.walk_len = self.walk_len_tt
    
    def validation_epoch_end(self, outputs):
        pred = torch.cat([output["pred"] for output in outputs])
        gt = torch.cat([output["gt"] for output in outputs])
        if self.metric == "mae":
            pred = pred.view(-1)
            gt = gt.view(-1)
        elif self.metric == "acc":
            pred = pred.argmax(-1).unsqueeze(-1)
            gt = gt.unsqueeze(-1)
        try:
            self.log(
                "val_" + self.metric, 
                self.evaluator.eval({
                    "y_pred": pred, 
                    "y_true": gt
                })[self.metric], 
                sync_dist=True
            )
        except:
            self.log(
                "val_" + self.metric, 
                float("NaN"), 
                sync_dist=True
            )
    
    def test_step(self, batched_data, batch_idx):
        pred = self(batched_data)
        gt = batched_data.y
        return { "pred": pred, "gt": gt, "idx": batched_data.idx }
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        # set walk length for test
        self.walk_len = self.walk_len_tt

    def test_epoch_end(self, outputs):
        pred = torch.cat([output["pred"] for output in outputs])
        gt = torch.cat([output["gt"] for output in outputs])
        if self.metric == "mae":
            pred = pred.view(-1)
            gt = gt.view(-1)
        elif self.metric == "acc":
            pred = pred.argmax(-1).unsqueeze(-1)
            gt = gt.unsqueeze(-1)
        if self.test_outfile:
            pred = pred.cpu().numpy()
            idx = torch.cat([output["idx"] for output in outputs])
            torch.save(pred, "y_pred.pt")
            torch.save(idx, "idx.pt")
        else:
            self.log(
                "test_" + self.metric, 
                self.evaluator.eval({
                    "y_pred": pred, 
                    "y_true": gt
                })[self.metric], 
                sync_dist=True
            )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), self.peak_lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": LinearWarmupLR(
                optimizer,
                self.warmup_steps,
                self.total_steps,
                self.peak_lr,
                self.end_lr
            ),
            "name": "lr",
            "interval": "step",
            "frequecy": 1,
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("rwc")
        parser.add_argument("--n_layers",       type=int,   default=16)
        parser.add_argument("--hidden_dim",     type=int,   default=256)
        parser.add_argument("--ffn_dim",        type=int,   default=256)
        parser.add_argument("--edge_dim",       type=int,   default=64)
        parser.add_argument("--n_heads",        type=int,   default=8)
        parser.add_argument("--ffn_dropout",    type=float, default=0.1)
        parser.add_argument("--attn_dropout",   type=float, default=0.1)
        parser.add_argument("--conv_dropout",   type=float, default=0.1)
        parser.add_argument("--peak_lr",        type=float, default=1e-3)
        parser.add_argument("--end_lr",         type=float, default=1e-9)
        parser.add_argument("--warmup_steps",   type=int,   default=60000)
        parser.add_argument("--total_steps",    type=int,   default=1000000)
        parser.add_argument("--weight_decay",   type=float, default=0)
        parser.add_argument("--max_hop",        type=int,   default=5)
        parser.add_argument("--win_size",       type=int,   default=8)
        parser.add_argument("--kernel_size",    type=int,   default=9)
        parser.add_argument("--walk_len_tr",    type=int,   default=50)
        parser.add_argument("--walk_len_tt",    type=int,   default=100)
        parser.add_argument('--directed', action='store_true', default=False)
        parser.add_argument('--val', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)  
        return parent_parser
