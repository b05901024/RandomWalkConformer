import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        dropout=0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        n_heads, 
        dropout=0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.attn_dim = hidden_dim // n_heads
        self.scale = self.attn_dim ** -0.5

        self.ln = nn.LayerNorm(hidden_dim)
        self.q = nn.Linear(hidden_dim, n_heads * self.attn_dim)
        self.k = nn.Linear(hidden_dim, n_heads * self.attn_dim)
        self.v = nn.Linear(hidden_dim, n_heads * self.attn_dim)
        self.out = nn.Linear(n_heads * self.attn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_bias, walk_nodes, n_nodes):
        bs = x.size(0)

        x = self.ln(x)
        q = self.q(x).view(bs, -1, self.n_heads, self.attn_dim)
        k = self.k(x).view(bs, -1, self.n_heads, self.attn_dim)
        v = self.v(x).view(bs, -1, self.n_heads, self.attn_dim)

        counts = torch.zeros([bs, x.size(1)], device=x.device)
        for i in range(bs):
            _, count = walk_nodes[i].unique(return_counts=True)
            counts[i, :count.size(0)] += count
        weight = counts / walk_nodes.size(1) * n_nodes.unsqueeze(1)
        v = v * weight.view(bs, -1, 1, 1)

        q = q.transpose(1, 2)       # b, h, n, f
        k = k.permute(0, 2, 3, 1)   # b, h, f, n
        v = v.transpose(1, 2)       # b, h, n, f

        attn = torch.matmul(q, k) * self.scale  # b, h, n, n
        attn = attn + attn_bias
        attn = torch.softmax(attn, -1)

        out = torch.matmul(attn, v)             # b, h, n, f
        out = out.transpose(1, 2).contiguous()  # b, n, h, f
        out = out.view(bs, -1, self.n_heads * self.attn_dim)
        out = out
        out = self.out(out)
        out = self.dropout(out)
        return out

class ConvModule(nn.Module):
    def __init__(
        self,
        hidden_dim,
        edge_dim,
        win_size,
        kernel_size,
        dropout=0
    ):
        super().__init__()
        # select center of walk nodes
        self.center_start = kernel_size // 2
        self.center_end = -(kernel_size - 1 - self.center_start)

        self.ln = nn.LayerNorm(hidden_dim)
        # node feature:             hidden_dim
        # edge feature (in/ out):   edge_dim * 2
        # spatial encoding:         win_size
        self.pc1 = nn.Conv1d(
                hidden_dim + 2 * edge_dim + win_size,
                hidden_dim * 2, 1)  # pointwise
        self.glu = GLU(1)
        self.dc = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, 
            bias=False) # depthwise
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.swish = Swish()
        self.pc2 = nn.Conv1d(hidden_dim, hidden_dim, 1) # pointwise
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_feat, walk_nodes, walk_edges, encodings):
        bs, n, _ = x.size()
        device = x.device
        batch_iter = torch.arange(bs, device=device).unsqueeze(-1)

        x = self.ln(x)
        vn = x[:, 0]

        # b, n, f -> b, f, n
        x = x[batch_iter, walk_nodes + 1].permute(0, 2, 1)  # start from 1
        if edge_feat != None:
            walk_edge_feat = edge_feat[batch_iter, walk_edges].permute(
                0, 2, 1)
            pad_edge_feat = torch.zeros(
                [bs, walk_edge_feat.size(1), 1], device=device)
            walk_edge_feat_i = torch.cat([pad_edge_feat, walk_edge_feat], 2)
            walk_edge_feat_o = torch.cat([walk_edge_feat, pad_edge_feat], 2)
            
            x = torch.cat([x, walk_edge_feat_i, walk_edge_feat_o, encodings],
                          1)
        else:
            x = torch.cat([x, encodings], 1)
        x = self.pc1(x)
        x = self.glu(x)
        x = self.dc(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pc2(x)
        x = x.permute(0, 2, 1)  # b, n, f

        # update by index
        center_nodes = walk_nodes[:, self.center_start:self.center_end]
        node_out = scatter_mean(x, center_nodes, 1, dim_size=n - 1)
        vn_out = self.mlp(vn + node_out.sum(1)).unsqueeze(1) # update vn
        x = torch.cat([vn_out, node_out + vn_out], 1) # update nodes by vn
        x = self.dropout(x)
        return x

class ConformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ffn_dim,
        edge_dim,
        n_heads,
        win_size,
        kernel_size,
        ffn_dropout=0,
        attn_dropout=0,
        conv_dropout=0,
    ):
        super().__init__()
        self.ff1 = FeedForward(hidden_dim, ffn_dim, ffn_dropout)
        self.attn = MultiHeadAttention(hidden_dim, n_heads, attn_dropout)
        self.conv = ConvModule(
                hidden_dim, edge_dim, win_size, kernel_size, conv_dropout)
        self.ff2 = FeedForward(hidden_dim, ffn_dim, ffn_dropout)
        self.ln = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, x, attn_bias, edge_feat, walk_nodes, walk_edges, encodings,
        n_nodes
    ):
        x = self.ff1(x) * 0.5 + x
        x = self.attn(x, attn_bias, walk_nodes, n_nodes) + x
        x = self.conv(x, edge_feat, walk_nodes, walk_edges, encodings) + x
        x = self.ff2(x) * 0.5 + x
        x = self.ln(x)
        return x
        