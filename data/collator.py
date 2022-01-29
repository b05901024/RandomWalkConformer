import torch

class Batch:
    def __init__(self, idx, n_nodes, x, y, edge_index, edge_attr, adj, 
                 in_degree, out_degree, adj_offset, attn_bias):
        self.idx        = idx
        self.n_nodes    = n_nodes
        self.x          = x
        self.y          = y
        self.edge_index = edge_index
        self.edge_attr  = edge_attr
        self.adj        = adj
        self.in_degree  = in_degree
        self.out_degree = out_degree
        self.adj_offset = adj_offset
        self.attn_bias  = attn_bias

    def to(self, device):
        self.idx        = self.idx.to(device)
        self.n_nodes    = self.n_nodes.to(device)
        self.x          = self.x.to(device)
        self.y          = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr  = self.edge_attr.to(device)
        self.adj        = self.adj.to(device)
        self.in_degree  = self.in_degree.to(device)
        self.out_degree = self.out_degree.to(device)
        self.adj_offset = self.adj_offset.to(device)
        self.attn_bias  = self.attn_bias.to(device)
        return self
    
    def __len__(self):
        return self.idx.size(0)

def collate_fn(items):
    items = [(item.idx, item.n_nodes, item.x, item.y, item.edge_index,
              item.edge_attr, item.adj, item.in_degree, item.out_degree,
              item.adj_offset, item.attn_bias) for item in items]
    idxs, node_nums, xs, ys, edge_indices, edge_attrs, adjs, in_degrees, \
        out_degrees, adj_offsets, attn_biases = zip(*items)
    
    max_node_num = max(n for n in node_nums)
    max_edge_num = max(e.size(0) for e in edge_attrs)

    x = torch.cat([pad_2d(i, max_node_num) for i in xs])
    y = torch.cat(ys)
    edge_index = torch.cat(
        [pad_edge_index(i, max_edge_num) for i in edge_indices])
    edge_attr = torch.cat([pad_2d(i, max_edge_num) for i in edge_attrs])
    adj = torch.cat([pad_adj(i, max_node_num) for i in adjs])
    in_degree = torch.cat(
        [pad_1d_with_padding(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat(
        [pad_1d_with_padding(i, max_node_num) for i in out_degrees])
    adj_offset = torch.cat([pad_1d(i, max_node_num) for i in adj_offsets])
    attn_bias = torch.cat(
        [pad_attn_bias(i, max_node_num + 1) for i in attn_biases])
    return Batch(
        idx=torch.LongTensor(idxs), 
        n_nodes=torch.LongTensor(node_nums), 
        x=x, 
        y=y, 
        edge_index=edge_index, 
        edge_attr=edge_attr, 
        adj=adj, 
        in_degree=in_degree, 
        out_degree=out_degree, 
        adj_offset=adj_offset, 
        attn_bias=attn_bias
    )

def pad_1d(x, pad_len):
    x_len = x.size(0)
    if x_len < pad_len:
        x_pad = torch.zeros([pad_len], dtype=x.dtype)
        x_pad[:x_len]
        x = x_pad
    return x.unsqueeze(0)

def pad_1d_with_padding(x, pad_len):
    """
    start from 1
    """
    x = x + 1
    return pad_1d(x, pad_len)

def pad_2d(x, pad_len):
    """
    start from 1
    """
    x_len, x_dim = x.size()
    x = x + 1
    if x_len < pad_len:
        x_pad = torch.zeros([pad_len, x_dim], dtype=x.dtype)
        x_pad[:x_len, :] = x
        x = x_pad
    return x.unsqueeze(0)

def pad_edge_index(x, pad_len):
    x_len = x.size(1)
    if x_len < pad_len:
        x_pad = torch.zeros([2, pad_len], dtype=x.dtype)
        x_pad[:, :x_len] = x
        x = x_pad
    return x.unsqueeze(0)

def pad_adj(x, pad_len):
    x_len = x.size(0)
    if x_len < pad_len:
        x_pad = torch.zeros([pad_len, pad_len], dtype=x.dtype)
        x_pad[:x_len, :x_len] = x
        x = x_pad
    return x.unsqueeze(0)

def pad_attn_bias(x, pad_len):
    x_len = x.size(0)
    if x_len < pad_len:
        x_pad = torch.zeros(
            [pad_len, pad_len], dtype=x.dtype).fill_(float("-inf"))
        x_pad[:x_len, :x_len] = x
        x_pad[x_len:, :x_len] = 0
        x = x_pad
    return x.unsqueeze(0)
