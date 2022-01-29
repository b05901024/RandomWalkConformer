import torch
from torch_geometric.utils import sort_edge_index

def preprocess_item(item):
    n_nodes = item.x.size(0)

    x = single_emb(item.x)

    # must sort by source
    edge_index, edge_attr = sort_edge_index(
        item.edge_index, item.edge_attr, n_nodes) 
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    edge_attr = single_emb(edge_attr)

    adj = torch.zeros([n_nodes, n_nodes], dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    out_degree = adj.long().sum(1).view(-1)
    # to find edge index in walker
    adj_offset = torch.zeros(n_nodes, dtype=torch.long)
    for i in range(n_nodes - 1):
        adj_offset[i + 1] = adj_offset[i] + out_degree[i]

    attn_bias = torch.zeros([n_nodes + 1, n_nodes + 1], 
                            dtype=torch.float16) # virtual node at index 0
    
    item.n_nodes = n_nodes
    item.x = x
    item.edge_index = edge_index
    item.edge_attr = edge_attr
    item.adj = adj
    item.in_degree = adj.long().sum(0).view(-1)
    item.out_degree = out_degree
    item.adj_offset = adj_offset
    item.attn_bias = attn_bias
    return item
    
def single_emb(x):
    """
    prepare data for embedding layer
    """
    f_dim = x.size(1) if len(x.size()) > 1 else 1
    return x + 512*torch.arange(f_dim, dtype=torch.long) + 1 # 0 for padding
