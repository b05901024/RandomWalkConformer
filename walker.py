import torch
import numpy as np

MAXINT = np.iinfo(np.int64).max

def genWalk(
    length, max_hop, win_size, edge_index, edge_feat, node_num, adj,
    adj_offset, out_degree, n_layers, directed=False
):
    n_graphs = edge_index.size(0) * n_layers
    max_node_num = adj.size(1)
    device = edge_index.device
    batch_iter = torch.arange(n_graphs, dtype=torch.long, device=device)

    # generate walk for all layers at once
    edge_index_rp = repeat(edge_index, n_layers)
    if edge_feat != None:
        edge_feat_rp = repeat(edge_feat, n_layers)
    node_num_rp = repeat(node_num, n_layers)
    adj_rp = repeat(adj, n_layers)
    adj_offset_rp = repeat(adj_offset, n_layers)
    out_degree_rp = repeat(out_degree, n_layers)

    # length at dim 0 is easier for generating walk
    walk_nodes = torch.zeros([length + 1, n_graphs], 
                             dtype=torch.long, device=device)
    walk_edges = torch.zeros([length, n_graphs],
                             dtype=torch.long, device=device)
    s_enc = torch.zeros([length + 1, win_size, n_graphs], 
                        device=device, dtype=torch.long)
    # initial all nodes to be unreachable
    spatial_pos = (max_hop + 1) \
                * torch.ones([n_graphs, max_node_num, max_node_num], 
                             dtype=torch.long, device=device)
    if edge_feat != None:
        edge_input = torch.zeros(
            [n_graphs, max_node_num, max_node_num, max_hop, 
             edge_feat.size(-1)], 
            dtype=edge_feat.dtype, device=device)
    else:
        edge_input = None
    # generate all random numbers at once
    choices = torch.randint(0, MAXINT, [length + 1, n_graphs], device=device)
    
    # generate walk nodes, walk edges, identity and connectivity encoding
    walk_nodes[0] = choices[0] % node_num_rp    # start node
    # if node with out degree = 0 is selected, choose 0 to start
    nodes_degree_filter = torch.gather(out_degree_rp, 1, 
                                       walk_nodes[0].unsqueeze(-1)) != 1
    walk_nodes[0] = walk_nodes[0] * nodes_degree_filter.squeeze(-1)
    walk_nodes[1], walk_edges[0], _, _ = get_next_node(
        walk_nodes[0].unsqueeze(-1), out_degree_rp, 
        adj_offset_rp, choices[1].unsqueeze(-1), edge_index_rp)
    for i in range(1, length):
        next_node, chosen_edge, node_degree, node_adj_offset = \
            get_next_node(walk_nodes[i].unsqueeze(-1), out_degree_rp, 
                          adj_offset_rp, choices[i + 1].unsqueeze(-1), 
                          edge_index_rp)
        # non-backtracking
        chosen_edge += walk_nodes[i - 1] == next_node
        chosen_edge = (chosen_edge - node_adj_offset) \
                    % torch.clamp(node_degree, min=1) + node_adj_offset
        walk_nodes[i + 1] = torch.gather(
            edge_index_rp[:, 1], 1, chosen_edge.unsqueeze(-1)).squeeze(-1)
        walk_edges[i] = chosen_edge
    walk_nodes_t = walk_nodes.T
    walk_edges = walk_edges.T
    
    # random walk spatial position
    for d in range(max_hop, -1, -1):
        for i in range(length - d + 1):
            spatial_pos[batch_iter, walk_nodes_t[:, i], walk_nodes_t[:, i + d]] \
                = d + 1
            if not directed:
                spatial_pos[batch_iter, walk_nodes_t[:, i + d], walk_nodes_t[:, i]] \
                    = d + 1

    # spatial encoding for convolution module
    for i in range(length):
        l = min(i + 1, win_size)
        s_enc[i + 1, win_size - l:] = spatial_pos[
            batch_iter, walk_nodes[i + 1], walk_nodes[i + 1 - l:i + 1]]
    
    # random walk edge input
    if edge_feat != None:
        for l in range(max_hop, 0, -1):
            for i in range(length + 1 - l):
                edge_input[batch_iter, walk_nodes_t[:, i], 
                        walk_nodes_t[:, i + l], :l] \
                    = edge_feat_rp.gather(
                        1, walk_edges[:, i:i + l].unsqueeze(-1)) + 1
                edge_input[batch_iter, walk_nodes_t[:, i], 
                        walk_nodes_t[:, i + l], l:] = 0
                if not directed:
                    edge_input[batch_iter, walk_nodes_t[:, i + l], 
                            walk_nodes_t[:, i], :l] \
                        = edge_feat_rp.gather(
                            1, walk_edges[:, i:i + l].flip(1).unsqueeze(-1)
                            ) + 1
                    edge_input[batch_iter, walk_nodes_t[:, i + l], 
                            walk_nodes_t[:, i], l:] = 0
    else:
        walk_edges = None

    # l, w, b -> b, w, l
    s_enc = s_enc.permute(2, 1, 0)
    return walk_nodes_t, walk_edges, s_enc, spatial_pos, edge_input

def repeat(x, n_layers):
    return torch.cat([x for i in range(n_layers)])

def get_next_node(cur_node, out_degree, adj_offset, choice, edge_index):
    # out degree 0 was set to 1
    node_degree = torch.gather(out_degree, 1, cur_node) - 1
    node_adj_offset = torch.gather(adj_offset, 1, cur_node)
    chosen_edge = choice % torch.clamp(node_degree, min=1) + node_adj_offset
    next_node = torch.gather(edge_index[:, 1], 1, chosen_edge)
    return next_node.squeeze(-1), chosen_edge.squeeze(-1), \
           node_degree.squeeze(-1), node_adj_offset.squeeze(-1)
