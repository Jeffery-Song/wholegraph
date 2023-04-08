import dgl
from dgl.heterograph import DGLBlock

def pad(target, unit):
    return ((target + unit - 1) // unit) * unit

def pad_on_demand(target, unit, demand=True):
    if not demand:
        return target
    return pad(target, unit)

def do_sample(ids, dist_homo_graph, run_config):
    graph_block = dist_homo_graph.unweighted_sample_without_replacement(ids, run_config["max_neighbors"])
    return graph_block

def get_input_keys(graph_block):
    (target_gids, _, _, _, _, ) = graph_block
    return target_gids[0]