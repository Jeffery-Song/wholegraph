# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import os
import time
from optparse import OptionParser

import apex
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics.functional as MF
from apex.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
from ogb.linkproppred import Evaluator
from wg_torch import comm as comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch import graph_ops as graph_ops
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg

parser = OptionParser()
parser.add_option(
    "-r",
    "--root_dir",
    dest="root_dir",
    default="/nvme/songxiaoniu/graph-learning/wholegraph",
    # default="/dev/shm/dataset",
    help="dataset root directory.",
)
parser.add_option(
    "-g", "--graph_name", dest="graph_name", default="ogbl-citation2", help="graph name"
)
parser.add_option(
    "-e", "--epochs", type="int", dest="epochs", default=4, help="number of epochs"
)
parser.add_option(
    "-b", "--batchsize", type="int", dest="batchsize", default=8000, help="batch size"
)
parser.add_option("--skip_epoch", type="int", dest="skip_epoch", default=1, help="num of skip epoch for profile")
parser.add_option("--local_step", type="int", dest="local_step", default=125, help="num of steps on a GPU in an epoch")
parser.add_option(
    "-n",
    "--neighbors",
    dest="neighbors",
    default="10,25",
    help="train neighboor sample count",
)
parser.add_option(
    "--hiddensize", type="int", dest="hiddensize", default=256, help="hidden size"
)
parser.add_option(
    "-l", "--layernum", type="int", dest="layernum", default=2, help="layer number"
)
parser.add_option(
    "-m",
    "--model",
    dest="model",
    default="sage",
    help="model type, valid values are: sage, gcn, gat",
)
parser.add_option(
    "-f",
    "--framework",
    dest="framework",
    default="dgl",
    help="framework type, valid values are: dgl, pyg, wg",
)
parser.add_option("--heads", type="int", dest="heads", default=1, help="num heads")
parser.add_option(
    "-w",
    "--dataloaderworkers",
    type="int",
    dest="dataloaderworkers",
    default=8,
    help="number of workers for dataloader",
)
parser.add_option(
    "-d", "--dropout", type="float", dest="dropout", default=0.5, help="dropout"
)
parser.add_option("--lr", type="float", dest="lr", default=0.003, help="learning rate")
parser.add_option(
    "--use_nccl",
    action="store_true",
    dest="use_nccl",
    default=False,
    help="whether use nccl for embeddings, default False",
)

(options, args) = parser.parse_args()

use_chunked = True
use_host_memory = False


def parse_max_neighbors(num_layer, neighbor_str):
    neighbor_str_vec = neighbor_str.split(",")
    max_neighbors = []
    for ns in neighbor_str_vec:
        max_neighbors.append(int(ns))
    assert len(max_neighbors) == 1 or len(max_neighbors) == num_layer
    if len(max_neighbors) != num_layer:
        for i in range(1, num_layer):
            max_neighbors.append(max_neighbors[0])
    # max_neighbors.reverse()
    return max_neighbors


if options.framework == "dgl":
    import dgl
    from dgl.nn.pytorch.conv import SAGEConv, GATConv
elif options.framework == "pyg":
    from torch_sparse import SparseTensor
    from torch_geometric.nn import SAGEConv, GATConv
elif options.framework == "wg":
    from wg_torch.gnn.SAGEConv import SAGEConv
    from wg_torch.gnn.GATConv import GATConv


def create_gnn_layers(in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head):
    gnn_layers = torch.nn.ModuleList()
    for i in range(num_layer):
        layer_output_dim = (hidden_feat_dim // num_head if i != num_layer - 1 else class_count)        
        layer_input_dim = in_feat_dim if i == 0 else hidden_feat_dim
        mean_output = True if i == num_layer - 1 else False
        if options.framework == "pyg":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == "gat":
                concat = not mean_output
                gnn_layers.append(
                    GATConv(
                        layer_input_dim, layer_output_dim, heads=num_head, concat=concat
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, root_weight=False)
                )
        elif options.framework == "dgl":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "mean"))
            elif options.model == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        allow_zero_in_degree=True,
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim, "gcn"))
        elif options.framework == "wg":
            if options.model == "sage":
                gnn_layers.append(SAGEConv(layer_input_dim, layer_output_dim))
            elif options.model == "gat":
                gnn_layers.append(
                    GATConv(
                        layer_input_dim,
                        layer_output_dim,
                        num_heads=num_head,
                        mean_output=mean_output,
                    )
                )
            else:
                assert options.model == "gcn"
                gnn_layers.append(
                    SAGEConv(layer_input_dim, layer_output_dim, aggregator="gcn")
                )
    return gnn_layers


def create_sub_graph(
    target_gid,
    target_gid_1,
    edge_data,
    csr_row_ptr,
    csr_col_ind,
    sample_dup_count,
    add_self_loop: bool,
):
    if options.framework == "pyg":
        neighboor_dst_unique_ids = csr_col_ind
        neighboor_src_unique_ids = edge_data[1]
        target_neighbor_count = target_gid.size()[0]
        if add_self_loop:
            self_loop_ids = torch.arange(
                0,
                target_gid_1.size()[0],
                dtype=neighboor_dst_unique_ids.dtype,
                device=target_gid.device,
            )
            edge_index = SparseTensor(
                row=torch.cat([neighboor_src_unique_ids, self_loop_ids]).long(),
                col=torch.cat([neighboor_dst_unique_ids, self_loop_ids]).long(),
                sparse_sizes=(target_gid_1.size()[0], target_neighbor_count),
            )
        else:
            edge_index = SparseTensor(
                row=neighboor_src_unique_ids.long(),
                col=neighboor_dst_unique_ids.long(),
                sparse_sizes=(target_gid_1.size()[0], target_neighbor_count),
            )
        return edge_index
    elif options.framework == "dgl":
        if add_self_loop:
            self_loop_ids = torch.arange(
                0,
                target_gid_1.numel(),
                dtype=edge_data[0].dtype,
                device=target_gid.device,
            )
            block = dgl.create_block(
                (
                    torch.cat([edge_data[0], self_loop_ids]),
                    torch.cat([edge_data[1], self_loop_ids]),
                ),
                num_src_nodes=target_gid.size(0),
                num_dst_nodes=target_gid_1.size(0),
            )
        else:
            block = dgl.create_block(
                (edge_data[0], edge_data[1]),
                num_src_nodes=target_gid.size(0),
                num_dst_nodes=target_gid_1.size(0),
            )
        return block
    else:
        assert options.framework == "wg"
        return [csr_row_ptr, csr_col_ind, sample_dup_count]
    return None


def layer_forward(layer, x_feat, x_target_feat, sub_graph):
    if options.framework == "pyg":
        x_feat = layer((x_feat, x_target_feat), sub_graph)
    elif options.framework == "dgl":
        x_feat = layer(sub_graph, (x_feat, x_target_feat))
    elif options.framework == "wg":
        x_feat = layer(sub_graph[0], sub_graph[1], sub_graph[2], x_feat, x_target_feat)
    return x_feat

class DotPredictor(nn.Module):
    def forward(self, h_src, h_dst):
        return (h_src * h_dst).sum(1)

class EdgePredictionGNNModel(torch.nn.Module):
    def __init__(
        self,
        graph: graph_ops.HomoGraph,
        num_layer,
        hidden_feat_dim,
        class_count,
        max_neighbors: str,
        predictor
    ):
        super().__init__()
        self.graph = graph
        self.num_layer = num_layer
        self.hidden_feat_dim = hidden_feat_dim
        self.max_neighbors = parse_max_neighbors(num_layer, max_neighbors)
        self.class_count = class_count
        num_head = options.heads if (options.model == "gat") else 1
        assert hidden_feat_dim % num_head == 0
        in_feat_dim = self.graph.node_feat_shape()[1]
        self.gnn_layers = create_gnn_layers(
            in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head
        )
        self.mean_output = True if options.model == "gat" else False
        self.add_self_loop = True if options.model == "gat" else False
        self.gather_fn = embedding_ops.EmbeddingLookUpModule(need_backward=False)
        self.dropout = nn.Dropout(options.dropout)
        self.predictor = predictor

    def gnn_forward(self, ids, exclude_edge_hashset=None):
        ids = ids.to(self.graph.id_type()).cuda()
        (
            target_gids,
            edge_indice,
            csr_row_ptrs,
            csr_col_inds,
            sample_dup_counts,
        ) = self.graph.unweighted_sample_without_replacement(
            ids, self.max_neighbors, exclude_edge_hashset=exclude_edge_hashset
        )
        x_feat = self.gather_fn(target_gids[0], self.graph.node_feat)
        # x_feat = self.graph.gather(target_gids[0])
        # num_nodes = [target_gid.shape[0] for target_gid in target_gids]
        # print('num_nodes %s' % (num_nodes, ))
        for i in range(self.num_layer):
            x_target_feat = x_feat[: target_gids[i + 1].numel()]
            sub_graph = create_sub_graph(
                target_gids[i],
                target_gids[i + 1],
                edge_indice[i],
                csr_row_ptrs[i],
                csr_col_inds[i],
                sample_dup_counts[i],
                self.add_self_loop,
            )
            x_feat = layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graph)
            if i != self.num_layer - 1:
                if options.framework == "dgl":
                    x_feat = x_feat.flatten(1)
                x_feat = F.relu(x_feat)
                x_feat = self.dropout(x_feat)
        if options.framework == "dgl" and self.mean_output:
            out_feat = x_feat.mean(1)
        else:
            out_feat = x_feat
        return out_feat

    def predict(self, h_src, h_dst):
        return self.predictor(h_src, h_dst)

    def forward(self, src_ids, pos_dst_ids, neg_dst_ids):
        assert src_ids.shape == pos_dst_ids.shape and src_ids.shape == neg_dst_ids.shape
        id_count = src_ids.size(0)
        ids = torch.cat([src_ids, pos_dst_ids, neg_dst_ids])
        # add both forward and reverse edge into hashset
        exclude_edge_hashset = torch.ops.wholegraph.create_edge_hashset(
            torch.cat([src_ids, pos_dst_ids]), torch.cat([pos_dst_ids, src_ids])
        )
        ids_unique, reverse_map = torch.unique(ids, return_inverse=True)
        out_feat_unique = self.gnn_forward(
            ids_unique, exclude_edge_hashset=exclude_edge_hashset
        )
        out_feat = torch.nn.functional.embedding(reverse_map, out_feat_unique)
        src_feat, pos_dst_feat, neg_dst_feat = torch.split(out_feat, id_count)
        scores = self.predict(
            torch.cat([src_feat, src_feat]), torch.cat([pos_dst_feat, neg_dst_feat])
        )
        return scores[:id_count], scores[id_count:]


def train(dist_homo_graph, model, optimizer):
    train_step = 0
    epoch = 0
    train_start_time = time.time()
    while epoch < options.epochs:
        epoch_iter_count = min(options.local_step, dist_homo_graph.start_iter(options.batchsize))
        if comm.get_rank() == 0: print("%d steps for epoch %d." % (epoch_iter_count, epoch))
        iter_id = 0
        while iter_id < epoch_iter_count:
            src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id)
            # neg_dst_nid = torch.randint_like(src_nid, 0, dist_homo_graph.node_count)
            neg_dst_nid = dist_homo_graph.per_source_negative_sample(src_nid)
            optimizer.zero_grad()
            model.train()
            pos_score, neg_score = model(src_nid, pos_dst_nid, neg_dst_nid)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            loss.backward()
            optimizer.step()
            if comm.get_rank() == 0 and train_step % 100 == 0:
                print(
                    "[%s] [LOSS] step=%d, loss=%f"
                    % (
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        train_step,
                        loss.cpu().item(),
                    )
                )
            train_step = train_step + 1
            iter_id = iter_id + 1
        epoch = epoch + 1
    comm.synchronize()
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    if comm.get_rank() == 0:
        print(
            "[%s] [TRAIN_TIME] train time is %.2f seconds"
            % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_time)
        )
        print("[EPOCH_TIME] %.2f seconds" % (train_time / options.epochs,))


def main():
    wg.init_lib()
    torch.set_num_threads(1)
    comma = MPI.COMM_WORLD
    shared_comma = comma.Split_type(MPI.COMM_TYPE_SHARED)
    os.environ["RANK"] = str(comma.Get_rank())
    os.environ["WORLD_SIZE"] = str(comma.Get_size())
    # slurm in Selene has MASTER_ADDR env
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"
    local_rank = shared_comma.Get_rank()
    local_size = shared_comma.Get_size()
    print("Rank=%d, local_rank=%d" % (local_rank, comma.Get_rank()))
    dev_count = torch.cuda.device_count()
    assert dev_count > 0
    assert local_size <= dev_count
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    wm_comm = create_intra_node_communicator(
        comma.Get_rank(), comma.Get_size(), local_size
    )
    wm_embedding_comm = None
    if options.use_nccl:
        if comma.Get_rank() == 0:
            print("Using nccl embeddings.")
        wm_embedding_comm = create_global_communicator(
            comma.Get_rank(), comma.Get_size()
        )
    if comma.Get_rank() == 0:
        print("Framework=%s, Model=%s" % (options.framework, options.model))

    dist_homo_graph = graph_ops.HomoGraph()
    global use_chunked
    global use_host_memory
    dist_homo_graph.load(
        options.root_dir,
        options.graph_name,
        wm_comm,
        use_chunked,
        use_host_memory,
        wm_embedding_comm,
        feat_dtype=None,
        id_dtype=None,
        ignore_embeddings=None,
        link_pred_task=True,
    )
    print("Rank=%d, Graph loaded." % (comma.Get_rank(),))
    
    train_device = torch.device('cuda:{:}'.format(comma.Get_rank()))
    predictor = DotPredictor()
    model = EdgePredictionGNNModel(
        dist_homo_graph, options.layernum, options.hiddensize,
        options.classnum, options.neighbors, predictor
    )
    model.cuda(device=train_device)
    model = DDP(model, delay_allreduce=True)
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=options.lr)
    print("Rank=%d, model and optimizer constructed, begin trainning..." % (comma.Get_rank(),))

    train(dist_homo_graph, model, optimizer)

    wg.finalize_lib()
    print("Rank=%d, wholegraph shutdown." % (comma.Get_rank(),))


if __name__ == "__main__":
    num_class = {
        'reddit' : 41,
        'products' : 47,
        'twitter' : 150,
        'papers100M' : 172,
        'uk-2006-05' : 150,
        'com-friendster' : 100,

        'ogbn-papers100M' : 172,
        'ogbl-citation2' : 172,
    }
    options.classnum = num_class[options.graph_name]
    main()
