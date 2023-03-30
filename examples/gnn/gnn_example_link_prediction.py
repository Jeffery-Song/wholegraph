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

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# import torchmetrics.functional as MF
from apex.parallel import DistributedDataParallel as DDP
import dgl.multiprocessing as mp
# from ogb.linkproppred import Evaluator
from wg_torch import comm as comm
from wg_torch import embedding_ops as embedding_ops
from wg_torch import graph_ops as graph_ops
from wg_torch.wm_tensor import *

from wholegraph.torch import wholegraph_pytorch as wg

parser = OptionParser()
parser.add_option(
    "-c", "--num_workers", type="int", dest="num_workers", default=8, help="number of workers"
)
parser.add_option(
    "-r",
    "--root_dir",
    dest="root_dir",
    default="/nvme/songxiaoniu/graph-learning/wholegraph",
    # default="/dev/shm/dataset",
    help="dataset root directory.",
)
parser.add_option(
    # "-g", "--graph_name", dest="graph_name", default="ogbl-citation2", help="graph name"
    "-g", "--graph_name", dest="graph_name", default="ogbn-papers100M", help="graph name"
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
parser.add_option(
    "--amp-off",
    action="store_false",
    dest="use_amp",
    default=True,
    help="whether use amp for training, default True",
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
        options.max_neighbors = self.max_neighbors
        self.class_count = class_count
        num_head = options.heads if (options.model == "gat") else 1
        assert hidden_feat_dim % num_head == 0
        in_feat_dim = self.graph.node_feat_shape()[1]
        self.gnn_layers = create_gnn_layers(
            in_feat_dim, hidden_feat_dim, class_count, num_layer, num_head
        )
        self.mean_output = True if options.model == "gat" else False
        self.add_self_loop = True if options.model == "gat" else False
        options.add_self_loop = self.add_self_loop
        self.dropout = nn.Dropout(options.dropout)
        self.predictor = predictor

    def gnn_forward(self, sub_graphs, target_gid_cnt, x_feat):
        for i in range(self.num_layer):
            x_target_feat = x_feat[: target_gid_cnt[i]]
            x_feat = layer_forward(self.gnn_layers[i], x_feat, x_target_feat, sub_graphs[i])
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

    def forward(self, id_count, reverse_map, sub_graphs, target_gid_cnt, x_feat):
        out_feat_unique = self.gnn_forward(sub_graphs, target_gid_cnt, x_feat)
        out_feat = torch.nn.functional.embedding(reverse_map, out_feat_unique)
        src_feat, pos_dst_feat, neg_dst_feat = torch.split(out_feat, id_count)
        scores = self.predict(
            torch.cat([src_feat, src_feat]), torch.cat([pos_dst_feat, neg_dst_feat])
        )
        return scores


def train(dist_homo_graph, model, optimizer):
    # directly enumerate train_dataloader
    torch.cuda.synchronize()
    test_start_time = time.time()
    max_total_local_steps = dist_homo_graph.start_iter(options.batchsize)
    assert(max_total_local_steps > options.local_step * options.epochs)

    gather_fn = embedding_ops.EmbeddingLookUpModule(need_backward=False).cuda()   
    scaler = GradScaler()
   
    # estimate enumerate overhead
    torch.cuda.synchronize()
    iter_start_time = time.time()
    for iter_id in range(options.local_step):
        src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id)
    torch.cuda.synchronize()
    test_end_time = time.time()
    print(
        "!!!!dist_homo_graph enumerate latency per epoch: %f, per step: %f(start_iter latency %f)"
        % ((test_end_time - iter_start_time), ((test_end_time - iter_start_time) / options.local_step), (iter_start_time - test_start_time))
    )

    latency_s = 0
    latency_e = 0
    latency_t = 0
    sample_node_cnt = 0
    profile_steps = (options.epochs - options.skip_epoch) * options.local_step
    labels = torch.cat([torch.ones(options.batchsize), torch.zeros(options.batchsize)]).cuda()
    
    torch.cuda.synchronize()
    train_start_time = time.time()
    for epoch in range(options.epochs):
        # skip epoch
        if epoch == options.skip_epoch:
            torch.cuda.synchronize()
            skip_epoch_time = time.time()
            latency_s = 0
            latency_e = 0
            latency_t = 0
        if options.worker_id == 0: print("%d steps for epoch %d." % (options.local_step, epoch))
        
        for iter_id in range(options.local_step):
            torch.cuda.synchronize()
            step_start_time = time.time()

            # get pos edges and neg edges
            src_nid, pos_dst_nid = dist_homo_graph.get_train_edge_batch(iter_id + epoch * options.local_step)
            neg_dst_nid = dist_homo_graph.per_source_negative_sample(src_nid)
            assert src_nid.shape == pos_dst_nid.shape and src_nid.shape == neg_dst_nid.shape
            id_count = src_nid.size(0)
            ids = torch.cat([src_nid, pos_dst_nid, neg_dst_nid])
            # add both forward and reverse edge into hashset
            exclude_edge_hashset = torch.ops.wholegraph.create_edge_hashset(
                torch.cat([src_nid, pos_dst_nid]), torch.cat([pos_dst_nid, src_nid])
            )
            ids_unique, reverse_map = torch.unique(ids, return_inverse=True)

            # sample
            ids_unique = ids_unique.to(dist_homo_graph.id_type()).cuda()
            graph_block = dist_homo_graph.unweighted_sample_without_replacement(
                ids_unique, options.max_neighbors, exclude_edge_hashset=exclude_edge_hashset
            )
            (target_gids, edge_indice, csr_row_ptrs, csr_col_inds,sample_dup_counts,) = graph_block
            sample_node_cnt += target_gids[0].shape[0]
            sub_graphs = []
            target_gid_cnt = []
            for l in range(options.layernum):
                sub_graphs.append(create_sub_graph(
                    target_gids[l],
                    target_gids[l + 1],
                    edge_indice[l],
                    csr_row_ptrs[l],
                    csr_col_inds[l],
                    sample_dup_counts[l],
                    options.add_self_loop,
                ))
                target_gid_cnt.append(target_gids[l + 1].numel())
            torch.cuda.synchronize()
            sample_end_time = time.time()

            # extract
            x_feat = gather_fn(target_gids[0], dist_homo_graph.node_feat)
            torch.cuda.synchronize()
            extract_end_time = time.time()

            # train
            optimizer.zero_grad()
            if options.use_amp:
                with autocast(enabled=options.use_amp):
                    score = model(id_count, reverse_map, sub_graphs, target_gid_cnt, x_feat)
                    loss = F.binary_cross_entropy_with_logits(score, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                score = model(id_count, reverse_map, sub_graphs, target_gid_cnt, x_feat)
                loss = F.binary_cross_entropy_with_logits(score, labels)
                loss.backward()
                optimizer.step()

            # record SET step time
            torch.cuda.synchronize()
            step_end_time = time.time()
            latency_s += (sample_end_time - step_start_time)
            latency_e += (extract_end_time - sample_end_time)
            latency_t += (step_end_time - extract_end_time)
        if options.worker_id == 0:
            print(
                "[%s] [LOSS] epoch=%d, loss=%f"
                % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, loss.cpu().item())
            )
    options.global_barrier.wait()
    torch.cuda.synchronize()
    train_end_time = time.time()
    if options.worker_id == 0:
        print(
            "[TRAIN_TIME] train time is %.6f seconds"
            % (train_end_time - train_start_time)
        )
        print(
            "[EPOCH_TIME] %.6f seconds, maybe large due to not enough epoch skipped."
            % ((train_end_time - train_start_time) / options.epochs)
        )
        print(
            "[EPOCH_TIME] %.6f seconds"
            % ((train_end_time - skip_epoch_time) / (options.epochs - options.skip_epoch))
        )
        print(
            "[STEP_TIME] S = %.6f seconds, E = %.6f seconds, T = %.6f seconds"
            % (
                (latency_s / profile_steps),
                (latency_e / profile_steps),
                (latency_t / profile_steps)
            )
        )
        print(
            "[STEP_META] average sample node %d"
            % (sample_node_cnt / (options.local_step * options.epochs))
        )


def main(opt):
    global options
    options = opt
    print(options)

    wg.init_lib()
    torch.set_num_threads(1)
    os.environ["RANK"] = str(options.worker_id)
    os.environ["WORLD_SIZE"] = str(options.num_workers)
    # slurm in Selene has MASTER_ADDR env
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12335"
    local_rank = options.worker_id
    local_size = options.num_workers
    print("Rank=%d, local_rank=%d" % (local_rank, options.worker_id))
    dev_count = torch.cuda.device_count()
    assert dev_count > 0
    assert local_size <= dev_count
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    wm_comm = create_intra_node_communicator(
        options.worker_id, options.num_workers, local_size
    )
    wm_embedding_comm = None
    if options.use_nccl:
        if options.worker_id == 0:
            print("Using nccl embeddings.")
        wm_embedding_comm = create_global_communicator(
            options.worker_id, options.num_workers
        )
    if options.worker_id == 0:
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
    print("Rank=%d, Graph loaded." % (options.worker_id,))
    
    train_device = torch.device('cuda:{:}'.format(options.worker_id))
    predictor = DotPredictor()
    model = EdgePredictionGNNModel(
        dist_homo_graph, options.layernum, options.hiddensize,
        options.classnum, options.neighbors, predictor
    )
    model.cuda(device=train_device)
    model = DDP(model, delay_allreduce=True)
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=options.lr)
    print("Rank=%d, model and optimizer constructed, begin trainning..." % (options.worker_id,))

    model.train()
    train(dist_homo_graph, model, optimizer)

    wg.finalize_lib()
    print("Rank=%d, wholegraph shutdown." % (options.worker_id,))


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

    num_workers = options.num_workers
    # global barrier is used to sync all the sample workers and train workers
    options.global_barrier = mp.Barrier(num_workers, timeout=300.0)

    # fork child processes
    workers = []
    for worker_id in range(num_workers):
        options.worker_id = worker_id
        p = mp.Process(target=main, args=(options, ))
        p.start()
        workers.append(p)

    ret = wg.wait_one_child()
    if ret != 0:
        for p in workers:
            p.kill()
    for p in workers:
        p.join()

    if ret != 0:
        sys.exit(1)
