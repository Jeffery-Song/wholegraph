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

import os
from optparse import OptionParser

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import torch
from wg_torch.graph_ops import (
    check_data_integrity,
    numpy_dtype_to_string,
    load_meta_file,
    save_meta_file,
    get_part_filename,
    graph_name_normalize,
)

from wholegraph.torch import wholegraph_pytorch as wg

_dataset_root = '/graph-learning/samgraph/'

def parse_meta(meta):
    meta_dict = {}
    with open(meta, 'r') as F:
        for line in F.readlines():
            k, v = line.split()
            k = k.lower()
            v = int(v)
            meta_dict[k] = v
    return meta_dict


def csc2coo(num_edge, indptr, indices):
    data = np.zeros(num_edge, dtype=np.int32)
    csc_mat = csc_matrix((data, indices, indptr))
    coo_mat = csc_mat.tocoo()
    return np.stack([coo_mat.row, coo_mat.col])


def convert_node_classification_dataset(
    save_dir:str, ogb_root_dir:str, graph_name:str
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    normalized_graph_name = graph_name_normalize(graph_name)
    support_graph_dict = {
        'reddit' : {"node_name": "node", "relation" : "edge"},
        "products": {"node_name": "product", "relation": "copurchased"},
        "papers100M": {"node_name": "paper", "relation": "cites"},
        'twitter' : {"node_name": "node", "relation" : "edge"},
        'uk-2006-05' : {"node_name": "node", "relation" : "edge"},
        'com-friendster' : {"node_name": "node", "relation" : "edge"},
    }
    assert normalized_graph_name in support_graph_dict.keys()
    node_name = support_graph_dict[normalized_graph_name]["node_name"]
    relation_name = support_graph_dict[normalized_graph_name]["relation"]

    dataset_root = os.path.join(_dataset_root, graph_name)
    train_idx = np.fromfile(os.path.join(dataset_root, 'train_set.bin'), dtype=np.int32)
    valid_idx = np.fromfile(os.path.join(dataset_root, 'valid_set.bin'), dtype=np.int32)
    test_idx = np.fromfile(os.path.join(dataset_root, 'test_set.bin'), dtype=np.int32)

    meta_dict = parse_meta(os.path.join(dataset_root, 'meta.txt'))
    num_nodes = meta_dict['num_node']
    num_edges = meta_dict['num_edge']
    node_feat_dim = meta_dict['feat_dim']

    if os.path.exists(os.path.join(dataset_root, 'label.bin')):
        label = np.fromfile(os.path.join(dataset_root, 'label.bin'), dtype=np.int64)
    else:
        label = np.zeros(num_nodes, dtype=np.int64)

    indptr = np.fromfile(os.path.join(dataset_root, 'indptr.bin'), dtype=np.int32)
    indices = np.fromfile(os.path.join(dataset_root, 'indices.bin'), dtype=np.int32)
    edge_index = csc2coo(num_edges, indptr, indices)

    if os.path.exists(os.path.join(dataset_root, 'feat.bin')):
        node_feat = np.fromfile(os.path.join(dataset_root, 'feat.bin'), dtype=np.float32)
    else:
        node_feat = np.random.randn(num_nodes * node_feat_dim).astype(np.float32)

    node_feat_name_prefix = "_".join([normalized_graph_name, "node_feat", node_name])
    edge_index_name_prefix = "_".join(
        [normalized_graph_name, "edge_index", node_name, relation_name, node_name]
    )

    nodes = [
        {
            "name": node_name,
            "has_emb": True,
            "emb_file_prefix": node_feat_name_prefix,
            "num_nodes": num_nodes,
            "emb_dim": node_feat_dim,
            "dtype": numpy_dtype_to_string(np.dtype('float32')),
        }
    ]
    edges = [
        {
            "src": node_name,
            "dst": node_name,
            "rel": relation_name,
            "has_emb": False,
            "edge_list_prefix": edge_index_name_prefix,
            "num_edges": num_edges,
            "dtype": numpy_dtype_to_string(np.dtype('int32')),
            "directed": True,
        }
    ]
    meta_json = {"nodes": nodes, "edges": edges}
    save_meta_file(save_dir, meta_json, normalized_graph_name)
    train_label = label[train_idx]
    valid_label = label[valid_idx]
    test_label = label[test_idx]
    data_and_label = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "train_label": train_label,
        "valid_label": valid_label,
        "test_label": test_label,
    }
    import pickle

    # train, valid, test set and label
    with open(
        os.path.join(save_dir, normalized_graph_name + "_data_and_label.pkl"), "wb"
    ) as f:
        pickle.dump(data_and_label, f)

    # feature
    print("saving node feature...")
    with open(
        os.path.join(save_dir, get_part_filename(node_feat_name_prefix)), "wb"
    ) as f:
            node_feat.tofile(f)
    
    # edge
    print("converting edge index...")
    edge_index_int32 = np.transpose(edge_index).astype(np.int32)
    print("saving edge index...")
    with open(
        os.path.join(save_dir, get_part_filename(edge_index_name_prefix)), "wb"
    ) as f:
        edge_index_int32.tofile(f)


def build_homo_graph(root_dir: str, graph_name: str):
    normalized_graph_name = graph_name_normalize(graph_name)
    output_dir = os.path.join(root_dir, normalized_graph_name, "converted")
    meta_file = load_meta_file(output_dir, normalized_graph_name)
    graph_builder = wg.create_homograph_builder(torch.int32)
    wg.graph_builder_set_shuffle_id(graph_builder, False)
    wg.graph_builder_load_edge_data(
        graph_builder,
        [],
        os.path.join(output_dir, meta_file["edges"][0]["edge_list_prefix"]),
        False,
        torch.int32,
        0,
    )
    wg.graph_builder_set_edge_config(graph_builder, [], True, False, False)
    wg.graph_builder_set_graph_save_file(
        graph_builder,
        os.path.join(output_dir, "homograph_csr_row_ptr"),
        os.path.join(output_dir, "homograph_csr_col_idx"),
        os.path.join(output_dir, "homograph_id_mapping"),
    )

    wg.graph_builder_build(graph_builder)
    wg.destroy_graph_builder(graph_builder)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-r",
        "--root_dir",
        dest="root_dir",
        default="/dev/shm/dataset",
        help="graph root directory.",
    )
    parser.add_option(
        "-g",
        "--graph_name",
        dest="graph_name",
        default="twitter",
        help="graph name, reddit, products, twitter, papers100M, uk-2006-05, com-friendster",
    )

    (options, args) = parser.parse_args()

    # convert, set up meta info
    norm_graph_name = graph_name_normalize(options.graph_name)
    if (
        options.graph_name in {'reddit', 'products', 'twitter', 'papers100M', 'uk-2006-05', 'com-friendster'}
    ):
        if os.path.exists(os.path.join(options.root_dir, norm_graph_name)):
            print(f"dataset {norm_graph_name} exist")
            exit(0)
        convert_node_classification_dataset(
            os.path.join(options.root_dir, norm_graph_name, "converted"),
            options.root_dir,
            options.graph_name,
        )
    else:
        raise ValueError("graph name unknown.")

    # build
    build_homo_graph(os.path.join(options.root_dir), options.graph_name)
