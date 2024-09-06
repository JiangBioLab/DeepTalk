import argparse
import json
import random
import os
import time
import shutil
from typing import List
import warnings
#warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
import shutil
import pandas as pd
import math
import re
import networkx as nx
from .src.processing.attributed_graph import AttributedGraph
import pickle
from .src.ccc_model.process_walks_gcn import Processing_GCN_Walks
from .src.finetuning import (
    run_finetuning_wkfl2,
    run_finetuning_wkfl3,
    setup_finetuning_input,
)
import torch.optim as optim
from torch.autograd import Variable
from .src.pretraining import run_pretraining, setup_pretraining_input
from .src.processing.context_generator import ContextGenerator
from .src.processing.generic_attributed_graph import GenericGraph
from .src.utils.data_utils import load_pretrained_node2vec
from .src.utils.evaluation import run_evaluation_main
from .src.utils.link_predict import find_optimal_cutoff, link_prediction_eval
from .src.utils.utils import get_id_map, load_pickle
from .src.ccc_model.ccc_model import CCC, FinetuneLayer
from .src.ccc_model.process_walks_gcn import Processing_GCN_Walks
from .src.utils.context_metrics import PathMetrics
from .src.utils.data_utils import create_finetune_batches2
from .src.utils.utils import load_pickle, show_progress


def get_set_embeddings_details(args):
    if not args.pretrained_embeddings:
        if args.pretrained_method == "compgcn":
            args.pretrained_embeddings = (
                f"{args.emb_dir}/{args.data_name}/"
                f"act_{args.data_name}_{args.node_edge_composition_func}_500.out"
            )
        elif args.pretrained_method == "node2vec":
            args.base_embedding_dim = 128
            args.pretrained_embeddings = (
                f"{args.emb_dir}/{args.data_name}/{args.data_name}.emd"
            )
    else:
        if args.pretrained_method == "compgcn":
            args.base_embedding_dim = 200
        elif args.pretrained_method == "node2vec":
            args.base_embedding_dim = 128
    return args.pretrained_embeddings, args.base_embedding_dim


def get_graph(args, data_path, false_edge_gen):
    #print("\n Loading graph...")
    attr_graph = GenericGraph(data_path, false_edge_gen)
    context_gen = ContextGenerator(attr_graph, args.num_walks_per_node)

    return attr_graph, context_gen


def get_test_edges(paths: List[str], sep: str):
    # edges = set()
    edges = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                tokens = line.strip().split(sep)
                etype = tokens[0]
                source = tokens[1]
                destination = tokens[2]
                label = tokens[3]
                edge = (etype, source, destination, label)
                # edges.add(edge)
                edges.append(edge)

    return edges


class GenericGraph_pre(AttributedGraph):
    def __init__(
        self, main_dir, false_edge_gen, attributes_file="", sample_training_set=False
    ):
        """
        Assumes derived classes will create a networkx graph object along with
        following
        1) self.unique_relations = set()
        2) self.node_attr_dfs = dict()
        3) self.node_types = dict()
        4) self.G  = nx.Graph()
        """
        super().__init__()
        self.main_dir = main_dir
        self.false_edge_gen = false_edge_gen
        # self.load_graph(main_dir + "/kg_final.txt")
        self.load_graph(main_dir + "/test.txt")
        self.create_normalized_node_id_map()

        # get product attributes
        self.node_attr = dict()
        self.node_attr_list = []

        if attributes_file == "":
            for node_id in self.G.nodes():
                self.node_attr[node_id] = dict()
                self.node_attr[node_id]["defattr"] = "True"
            self.node_attr_list = ["defattr"]
        else:
            raise NotImplementedError

        self.node_attr_dfs = {
            "deftype": pd.DataFrame.from_dict(self.node_attr, orient="index")
        }
        self.sample_training_set = sample_training_set
        self.get_nodeid2rowid()
        self.get_rowid2nodeid()
        self.get_rowid2vocabid_map()
        self.set_relation_id()

    def generate_link_prediction_dataset(self, outdir, fr_test_edges):
        #print("In GenericGraph link Prediction Generation")
        false_edge_gen = self.false_edge_gen

        self.test_edges = self.load_test_edges(self.main_dir + "/test.txt")
        return self.test_edges

    def load_graph(self, filename):
        """
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.node_attr_dfs
        3) self.unique_relations
        4) self.node_types
        """
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split(" ")
                if len(arr) >= 3:
                    edge_type = arr[0]
                    source = arr[1]
                    target = arr[2]
                    if len(arr) == 3 or (len(arr) == 4 and arr[3] == "1"):
                        self.G.add_edge(source, target, label=edge_type)
                        self.unique_relations.add(edge_type)
                        self.node_types[source] = "deftype"
                        self.node_types[target] = "deftype"
        return

    def load_test_edges(self, filename):
        true_and_false_edges = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split()
                relation = arr[0]
                source = arr[1]
                target = arr[2]
                if len(arr) == 4:
                    true_and_false_edges.append((relation, source, target, arr[3]))
                else:
                    # GATNE dataset have no false edges for training
                    true_and_false_edges.append((relation, source, target, "1"))
        return true_and_false_edges

    def get_continuous_cols(self):
        return {"deftype": None}

    def get_wide_cols(self):
        return {"deftype": self.node_attr_list}

class ContextGenerator_pre:
    def __init__(
        self,
        attributed_graph_obj,
        num_walks_per_node,
        pretrained_embeddings=None,
        optimal_context_per_relation=None,  # FIXME - removed dangerous empty dict init
    ):
        self.attr_graph = attributed_graph_obj
        self.num_walks_per_node = num_walks_per_node
        self.pretrained_embeddings = pretrained_embeddings
        random.seed(10)

    def get_random_k_nbrs(self, nodeid, exclude_list, k):
        """
        returns randomly sampled 'k' nbrs for a given node as
        list[node_ids]
        if k == -1 :  returns all neighbours
        """
        all_nbrs = list(set(list(nx.all_neighbors(self.attr_graph.G, nodeid))))
        valid_nbrs = all_nbrs
        if exclude_list is None or len(exclude_list) == 0:
            valid_nbrs = all_nbrs
        else:
            valid_nbrs = [x for x in all_nbrs if x not in exclude_list]
        if k >= 0 and len(valid_nbrs) >= k:
            return random.sample(valid_nbrs, k)
        else:
            return all_nbrs

    def gen_random_walk_context(
        self, beam_width, max_num_edges, walk_type, valid_patterns=None
    ):  # FIXME - removed dangerous valid_patterns list init
        """
        Given as input an attributed networkx graph, generates
        random walk based node_context for  each node.
        Returns : list[node_context] == list[list[(source, dest, relation)]]
        Node context is generated by performing random walk around the
        node with beam_width = beam_width and depth = 2
        """
        G = self.attr_graph.get_graph()
        all_contexts = []
        for source in list(G.nodes()):
            # for u,v in nx.bfs_beam_edges(G, source, centrality.get, beam_width):
            # for u,v in nx.bfs_beam_edges(G, source, random_neighbour_goodness, beam_width):
            #    node_context.append((u,v, G.edges[u,v]['label']))
            for _ in range(self.num_walks_per_node):
                if walk_type == "bfs":
                    node_context = self.bfs_beam(source, beam_width)
                elif walk_type == "dfs":
                    node_context = self.random_chain(source, max_num_edges)
                else:
                    print("Unknown context generation strategy, select bfs/dfs")
                all_contexts.append(node_context)
        # valid_patterns = set(['1_1_1', '2_2_2', '1_2_1', '2_1_2'])
        # all_contexts = self.filter_walks_by_patterns(all_contexts, valid_patterns)
        return all_contexts

    def filter_walks_by_patterns(self, walks, valid_patterns):
        # valid_walks = [walk for walk in walks  if self.get_pattern(walk) in
        #               valid_patterns]
        valid_walks = []
        for walk in walks:
            walk_pattern = self.get_pattern(walk)
            if walk_pattern in valid_patterns:
                valid_walks.append(walk)
        return valid_walks

    def get_pattern(self, walk, first_last_same=False):
        """
        given a path = list[(source, target, relation)]
        returns a pattern = [source[0].type, source[1].type, source[2].type...]
        """
        pattern = []
        for edge in walk:
            # source = edge[0]
            relation = edge[2]
            # source_type = self.attr_graph.get_node_type(source)
            # pattern.append(source_type)
            pattern.append(relation)
        return "_".join(pattern)
        # return pattern

    def random_chain(self, source, max_num_edges):
        # generates a random walk of length=max_num_edges, without cycles
        # starting from source
        path = []
        nodes_so_far = []
        G = self.attr_graph.get_graph()
        for _ in range(max_num_edges):
            nbr = self.get_random_k_nbrs(source, exclude_list=nodes_so_far, k=1)[0]
            try:
                relation = G.edges[source, nbr]["label"]
            except KeyError:  # FIXME - replaced bare except
                relation = G.edges[nbr, source]["label"]
            path.append((source, nbr, relation))
            nodes_so_far.append(source)
            source = nbr
        return path

    def bfs_beam(self, source, beam_width):
        """
        Given a source node, a beam width and depth=hops
        return a subgraph around source node .
        The subgraph is retuned as [(source, target, relation)]
        """
        subgraph = list()
        G = self.attr_graph.get_graph()
        source_nbrs = self.get_random_k_nbrs(source, exclude_list=[], k=beam_width)
        for x in source_nbrs:
            try:
                relation = G.edges[source, x]["label"]
            except KeyError:  # FIXME - replaced bare except
                relation = G.edges[x, source]["label"]
            subgraph.append((source, x, relation))
        for src_nbr in source_nbrs:
            src_nbr_hop2_cands = self.get_random_k_nbrs(
                src_nbr, exclude_list=[source], k=beam_width
            )
            for x in src_nbr_hop2_cands:
                try:
                    relation = G.edges[src_nbr, x]["label"]
                except KeyError:  # FIXME - replaced bare except
                    relation = G.edges[x, src_nbr]["label"]
                subgraph.append((src_nbr, x, relation))
        # print(len(subgraph))
        return subgraph

    def get_pretrain_subgraphs(
        self,
        data_path,
        data_name,
        beam_width,
        max_num_edges,
        walk_type,
        mode="pretrain",
    ):
        walk_file = os.path.join(data_path, data_name + "_walks.txt")
        if os.path.exists(walk_file):
            #print("\n load walks ...")
            fin = open(walk_file, "r")
            all_walks = []
            for line in fin:
                line = json.loads(line)
                all_walks.append(line)
            fin.close()
        else:
            #print("\n generate walks ...")
            # context_gen = ContextGenerator(attr_graph)
            # all_walks = self.gen_random_walk_context(beam_width)
            # for pretraining should we geerate walks without taking out test
            # and validation
            all_walks = self.gen_random_walk_context(
                beam_width, max_num_edges, walk_type
            )
            walk_file = os.path.join(data_path, data_name + "_walks.txt")
            fout = open(walk_file, "w")
            for walk in all_walks:
                json.dump(walk, fout)
                fout.write("\n")
            fout.close()
        #print("no. of walks", len(all_walks))
        return all_walks

    def get_finetune_subgraphs(
        self, test_edges, beam_width, max_seq_len, option
    ):

        # print("size of train, valid, test set", len(train_edges),
        #      len(valid_edges), len(test_edges))
        #true_test_edges = [x for x in test_edges if x[3] == "1"]
        #true_valid_edges = [x for x in valid_edges if x[3] == "1"]
        #to_be_removed_edges = true_test_edges + true_valid_edges
        #to_be_removed_source_target_pairs = [(x[1], x[2]) for x in to_be_removed_edges]
        #self.attr_graph.G.remove_edges_from(to_be_removed_source_target_pairs)
        walks_by_task_dict = dict()
        for task, task_edges in zip(
            ["test"], [test_edges]
        ):
            task_subgraphs = []
            for edge in task_edges:
                source = edge[1]
                target = edge[2]
                relation = edge[0]
                is_true = edge[3]
                # source_target_contexts = self.get_path_with_edge_label(source, target,
                #                                      max_seq_len-1)
                source_target_contexts = self.get_selective_context(
                    source, target, beam_width, max_seq_len - 1, option
                )
                if len(source_target_contexts) == 0:
                    first_edge = (source, target, relation, is_true)
                    task_subgraphs.append([first_edge])
                else:
                    for subgraph in source_target_contexts:
                        first_edge = (source, target, relation, is_true)
                        subgraph.insert(0, first_edge)
                        task_subgraphs.append(subgraph)
                walks_by_task_dict[task] = task_subgraphs
        # print(walks_by_task_dict)
        # add edges back to graph, in case attributed graph is used by rest of
        # code
        return walks_by_task_dict

    def get_context(self, source, target, beam_width, max_seq_len):
        all_contexts = []
        # for i in range(self.num_walks_per_node):
        source_context = self.bfs_beam(source, beam_width)[0:max_seq_len]
        # target_context = self.bfs_beam(target, beam_width)[0:max_seq_len]
        all_contexts.append(source_context)
        # all_contexts.append(target_context)
        return all_contexts

    def get_selective_context(
        self, source, target, beam_width, max_seq_len, option, valid_patterns=None
    ):  # FIXME - removed dangerous default list init in valid_patterns

        G = self.attr_graph.G
        paths = []
        if option == "shortest":
            try:
                path_node_list = nx.bidirectional_shortest_path(G, source, target)
                path_with_edge_label = []
                for i in range(len(path_node_list) - 1):
                    u = path_node_list[i]
                    v = path_node_list[i + 1]
                    path_with_edge_label.append((u, v, G.edges[u, v]["label"]))
                paths = [path_with_edge_label[0:max_seq_len]]
            except nx.exception.NetworkXNoPath:
                return []

        elif option == "all":
            paths = self.get_path_with_edge_label(source, target, max_seq_len)
            # print("found all paths", source, target, paths)
        elif option == "pattern":
            paths = self.get_path_with_edge_label(source, target, max_seq_len)
            if len(valid_patterns) > 0:
                paths = self.filter_walks_by_patterns(paths, valid_patterns)
            #print("found pattern paths", source, target, paths)
        elif option == "random":
            node_list_paths = self.get_random_paths_between_nodes(
                source, target, beam_width, max_seq_len, currpath=[]
            )
            paths = []
            for node_list_path in node_list_paths:
                path_with_edge_label = []
                node_list_path.insert(0, source)
                for i in range(len(node_list_path) - 1):
                    u = node_list_path[i]
                    v = node_list_path[i + 1]
                    path_with_edge_label.append((u, v, G.edges[u, v]["label"]))
                paths.append(path_with_edge_label)
        elif option == "default":
            paths = self.get_context(source, target, beam_width, max_seq_len)
        return paths

    def get_random_paths_between_nodes(
        self, source, target, beam_width, max_seq_len, currpath
    ):
        """
        Generates random path between source and target , by selecting
        'beam_width' number of neighbors and upto length >= max_seq_len
        """
        if len(currpath) > max_seq_len or source == target:
            return []
        all_paths = []
        exclude_list = []

        # print("In generate paths", source, target, currpath)

        source_nbrs = self.get_random_k_nbrs(source, exclude_list, beam_width)
        # print("selected source nbrs", source_nbrs)
        if target in source_nbrs:
            # print("Found path ending in target", source_nbrs, target)
            path = [target]
            all_paths.append(path)

        for n in source_nbrs:
            if n != target and n not in currpath:
                new_source = n
                new_path = list(currpath)
                new_path.append(new_source)
                nbr_paths = self.get_random_paths_between_nodes(
                    new_source, target, beam_width, max_seq_len, new_path
                )
                new_path = new_path[0:-1]
                for p in nbr_paths:
                    p.insert(0, new_source)
                all_paths.extend(nbr_paths)
        return all_paths

    def get_path_metrics(self, G, source, target, paths):
        """
        Metrics per node pair:
            1) Number of paths connecting the nodes
            2) Number of metapaths
            5) Polysemy behavior : Variation in number of unique relations each node participates in
            6)
        """
        return

    def get_path_with_edge_label(self, source, target, max_seq_len):
        G = self.attr_graph.G
        paths = nx.all_simple_paths(G, source, target, cutoff=max_seq_len)
        paths_with_edge_labels = []
        for path in map(nx.utils.pairwise, paths):
            path_with_edge_label = []
            for u, v in path:
                path_with_edge_label.append((u, v, G.edges[u, v]["label"]))
            paths_with_edge_labels.append(path_with_edge_label)
        return paths_with_edge_labels

def get_graph(args, data_path, false_edge_gen):
    #print("\n Loading graph...")
    attr_graph = GenericGraph_pre(data_path, false_edge_gen)
    context_gen = ContextGenerator_pre(attr_graph, args.num_walks_per_node)

    return attr_graph, context_gen

def split_data_pre(data_list, data_path, data_name):
    random.shuffle(data_list)
    data_size = len(data_list)
    data_test = data_list
    tasks = ["test"]
    split_file = {
        task: os.path.join(data_path, data_name + "_walks_" + task + ".txt")
        for task in tasks
    }
    for task in tasks:
        #print(task, split_file[task])
        fout = open(split_file[task], "w")
        if task == "train":
            walk_data = data_train
        elif task == "validate":
            walk_data = data_validate
        else:
            walk_data = data_test

        for walk in walk_data:
            json.dump(walk, fout)
            fout.write("\n")
        fout.close()

    return data_test

def create_batches_pre(data_list, data_path, data_name, task, batch_size):
    folder = os.path.join(data_path, task)
    if os.path.exists(folder):
        cnt = math.ceil(len(data_list) / batch_size)
        return cnt
    else:
        os.mkdir(folder)
        random.shuffle(data_list)
        data_size = len(data_list)
        cnt = 0
        data = []
        for ii in range(data_size):
            data.append(data_list[ii])
            if len(data) == batch_size:
                file_name = data_name + "_" + task + "_batch_" + str(cnt) + ".txt"
                fout = open(os.path.join(folder, file_name), "w")
                for jj, _ in enumerate(data):
                    json.dump(data[jj], fout)
                    fout.write("\n")
                fout.close()
                data = []
                cnt += 1

        if len(data) > 0:
            file_name = data_name + "_" + task + "_batch_" + str(cnt) + ".txt"
            fout = open(os.path.join(folder, file_name), "w")
            for jj, _ in enumerate(data):
                json.dump(data[jj], fout)
                fout.write("\n")
            fout.close()
            cnt += 1

        return cnt

def setup_pretraining_input_pre(args, attr_graph, context_gen, data_path):
    # split train/validation/test dataset
    tasks = ["test"]
    #print("\n Generating subgraphs for pre-training ...")
    all_walks = context_gen.get_pretrain_subgraphs(
        data_path,
        args.data_name,
        args.beam_width,
        args.max_length,
        args.walk_type,
    )
    #print("\n split data to train/validate/test and save the files ...")
    walk_test = split_data_pre(
        all_walks, data_path, args.data_name
    )
    #print(len(walk_test))

    # create batches
    num_batches = {}
    for task in tasks:
        if task == "train":
            walk_data = walk_train
        elif task == "validate":
            walk_data = walk_validate
        else:
            walk_data = walk_test
        cnt = create_batches_pre(
            walk_data, data_path, args.data_name, task, args.batch_size
        )
        num_batches[task] = cnt
    #print("number of batches for pre-training: ", num_batches)
    return num_batches

def create_finetune_batches2_pre(finetune_data, finetune_path, data_name, batch_size):

    no_batches = {}
    tasks = ["test"]
    #task_file = {task: os.path.join(finetune_path, task + ".txt") for task in tasks}
    for task in tasks:
        #print(task)
        all_task_subgraphs = finetune_data[task]
        tmp_no = create_batches_pre(
            all_task_subgraphs, finetune_path, data_name, task, batch_size
        )
        no_batches[task] = tmp_no
    return no_batches

def setup_finetuning_input_pre(args, attr_graph, context_gen):
    # finetune
    #print("\n Generate data for finetuning ...")
    num_batches = {}
    finetune_path = args.outdir + "/finetune/"
    if not os.path.exists(finetune_path):
        os.mkdir(finetune_path)

    if os.path.exists(finetune_path + "/finetune_walks.txt"):
        finetune_walks_per_task = json.load(open(finetune_path + "/finetune_walks.txt"))
    else:
        test_edges = attr_graph.generate_link_prediction_dataset(
            finetune_path, fr_test_edges=0.1)
        
        finetune_walks_per_task = context_gen.get_finetune_subgraphs(
            test_edges,
            args.beam_width,
            args.max_length,
            args.path_option,
        )
        json.dump(
            finetune_walks_per_task, open(finetune_path + "/finetune_walks.txt", "w")
        )

        #print(
        #    "Number of train, valid, test",
        #    len(test_edges),
        #)
    num_batches = create_finetune_batches2_pre(
        finetune_walks_per_task, finetune_path, args.data_name, args.ft_batch_size
    )
    #print("No. of batches for finetuning:", num_batches)

    return num_batches


def Predict(
    args, attr_graph, no_batches, pretrained_node_embedding, ent2id, rel2id,model_path,best_id,is_trained,model_name
):

    # data_path = args.data_path + args.data_name + "/"
    outdir = args.outdir + "/"
    finetune_path = outdir + "finetune/"
    ft_out_dir = finetune_path + "/results"
    #try:
    #    shutil.rmtree(ft_out_dir)
    #   os.mkdir(ft_out_dir)
    #except:  # FIXME - need to replace bare except
    #   os.mkdir(ft_out_dir)
    relations = attr_graph.relation_to_id
    nodeid2rowid = attr_graph.get_nodeid2rowid()
    walk_processor = Processing_GCN_Walks(
        nodeid2rowid, relations, args.n_pred, args.max_length, args.max_pred
    )

    # process minibatch for finetune(ft)
    #print("\n processing minibaches for finetuning:")
    ft_batch_input_file = os.path.join(
        finetune_path, args.data_name + "_ft_batch_input.pickled"
    )
    if os.path.exists(ft_batch_input_file):
        #print("loading saved files ...")
        ft_batch_input = load_pickle(ft_batch_input_file)
    else:
        ft_batch_input = {}
        tasks = ["test"]
        for task in tasks:
            #print(task)
            ft_batch_input[task] = {}
            for batch_id in range(no_batches[task]):
                (
                    subgraphs,
                    all_nodes,
                    labels,
                ) = walk_processor.process_finetune_minibatch(
                    finetune_path, args.data_name, task, batch_id
                )
                ft_batch_input[task][batch_id] = [subgraphs, all_nodes, labels]
        pickle.dump(ft_batch_input, open(ft_batch_input_file, "wb"))

    if args.is_pre_trained:
        pretrained_node_embedding_tensor = pretrained_node_embedding
    else:
        pretrained_node_embedding_tensor = None

    ccc = CCC(
        args.n_layers,
        args.device,
        args.d_model,
        args.d_k,
        args.d_v,
        args.d_ff,
        args.n_heads,
        attr_graph,
        pretrained_node_embedding_tensor,
        args.is_pre_trained,
        args.base_embedding_dim,
        args.max_length,
        args.num_gcn_layers,
        args.node_edge_composition_func,
        args.gcn_option,
        args.get_bert_encoder_embeddings,
        ent2id,
        rel2id,
    )

    pretrained_node_embedding_tensor = ccc.gcn_graph_encoder.node_embedding

    ft_linear = FinetuneLayer(
        args.device,
        args.d_model,
        args.ft_d_ff,
        args.ft_layer,
        args.ft_drop_rate,
        attr_graph,
        args.ft_input_option,
        args.n_layers,
    )
    

    # run in fine tuning mode
    ccc.set_fine_tuning()

    # testing
    print("Begin Predicting")
    ccc.eval()
    ft_linear.eval()
    if is_trained:
        fl_ = model_name
    else:
        fl_ = os.path.join(model_path, "finetune_{}.model".format(best_id))
    ft_linear.load_state_dict(
        torch.load(fl_, map_location=lambda storage, loc: storage)
    )

    pred_data = []
    true_data = []

    with torch.no_grad():
        att_weights = {}
        final_emd = {}
        for batch_test_id in range(no_batches["test"]):
            #if batch_test_id % 100 == 0:
                #print('###')
                #print("Evaluating test batch {}".format(batch_test_id))
            #print(batch_test_id)
            subgraphs, all_nodes, labels = ft_batch_input["test"][batch_test_id]
            masked_pos = torch.randn(args.batch_size, 2)
            masked_nodes = Variable(
                torch.LongTensor([[] for ii in range(args.ft_batch_size)])
            )
            _, layer_output, att_output = ccc(
                subgraphs, all_nodes, masked_pos, masked_nodes
            )
            score, src_embedding, dst_embedding = ft_linear(layer_output)
            score = score.data.cpu().numpy().tolist()
            labels = labels.data.cpu().numpy().tolist()

            for ii, _ in enumerate(score):
                pred_data.append(score[ii][0])
                true_data.append(labels[ii][0])
            # print(len(score), len(labels))


    return pred_data, true_data



def run_predict(data_name='Apoe_Grm5',data_path='./Test/single-cell/data/',
    outdir='./Test/single-cell/data/Apoe_Grm5/output/',
    pretrained_embeddings='./Test/single-cell/data/data_pca.emd', model_path = './Test/single-cell/data',best_id = 9, is_trained = False,model_name = 'Mif_Epha5.model'):

    #torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", type=str, help='specify device')
    parser.add_argument("--data_name", default="EFNA5_EPHA2", help="name of the dataset")
    parser.add_argument("--data_path", default="./data/", help="path to dataset")
    parser.add_argument("--outdir", default="./data/EFNA5_EPHA2/output", help="path to output dir")
    parser.add_argument(
        "--pretrained_embeddings", default="./data_pca.emd",help="absolute path to pretrained embeddings"
    )
    parser.add_argument(
        "--pretrained_method", default="node2vec", help="compgcn|node2vec"
    )
    # Walks options
    parser.add_argument(
        "--beam_width",
        default=1,
        type=int,
        help="beam width used for generating random walks",
    )
    parser.add_argument(
        "--num_walks_per_node", default=1, type=int, help="walks per node"
    )
    parser.add_argument("--walk_type", default="dfs", help="walk type bfs/dfs")
    parser.add_argument("--max_length", default=5, type=int, help="max length of walks")
    parser.add_argument(
        "--n_pred", default=1, help="number of tokens masked to be predicted"
    )
    parser.add_argument(
        "--max_pred", default=1, help="max number of tokens masked to be predicted"
    )
    # Pretraining options
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument(
        "--n_epochs", default=50, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--checkpoint", default=100, type=int, help="checkpoint for validation"
    )
    parser.add_argument(
        "--base_embedding_dim",
        default=200,
        type=int,
        help="dimension of base embedding",
    )
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="number of data sample in each batch",
    )
    parser.add_argument(
        "--emb_dir",
        default="data",
        type=str,
        help="Used to generate embeddings path if --pretrained_embeddings is not set",
    )
    parser.add_argument(
        "--get_bert_encoder_embeddings",
        default=False,
        help="indicate if need to get node vectors from BERT encoder output, save code "
             "commented out in src/pretraining.py",
    )
    # BERT Layer options
    parser.add_argument(
        "--n_layers", default=4, type=int, help="number of encoder layers in bert"
    )
    parser.add_argument(
        "--d_model", default=200, type=int, help="embedding size in bert"
    )
    parser.add_argument("--d_k", default=64, type=int, help="dimension of K(=Q), V")
    parser.add_argument("--d_v", default=64, type=int, help="dimension of K(=Q), V")
    parser.add_argument("--n_heads", default=4, type=int, help="number of head in bert")
    parser.add_argument(
        "--d_ff",
        default=200 * 4,
        type=int,
        help="4*d_model, FeedForward dimension in bert",
    )
    # GCN Layer options
    parser.add_argument(
        "--is_pre_trained",default=False,
        action="store_true",
        help="if there is pretrained node embeddings",
    )
    parser.add_argument(
        "--gcn_option",
        default="no_gcn",
        help="preprocess bert input once or alternate gcn and bert, preprocess|alternate|no_gcn",
    )
    parser.add_argument(
        "--num_gcn_layers", default=4, type=int, help="number of gcn layers before bert"
    )
    parser.add_argument(
        "--node_edge_composition_func",
        default="no_rel",
        help="options for node and edge compostion, sub|circ_conv|mult|no_rel",
    )

    # Finetuning options
    parser.add_argument("--ft_lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument(
        "--ft_batch_size",
        default=1024,
        type=int,
        help="number of data sample in each batch",
    )
    parser.add_argument(
        "--ft_checkpoint", default=1000, type=int, help="checkpoint for validation"
    )
    parser.add_argument(
        "--ft_d_ff", default=512, type=int, help="feedforward dimension in finetuning"
    )
    parser.add_argument(
        "--ft_layer", default="ffn", help="options for finetune layer: linear|ffn"
    )
    parser.add_argument(
        "--ft_drop_rate", default=0.1, type=float, help="dropout rate in finetuning"
    )
    parser.add_argument(
        "--ft_input_option",
        default="last4_cat",
        help="which output layer from graphbert will be used for finetuning, last|last4_cat|last4_sum",
    )
    parser.add_argument(
        "--false_edge_gen",
        default="basic",
        help="false edge generation pattern/double/basic",
    )
    parser.add_argument(
        "--ft_n_epochs", default=10, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--path_option",
        default="shortest",
        help="fine tuning context generation: shortest/all/pattern/random",
    )
    args = parser.parse_args(args=[])

    # Default values
    args.pretrained_embeddings, args.base_embedding_dim = get_set_embeddings_details(
        args
    )
    args.d_model = args.base_embedding_dim
    args.d_ff = args.base_embedding_dim
    args.data_name = data_name
    args.data_path = data_path
    args.outdir = outdir
    args.pretrained_embeddings = pretrained_embeddings
    #args.n_epochs = n_epochs
    #args.ft_n_epochs = ft_n_epochs

    data_path = f"{args.data_path}/{args.data_name}"
    attr_graph, context_gen = get_graph(args, data_path, args.false_edge_gen)
    attr_graph.dump_stats()
    stime = time.time()
    id_maps_dir = data_path
    ent2id = get_id_map(f"{id_maps_dir}/ent2id.txt")
    rel2id = get_id_map(f"{id_maps_dir}/rel2id.txt")

    pretrained_node_embedding = load_pretrained_node2vec(
            args.pretrained_embeddings, ent2id, args.base_embedding_dim
        )
    outdir = data_path+"/outdir"
    isExists = os.path.exists(outdir)
    if isExists:
        shutil.rmtree(outdir)
        os.makedirs(outdir)
    if not isExists:
        os.makedirs(outdir)
    
    outpath = args.outdir + "/"
    isExists = os.path.exists(outpath)
    if isExists:
        shutil.rmtree(outpath)
        os.makedirs(outpath)
    if not isExists:
        os.makedirs(outpath)
    
    finetune_path = outpath + "/finetune/"
    try:
        shutil.rmtree(finetune_path)
        os.mkdir(finetune_path)
    except:  # FIXME - need to replace bare except
        os.mkdir(finetune_path)
    ft_out_dir = finetune_path + "results"
    try:
        shutil.rmtree(ft_out_dir)
        os.mkdir(ft_out_dir)
    except:  # FIXME - need to replace bare except
        os.mkdir(ft_out_dir)
    
    test_path = data_path + "/test.txt"
    test_edges_paths = [test_path]
    test_edges = list(get_test_edges(test_edges_paths, " "))
    #print("No. edges in test data: ", len(test_edges))

    pre_num_batches = setup_pretraining_input_pre(args, attr_graph, context_gen, args.outdir)
    ft_num_batches = setup_finetuning_input_pre(args, attr_graph, context_gen)

    (
        pred_data_test,
        true_data_test,
    ) = Predict(
        args, attr_graph, ft_num_batches, pretrained_node_embedding, ent2id, rel2id, model_path,best_id,is_trained,model_name
    )

    y_pred = np.zeros(len(pred_data_test), dtype=np.int32)

    for i, _ in enumerate(pred_data_test):
        if pred_data_test[i] >= 0.5:
            y_pred[i] = 1
    print('Done')
    CCIlist_new = np.loadtxt(test_path,skiprows=0,dtype=str,delimiter=' ')
    columns = [0, 1, 2]
    CCIlist_new_score = CCIlist_new[:, columns]
    CCIlist_new_score = np.hstack((CCIlist_new_score, y_pred[:, np.newaxis]))
    np.savetxt(data_path + '/predict_ccc.txt', CCIlist_new_score, fmt='%s', delimiter=' ')
    #os.remove(outdir + '/test.txt')
    #return y_pred