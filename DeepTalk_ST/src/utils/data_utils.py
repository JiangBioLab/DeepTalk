import json
import math
import os
import random
import re
import torch


def split_data(data_list, data_path, data_name):
    random.shuffle(data_list)
    data_size = len(data_list)

    data_train = []
    data_validate = []
    data_test = []
    for ii in range(data_size):
        if ii < round(data_size * 0.8):
            data_train.append(data_list[ii])
        elif ii in range(round(data_size * 0.8), round(data_size * 0.9)):
            data_validate.append(data_list[ii])
        else:
            data_test.append(data_list[ii])

    tasks = ["train", "validate", "test"]
    split_file = {
        task: os.path.join(data_path, data_name + "_walks_" + task + ".txt")
        for task in tasks
    }
    for task in tasks:
        print(task, split_file[task])
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

    return data_train, data_validate, data_test


def create_batches(data_list, data_path, data_name, task, batch_size):
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


def create_finetune_batches(finetune_path, data_name, batch_size):
    no_batches = {}
    tasks = ["train", "valid", "test"]
    # finetune_path = os.path.join(data_path)
    task_file = {task: os.path.join(finetune_path, task + ".txt") for task in tasks}
    for task in tasks:
        print(task)
        data_arr = []
        fin = open(task_file[task], "r")
        cnt = 0
        for line in fin:
            arr = []
            line = re.split(" ", line[:-1])
            line = [line[1], line[2], line[0], line[3]]
            arr.append(line)
            data_arr.append(arr)
            cnt += 1
            if cnt % 50000 == 0:
                print(cnt)
        fin.close()

        tmp_no = create_batches(data_arr, finetune_path, data_name, task, batch_size)
        no_batches[task] = tmp_no

    return no_batches


def create_finetune_batches2(finetune_data, finetune_path, data_name, batch_size):
    """
    receives a dictionary of all test and valid, cases .
    each subgraph is a list of edegs, where first edge contains (test source,
    test detination, test relation, 0/1). Other edges are context generated
    around source and destination. Example
    "train" : [
                [(train_edge_1, label), (context_edge_11), (context_edge12)..]
                [(train_edge_2, label), (context_edge12), (context_edge22)..]
                ]

    "test" : [
                [(test_edge_1, label), (context_edge_11), (context_edge12)..]
                [(test_edge_2, label), (context_edge12), (context_edge22)..]
                ]

    "valid" : [
                [(valid_edge_1, label), (context_edge_11), (context_edge12)..]
                [(valid_edge_2, label), (context_edge12), (context_edge22)..]
                ]

    Generates train, validtaion and test files per batch
    """

    no_batches = {}
    tasks = ["train", "valid", "test"]
    #task_file = {task: os.path.join(finetune_path, task + ".txt") for task in tasks}
    for task in tasks:
        print(task)
        all_task_subgraphs = finetune_data[task]
        tmp_no = create_batches(
            all_task_subgraphs, finetune_path, data_name, task, batch_size
        )
        no_batches[task] = tmp_no
    return no_batches


def load_pretrained_node2vec(filename, ent2id, base_emb_dim):
    """
    loads embeddings from node2vec style file, where each line is
    nodeid node_embedding
    returns tensor containing node_embeddings
    for graph nodes 0 to n-1
    """
    node_embeddings = dict()
    with open(filename, "r") as f:
        #header = f.readline()
        #emb_dim = int(header.strip().split()[1])
        for line in f:
            arr = line.strip().split(' ')
            graph_node_id = arr[0]
            node_emb = [float(x) for x in arr[1:]]
            try:
                vocab_id = ent2id[graph_node_id]
            except KeyError:  # FIXME - replaced bare except
                vocab_id = int(graph_node_id)
            node_embeddings[vocab_id] = torch.tensor(node_emb)
            # print(torch.tensor(node_emb).size())

    num_nodes = len(ent2id)
    embedding_tensor = torch.empty(num_nodes, base_emb_dim)
    print("check", embedding_tensor.size(), len(node_embeddings))
    for i in range(num_nodes):
        # except is updated for KGAT format since some nodes are not used in the graph
        # there is no pre-trained node2vec ebmeddings for them
        try:
            embedding_tensor[i] = node_embeddings[i]
        except KeyError:  # FIXME - replaced bare except
            pass

    out = torch.tensor(embedding_tensor)
    out = embedding_tensor
    print("node2vec tensor", out.size())

    return out
