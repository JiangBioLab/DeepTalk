import json
import os

import torch
from torch.autograd import Variable

from .mask_generation import GenerateGraphMask


def get_normalized_masked_ids(masked_node_ids, ent2id):
    """
    Input : [batch * list[masked_node_id]]
    returns : [batch * list(normalized_node_id) ] #len(attr-graph.get_number_of_nodes)]
    """

    normalized_node_ids = list()
    for batch_node_ids in masked_node_ids:
        normalized_node_ids_batch = [
            ent2id[str(x.cpu().numpy().tolist())] for x in batch_node_ids
        ]
        normalized_node_ids.append(normalized_node_ids_batch)

    return torch.LongTensor(normalized_node_ids)


class Processing_GCN_Walks:
    def __init__(self, nodeid2rowid, relations, n_pred, max_length, max_pred):

        self.nodeid2rowid = nodeid2rowid
        self.relations = relations
        self.n_pred = n_pred
        self.max_length = max_length
        self.max_pred = max_pred
        self.special_tokens = {}

    def getSpecialTokens(self):
        no_nodes = 0
        for node_type in self.nodeid2rowid:
            no_nodes += len(self.nodeid2rowid[node_type])
        special_tokens_set = ["[PAD]", "[MASK]"]
        for ii, _ in enumerate(special_tokens_set):
            self.special_tokens[special_tokens_set[ii]] = no_nodes + ii

        return self.special_tokens

    def masking_and_padding(self, subgraphs_list):
        for ii, _ in enumerate(subgraphs_list):
            # each subgraph has max_length edges
            subgraphs_list[ii] = subgraphs_list[ii][: self.max_length + 1]

        # generate mask
        graph_mask = GenerateGraphMask(subgraphs_list, self.n_pred)
        input_mask, masked_nodes, masked_position = graph_mask.GCN_MaskGeneration()
        # print('\n input mask:')
        # for ii in range(len(input_mask)):
        #     print(input_mask[ii])

        for ii, _ in enumerate(subgraphs_list):
            for jj in range(len(subgraphs_list[ii])):
                if input_mask[ii][jj] == 0:
                    subgraphs_list[ii][jj] = self.special_tokens["[MASK]"]
            # padding subgraph to max_length
            if len(subgraphs_list[ii]) < self.max_length + 1:
                n_pad = self.max_length + 1 - len(subgraphs_list[ii])
                for _ in range(n_pad):
                    subgraphs_list[ii].append(self.special_tokens["[PAD]"])

        return subgraphs_list, masked_nodes, masked_position

    def process_minibatch(self, data_path, data_name, task, batch_id):
        folder = os.path.join(data_path, task)
        file_name = data_name + "_" + task + "_batch_" + str(batch_id) + ".txt"
        fp = open(os.path.join(folder, file_name), "r")
        subgraphs = []
        cnt = 0
        for line in fp:
            line = json.loads(line)
            subgraphs.append(line)
            cnt += 1
            # if cnt == 3:
            #     break
        fp.close()

        all_nodes = []
        for subgraph in subgraphs:
            nodes = []
            for edge in subgraph:
                source_node = edge[0]
                target_node = edge[1]
                if int(source_node) not in nodes:
                    nodes.append(int(source_node))
                if int(target_node) not in nodes:
                    nodes.append(int(target_node))
            all_nodes.append(nodes)

        special_tokens = self.getSpecialTokens()
        all_nodes, masked_nodes, masked_postion = self.masking_and_padding(all_nodes)
        # print('\n subgraphs \n', subgraphs)
        # print('\n all_nodes \n', all_nodes)
        # print('\n masked_nodes\n', masked_nodes)
        # print('\n masked_postion\n', masked_postion)

        masked_nodes = Variable(torch.LongTensor(masked_nodes))
        masked_postion = Variable(torch.LongTensor(masked_postion))

        return subgraphs, all_nodes, masked_nodes, masked_postion

    def pad_max_seq_len(self, subgraphs_list, max_seq_len):
        padded_subgraphs = []
        for subgraph in subgraphs_list:
            subgraph_actual_len = len(subgraph)
            for _ in range(max_seq_len - subgraph_actual_len):
                subgraph.append(self.special_tokens["[PAD]"])
            padded_subgraphs.append(subgraph)

        return padded_subgraphs

    def process_finetune_minibatch(self, finetune_path, data_name, task, batch_id):
        """
        loads all subgraph for a given batch = batch_id , from
        finetune_path + "/" task + "/" + data_name+'_'+task+'_batch_'+str(batch_id)+'.txt'
        Expects each line of format
        [[source, target, relation, 0/1]]
        The last value indicate true or false edge
        """
        folder = os.path.join(finetune_path, task)
        file_name = data_name + "_" + task + "_batch_" + str(batch_id) + ".txt"
        fp = open(os.path.join(folder, file_name), "r")
        subgraphs = []
        labels = []
        cnt = 0
        for line in fp:
            line = json.loads(line)
            labels.append(int(line[0][-1]))
            line[0] = line[0][:-1]
            subgraphs.append(line)
            cnt += 1
            # if cnt == 3:
            #     break
        fp.close()

        # print("In fine tuning process batch, loaded walks", all_walks)
        all_nodes = []
        for subgraph in subgraphs:
            nodes = []
            for edge in subgraph:
                source_node = edge[0]
                target_node = edge[1]
                if source_node not in nodes:
                    nodes.append(source_node)
                if target_node not in nodes:
                    nodes.append(target_node)
            all_nodes.append(nodes)

        special_tokens = self.getSpecialTokens()
        all_nodes = self.pad_max_seq_len(all_nodes, self.max_length + 1)
        # print("created rowid subgraphs for batch", subgraphs_list, subgraph_len)
        labels = Variable(torch.FloatTensor(labels)).unsqueeze(1)
        # print(subgraphs, labels)
        #print(labels)

        return subgraphs, all_nodes, labels
