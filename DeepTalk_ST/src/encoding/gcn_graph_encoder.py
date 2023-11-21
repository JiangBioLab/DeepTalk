import torch
from torch.autograd import Variable

from .input_encoder import NodeEncoder


def get_norm_id(id_map, some_id):
    if some_id not in id_map:
        id_map[some_id] = len(id_map)
    return id_map[some_id]


def norm_graph(node_id_map, edge_id_map, edge_list):
    norm_edge_list = []
    for e in edge_list:
        norm_edge_list.append(
            (
                get_norm_id(node_id_map, e[0]),
                get_norm_id(node_id_map, e[1]),
                get_norm_id(edge_id_map, e[2]),
            )
        )
    return norm_edge_list


def norm_context_subgraphs_for_gcn_encoding(subgraph_list):
    """
    Function to transform a subgraph_list (same data structure passed to
    GraphBERT.forward() to a tuple of data structures that will be fed
    into GCN.
    Purpose of this data structure: Given a node embedding tensor from GCN
    or BERT, we want to know which node in a subgraph it corresponds to.

    param subgraph_list: A list of subgraphs of len batch_sz. Each subgraph
                         is a list of triples of form (src, dst, edge_type)
    ret norm_subgraph_list: A transformed version of subgraph_list where
                         each node id is normalized. See picture below for
                         example. Nodes mentioned in () and relations in [].
                               (0)     (5)              (0)     (2)
                                 \     /                  \     /
                                 [0]  [2]                 [0]  [1]
                                   \ /                      \ /
                                   (2)                      (1)
                              subgraph_list[i]      norm_subgraph_list[i]
    ret node_idx:        A (batch_sz, max_nodes) tensor. node_idx[i]
                         would return a tensor containing normalized node
                         ids for subgraph subgraph_list[i].
                         max_nodes is the maximum number of nodes present
                         in any of the subgraphs in subgraph_list.
                         If subgraph_list[i] has N_i nodes, then
                         node_idx[i][0:(N_i-1)] will return corresponding
                         node_ids in subgraph_list[i].
    ret edge_type_idx:   Similar (batch_sz, max_edge_types) tensor that
                         maps edge types found in subgraph_list[i] to a
                         range (0, num_unique_relations(subgraph_list[i]-1))
    ret batch_counts:    A list of pairs containing the number of nodes
                         and edges for each entry in subgraph_list
    """
    num_subgraphs = len(subgraph_list)
    norm_subgraph_list = []
    batch_counts = []
    batch_id_maps = []

    for i in range(num_subgraphs):
        node_id_map = dict()
        edge_type_map = dict()
        norm_subgraph_list.append(
            norm_graph(node_id_map, edge_type_map, subgraph_list[i])
        )
        batch_id_maps.append((node_id_map, edge_type_map))
        batch_counts.append((len(node_id_map), len(edge_type_map)))

    max_nodes = max(map(lambda counts: counts[0], batch_counts))
    max_relations = max(map(lambda counts: counts[1], batch_counts))
    node_idx = [[None for y in range(max_nodes)] for x in range(num_subgraphs)]
    # node_idx = torch.empty(num_subgraphs, max_nodes)
    edge_idx = torch.empty(num_subgraphs, max_relations)

    for i in range(num_subgraphs):
        node_id_map = batch_id_maps[i][0]
        edge_type_map = batch_id_maps[i][1]
        for orig_node_id in sorted(node_id_map.keys()):
            node_idx[i][node_id_map[orig_node_id]] = orig_node_id
        for orig_edge_type in sorted(edge_type_map.keys()):
            edge_idx[i][edge_type_map[orig_edge_type]] = orig_edge_type

    return norm_subgraph_list, node_idx, edge_idx, batch_counts


class GCNGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        device,
        G,
        pretrained_node_embedding_tensor,
        is_pre_trained,
        base_embedding_dim,
        max_length,
        ent2id,
        rel2id,
    ):

        super().__init__()
        self.attr_graph = G
        self.base_embedding_dim = base_embedding_dim
        self.max_length = max_length
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.no_nodes = self.attr_graph.get_number_of_nodes()
        self.no_relations = self.attr_graph.get_number_of_relations()
        # print('check *************', self.no_relations)

        self.node_embedding = NodeEncoder(
            base_embedding_dim,
            self.no_nodes,
            pretrained_node_embedding_tensor,
            is_pre_trained,device
        )
        self.relation_embed = torch.nn.Embedding(self.no_relations, base_embedding_dim)
        # updated for KGAT data
        # torch.nn.Embedding(len(self.rel2id), base_embedding_dim) 
        self.relation_embed.weight.data.uniform_(-1, 1)

        self.special_tokens = {"[PAD]": 0, "[MASK]": 1}
        self.special_embed = torch.nn.Embedding(
            len(self.special_tokens), base_embedding_dim
        )
        self.special_embed.weight.data.uniform_(-1, 1)

    def forward(self, subgraphs_list, masked_nodes):
        num_subgraphs = len(subgraphs_list)
        norm_subgraph_list = []
        batch_counts = []
        batch_id_maps = []

        for ii, _ in enumerate(subgraphs_list):
            node_id_map = dict()
            edge_type_map = dict()
            norm_subgraph_list.append(
                norm_graph(node_id_map, edge_type_map, subgraphs_list[ii])
            )
            batch_id_maps.append((node_id_map, edge_type_map))
            batch_counts.append((len(node_id_map), len(edge_type_map)))

        # max_nodes = max(map(lambda counts: counts[0], batch_counts))
        max_relations = max(map(lambda counts: counts[1], batch_counts))

        node_emb = torch.zeros(
            num_subgraphs, self.max_length + 1, self.base_embedding_dim
        )
        #print(node_emb.shape)
        relation_emb = torch.zeros(
            num_subgraphs, max_relations, self.base_embedding_dim
        )
        #print(self.node_embedding(2397))
        for ii in range(len(subgraphs_list)):
            node_id_map = batch_id_maps[ii][0]
            edge_type_map = batch_id_maps[ii][1]
            masked_set = masked_nodes[ii].cpu().numpy().tolist()

            for node_id, norm_node_id in node_id_map.items():
                if node_id not in masked_set:  # used to ignore the masked nodes
                    try:
                        normalized_node_id = self.ent2id[node_id]
                    except KeyError:  # FIXME - replaced bare except
                        normalized_node_id = int(node_id)
                    #print(normalized_node_id)

                    node_emb[ii][norm_node_id] = self.node_embedding(normalized_node_id)

            # print('edge_type_map\n', edge_type_map)
            for relation_id, norm_relation_id in edge_type_map.items():
                try:
                    tmp = Variable(torch.LongTensor([int(self.rel2id[relation_id])]))
                except KeyError:  # FIXME - replaced bare except
                    tmp = Variable(torch.LongTensor([int(relation_id) - 1]))

                relation_emb[ii][norm_relation_id] = self.relation_embed(tmp)

        # get embeddings for special tokens
        # will be used for masking and padding before bert layer
        special_tokens_embed = {}
        for token in self.special_tokens:
            node_id = Variable(torch.LongTensor([self.special_tokens[token]]))
            tmp_embed = self.special_embed(node_id)
            special_tokens_embed[self.special_tokens[token] + self.no_nodes] = {
                "token": token,
                "embed": tmp_embed,
            }

        return (
            norm_subgraph_list,
            node_emb,
            relation_emb,
            batch_id_maps,
            special_tokens_embed,
        )
