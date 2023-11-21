"""
    Acknowledgement:
    This code builds on top of the CompGCN (https://github.com/malllabiisc/CompGCN) and
    PyTorch Geometric's message passing implementation.
"""

import sys
import torch
from ..CompGCN.model.compgcn_conv import CompGCNConv


class GCNTransform(torch.nn.Module):
    """
    Transforms a list of subgraphs extracted from a heterogeneous graph to their
    vector representations through a multi-relational GCN transformation.
    """

    def __init__(self, emb_sz, num_gcn_layers, node_edge_composition_func):
        """
        :param emb_sz:         Dimension of node and relation embeddings. Must be same.
        :param num_gcn_layers: Number of GCN transforms/rounds of message propoagation
                               that will happen on the input in forward().
        :param node_edge_composition_func: 'no_rel'|'mult'|'sub'|'circ_conv'
        """
        super(GCNTransform, self).__init__()
        self.embedding_dims = emb_sz
        self.num_gcn_layers = num_gcn_layers
        self.__gcn_layers = []
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        for _ in range(num_gcn_layers):
            self.__gcn_layers.append(
                CompGCNConv(emb_sz, emb_sz, node_edge_composition_func).to(self.device)
            )
            # self.__conv1 = CompGCNConv(embed_dim, embed_dim, act=torch.tanh, params=self.p)

        self.drop = torch.nn.Dropout(0.3)
        return

    def __get_node_counts(self, subgraph_list):
        """
        Returns a list containing the number of nodes in each graph in the subgraph_list.
        """
        # print('subgraph_list in __get_node_counts', subgraph_list)
        node_counts = []
        for subgraph in subgraph_list:
            node_set = set()
            for edge in subgraph:
                node_set.add(edge[0])
                node_set.add(edge[1])
            node_counts.append(len(node_set))
        return node_counts

    def forward_subgraph_list(self, subgraph_list, in_node_emb, in_rel_emb):
        """
        Transforms a list of subgraphs [batch_sz] to their GCN transformed embeddings.
        :param subgraph_list:       A list of subgraphs where each subgraph is a list
                                    of (src, dst, relation) triples.
        :param in_node_emb: [batch_sz, max_nodes, emb_dim] dimension FloatTensor. max_nodes
                            refers to the maximum number of nodes across all subgraphs in
                            the subgraph_list. in_node_emb[i][j] will return the embedding
                            for node j in batch i.  This indices are zero based/normalized
                            to 0-(max_nodes-1).
        :param in_rel_emb:  [batch_sz, max_rels, emb_dim] dimension FloatTensor. max_rels
                            refers to the maximum number of relations across all subgraphs
                            in the subgraph_list. in_rel_emb[i][j] will return the embedding
                            for relation j in batch i. This indices are zero based/normalized
                            to 0-(max_relations-1).
        :ret out_node_emb:  Transformed version of in_node_emb with identical shape.
        :ret out_rel_emb:   Transformed version of in_rel_emb with identical shape.
        """
        # Obtain the number of nodes in each subgraph
        # node_counts = self.__get_node_counts(subgraph_list)
        # max_node_count = max(node_counts)
        out_node_emb = in_node_emb.clone()
        out_relation_emb = in_rel_emb.clone()

        for i, edge_list in enumerate(subgraph_list):
            x = out_node_emb[i]
            r = out_relation_emb[i]  # torch.cat([in_rel_emb, -in_rel_emb], dim=0)
            # Each subgraph in the subgraph_list is a list of (src, dst, relation)
            # MAJOR ASSUMPTION: the src/dst and relation ids are normalized such
            # that we can look up node_embeddings and relation_embeddings to fetch
            # corresponding embedding.

            # edge_index = [] # [(e[0], e[1]) for e in edge_list]
            # edge_types = [] # [e[2] for e in edge_list]
            # for e in edge_list:
            #     edge_index.append((e[0], e[1]))
            #     edge_types.append(e[2])
            # for e in edge_list:
            #     edge_index.append((e[1], e[0]))
            #     edge_types.append(e[2])

            # edge_index = torch.LongTensor(edge_index).t()
            # edge_types = torch.LongTensor(edge_types)

            # for i in range(self.num_gcn_layers):
            #     x, r = self.__gcn_layers[i](x, edge_index, edge_types, rel_embed=r)

            # x = self.drop(x)

            x, r = self.forward_subgraph(edge_list, x, r)
            out_node_emb[i] = x
            out_relation_emb[i] = r

        return out_node_emb, out_relation_emb

    def forward_subgraph(self, subgraph, in_node_emb, in_rel_emb):
        """
        Transforms a  subgraph's node and relation embeddings via GCN/Multi-Relational GCN.
        :param subgraph:    A list of (src, dst, relation) triples. Node ids are zero based
                            and normalized to range (0, number_of_nodes(subgraph)-1).
        :param in_node_emb: [N, emb_dim] dimension FloatTensor where N is at least number
                            of nodes in subgraph. in_node_emb[i] returns embedding for node i.
        :param in_rel_emb:  [N_e, emb_dim] dimension FloatTensor where N_e is the number of
                            distinct node types in subgraph/input graph. in_rel_emb[i] will
                            return the embedding for relation type i.
        :ret out_node_emb:  Transformed version of in_node_emb with identical shape.
        :ret out_rel_emb:   Transformed version of in_rel_emb with identical shape.
        """
        # Obtain the number of nodes in each subgraph
        # node_counts = self.__get_node_counts([subgraph])
        # max_node_count = max(node_counts)
        # out_node_emb = in_node_emb.clone()
        # out_relation_emb = in_rel_emb.clone()

        # edge_list = subgraph_list[i]
        edge_list = subgraph
        # Each subgraph in the subgraph_list is a list of (src, dst, relation)
        # MAJOR ASSUMPTION: the src/dst and relation ids are normalized such
        # that we can look up node_embeddings and relation_embeddings to fetch
        # corresponding embedding.
        edge_index = []  # [(e[0], e[1]) for e in edge_list]
        edge_types = []  # [e[2] for e in edge_list]
        for e in edge_list:
            edge_index.append((e[0], e[1]))
            edge_types.append(e[2])
        for e in edge_list:
            edge_index.append((e[1], e[0]))
            edge_types.append(e[2])

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_types = torch.LongTensor(edge_types).to(self.device)

        x = in_node_emb.clone().to(self.device)
        r = in_rel_emb.clone().to(self.device)

        for i in range(self.num_gcn_layers):
            x, r = self.__gcn_layers[i](x, edge_index, edge_types, rel_embed=r)

        x = self.drop(x)
        return x, r

    def forward(self, option, subgraph_list, in_node_emb, in_rel_emb):
        if option == "subgraph_list":
            return self.forward_subgraph_list(subgraph_list, in_node_emb, in_rel_emb)
        elif option == "subgraph":
            return self.forward_subgraph(subgraph_list, in_node_emb, in_rel_emb)
        else:
            print("Unknown option: %s" % option)
            sys.exit(1)
