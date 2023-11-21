import numpy as np
import torch
import networkx as nx
from ..ccc_model.ccc_model import FinetuneLayer


class PathMetrics:
    # key: [source, target]
    # value: list of dictionary of [length, coherence, metapath, score, max_degree]
    #   shortest path between source and target, number of paths, number of metapaths
    def __init__(
        self,
        device,
        pretrained_node_embedding,
        G,
        ent2id,
        d_model,
        ft_d_ff,
        ft_layer,
        ft_drop_rate,
        attr_graph,
        ft_input_option,
        n_layers,
    ):
        self.path_metric_score = dict()
        self.pretrained_node_embedding = pretrained_node_embedding
        self.G = G
        self.ent2id = ent2id
        self.ft_input_option = ft_input_option
        self.ft_layer = ft_layer
        self.n_layers = n_layers
        self.ft_linear = FinetuneLayer(device,
            d_model,
            ft_d_ff,
            ft_layer,
            ft_drop_rate,
            attr_graph,
            ft_input_option,
            n_layers,
        )

    def update_batch_graphbert(
        self, subgraph_batch, scores, labels, graphbert_output, nodes_seq, train_stage
    ):
        for ii, _ in enumerate(subgraph_batch):
            edge = subgraph_batch[ii][0]
            source = edge[0]
            target = edge[1]
            relation = edge[2]
            label = int(labels[ii][0])
            path = subgraph_batch[ii][1:]
            length = len(path)
            pred_score = scores[ii][0]
            embedding = {}
            no_layers = graphbert_output.size(1)
            for jj in range(len(nodes_seq[ii])):
                if train_stage == "graphbert_pre" or self.ft_input_option == "last":
                    emb = graphbert_output[ii, jj, :].unsqueeze(0).unsqueeze(0)
                elif train_stage == "graphbert_ft":
                    # add for ablation study with n_layer = 0
                    if self.n_layers == 0:
                        start_layer = 0
                    else:
                        start_layer = no_layers - 4
                    for kk in range(start_layer, no_layers):
                        tmp = graphbert_output[ii, kk, jj, :].unsqueeze(0).unsqueeze(0)
                        if kk == start_layer:
                            emb = tmp
                        else:
                            if self.ft_input_option == "last4_cat":
                                emb = torch.cat((emb, tmp), 2)
                            elif self.ft_input_option == "last4_sum":
                                emb = torch.add(emb, 1, tmp)
                    if self.ft_layer == "ffn":
                        emb = torch.relu(self.ft_linear.dropout(self.ft_linear.ffn1(emb)))
                        emb = self.ft_linear.ffn2(emb)
                    elif self.ft_layer == "linear":
                        emb = torch.relu(self.ft_linear.dropout(self.ft_linear.ft_decoder(emb)))
                        #emb = self.ft_linear.ffn2(emb)
                embedding[nodes_seq[ii][jj]] = emb

            coherence, variance, min_score, metapath = self.get_path_metrics_graphbert(
                path, embedding
            )
            tmp_dict = {
                "label": label,
                "relation": relation,
                "score": pred_score,
                "path": path,
                "length": length,
                "coherence": coherence,
                "variance": variance,
                "min_score": min_score,
                "metapath": metapath,
            }
            tmp_key = source + "-" + target
            try:
                self.path_metric_score[tmp_key]["paths"].append(tmp_dict)
            except KeyError:  # FIXME - replaced bare except
                self.path_metric_score[tmp_key] = {
                    "shortest_path_length": self.get_shortest_path_length(
                        source, target
                    ),
                    "similarity_score": None,
                    "paths": [tmp_dict],
                }

    def update_batch(self, subgraph_batch, labels):
        for ii in range(len(subgraph_batch)):
            edge = subgraph_batch[ii][0]
            source = edge[0]
            target = edge[1]
            relation = edge[2]
            label = int(labels[ii][0])
            path = subgraph_batch[ii][1:]
            length = len(path)
            pred_score = self.get_similarity_score(source, target)
            coherence, variance, min_score, metapath = self.get_path_metrics(
                source, target, path
            )
            tmp_dict = {
                "label": label,
                "relation": relation,
                "score": pred_score,
                "path": path,
                "length": length,
                "coherence": coherence,
                "variance": variance,
                "min_score": min_score,
                "metapath": metapath,
            }
            tmp_key = source + "-" + target
            try:
                self.path_metric_score[tmp_key]["paths"].append(tmp_dict)
            except KeyError:  # FIXME - replaced bare except
                self.path_metric_score[tmp_key] = {
                    "shortest_path_length": self.get_shortest_path_length(
                        source, target
                    ),
                    "similarity_score": self.get_similarity_score(source, target),
                    "paths": [tmp_dict],
                }

    def get_shortest_path_length(self, source, target):
        try:
            length = nx.shortest_path_length(self.G, source, target)
        except:  # FIXME - need to replace bare except
            length = None

        return length

    def get_similarity_score(self, source, target):
        try:
            source_id = self.ent2id[source]
            target_id = self.ent2id[target]
        except KeyError:  # FIXME - replaced bare except
            source_id = int(source)
            target_id = int(target)
        # print("source, target", source, target)
        try:
            source_vec = self.pretrained_node_embedding[source_id]
            target_vec = self.pretrained_node_embedding[target_id]
            source_vec = source_vec.unsqueeze(0).unsqueeze(0)
            target_vec = target_vec.unsqueeze(0).unsqueeze(0).transpose(1, 2)
        except (KeyError, TypeError) as e:  # FIXME - replaced bare except
            source_vec = self.pretrained_node_embedding(source_id).unsqueeze(0)
            target_vec = self.pretrained_node_embedding(target_id).unsqueeze(0)
            target_vec = target_vec.transpose(1, 2)
        score = torch.bmm(source_vec, target_vec)
        score = torch.sigmoid(score).data.cpu().numpy().tolist()[0][0][0]

        return score

    def get_path_metrics(self, source, target, path):
        """
        Metrics per node pair:
            1) Number of paths connecting the nodes
            2) Number of metapaths
            3) Shifting of embeddings from static to contextual embeddings
            4) Distribution of pairwise attention for node embeddings in a context
        """
        score_list = []
        for edge in path:
            score = self.get_similarity_score(edge[0], edge[1])
            score_list.append(score)
        coherence = np.mean(score_list)
        variance = np.var(score_list)
        try:
            min_score = np.min(score_list)
        except:  # FIXME - need to replace bare except
            min_score = None

        metapath = "_".join([relation for (_, _, relation) in path])

        return coherence, variance, min_score, metapath

    def get_path_metrics_graphbert(self, path, embedding):
        score_list = []
        for edge in path:
            source_vec = embedding[edge[0]]
            target_vec = embedding[edge[1]]
            target_vec = target_vec.transpose(1, 2)
            # print(source_vec.size(), target_vec.size())
            score = torch.bmm(source_vec, target_vec)
            score = torch.sigmoid(score).data.cpu().numpy().tolist()[0][0][0]
            score_list.append(score)
        coherence = np.mean(score_list)
        variance = np.var(score_list)
        try:
            min_score = np.min(score_list)
        except:  # FIXME - need to replace bare except
            min_score = None

        metapath = "_".join([relation for (_, _, relation) in path])

        return coherence, variance, min_score, metapath

    def finalize(self):
        for kk in self.path_metric_score:
            no_paths = len(self.path_metric_score[kk]["paths"])
            metapath_set = []
            for pp in self.path_metric_score[kk]["paths"]:
                metapath = pp["metapath"]
                metapath_set.append(metapath)
            metapath_set = list(set(metapath_set))
            self.path_metric_score[kk]["number_of_paths"] = no_paths
            self.path_metric_score[kk]["number_of_metapaths"] = len(metapath_set)
            self.path_metric_score[kk]["metapath_set"] = metapath_set

        return self.path_metric_score

        # counts_by_metapath = Counter()
        # # coh_by_metapath = defaultdict(list)
        # path_coherence_list = []

        # # A path p is a list of edges/triples of form (src, dst, rel)
        # metapath = '_'.join([relation for (_,_,relation) in path])
        # counts_by_metapath[metapath] += 1
        # contextual_coh = get_pairwise_dot_product_mean_using_static_embeddings(path)
        # coh_by_metapath[metapath].append(contextual_coh)
        # path_coherence_list.append(contextual_coh)

        # coh_summary_by_metapath = dict()
        # for metapath, coherence_list in coh_by_metapath.items():
        #     coh_summary_by_metapath[metapath] = (np.mean(coherence_list),\
        #             np.std(coherence_list))

        # num_paths = len(paths)
        # num_metapaths = len(counts_by_metapath)

        # # Distance betweeen source and target - based off CompGCN embeddings
        # emb_dist_1 = get_cosine_dist_by_static_emb(source, target)
        # # Distance betweeen source and target - based off GraphBERT embeddings
        # emb_dist_2 = get_cosine_dist_by_dynamic_emb(source, target)
        # emb_shift_from_ccc = emb_dist_2 - emb_dist_1

        # # Looking at divergence from static CompGCN embedding and the dynamically
        # # generated Graph-BERT embedding
        # #emb_dist_src = get_emb_shift_from_static_to_bert(source)
        # #emb_dist_dst = get_emb_shift_from_static_to_bert(target)

        # coh_mean = np.mean(path_coherence_list)
        # coh_std = np.std(path_coherence_list)
        # return num_paths, num_metapaths, emb_shift_from_ccc, coh_mean, coh_std, \
        #         counts_by_metapath, coh_summary_by_metapath

        # Mahantesh - Sutanay - Ping - Pin-Yu - Baichuan
        # emb(Mahantesh) = emb(Mahanteh) + f_dot(Mahantesh, Sutanay) + f_dot(Mahantesh, Ping) \
        #         + f_dot(Mahantesh, Pin-Yu) + f_dot(Mahantesh, Baichuan)
        # emb(Mahantesh) = emb(Mahanteh) + c_MS + c_MP + c_MPY + c_MB

        # paper_5 - Ping - Chandan - WebConf - QA

        # paper_5 - Ping - paper_9 ------

        # src, dst, y_true, y_pred, score, feature1, feature2, .....

        # return coherence, metapath, max_degree
