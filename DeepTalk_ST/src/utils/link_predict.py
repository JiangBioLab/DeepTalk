import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .evaluation import run_evaluation_main


class LinkPrediction(torch.nn.Module):
    """
    Implementation of link prediction strategies based on src
    and dst node embeddings. Ranges from dot product based implementation
    to TransE like approach.
    """

    def __init__(self, opt, ent2id, rel2id, pretrained_node_embeddings):
        """
        Args:
            opt: specify the link prediction strategy.
        """
        super(LinkPrediction, self).__init__()
        self.opt = opt
        self.ent2id = ent2id
        self.pretrained_node_embeddings = pretrained_node_embeddings

    def __predict_via_dot_product(self, edge):
        try:
            source_id = self.ent2id[edge[1]]
            target_id = self.ent2id[edge[2]]
        except KeyError:
            source_id = int(edge[1])
            target_id = int(edge[2])
        source_vec = self.pretrained_node_embeddings[source_id]
        target_vec = self.pretrained_node_embeddings[target_id]
        source_vec = source_vec.unsqueeze(0).unsqueeze(0)
        target_vec = target_vec.unsqueeze(0).unsqueeze(0).transpose(1, 2)
        score = torch.bmm(source_vec, target_vec)
        score = torch.sigmoid(score).data.cpu().numpy().tolist()[0][0][0]

        return score

    def forward(self, triple):
        if self.opt == "dot_product":
            score = self.__predict_via_dot_product(triple)

        return score


def get_node_embeddings(path: str):
    node_embeddings = {}
    header = True
    num_nodes = 0
    size_embedding = 0
    line_count = 0
    with open(path, "r") as f:
        for line in f:
            line_count += 1
            tokens = line.strip().split(" ")
            if header:
                num_nodes = int(tokens[0])
                size_embedding = int(tokens[1])
                print(f"Num nodes: {num_nodes}")
                print(f"Size embedding vector: {size_embedding}")
                header = False
                continue
            node_id = str(tokens[0])
            embedding_vector = np.asarray(tokens[1:], dtype="float32")
            # torch.tensor(np.asarray(tokens[1:], dtype='float32'))
            node_embeddings[node_id] = embedding_vector

    print(f"Num lines: {line_count}")
    return node_embeddings


def get_node_vals(line: str, data_format: str):
    label = None
    tokens = line.strip().split(" ")
    etype = tokens[0]
    node1 = tokens[1]
    node2 = tokens[2]
    # Don't include non-edges for test or valid data
    if data_format in ("valid", "test"):
        label = tokens[3]

    return etype, node1, node2, label


def get_test_edges(paths: List[str]):
    edges = set()
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                etype, source, destination, label = get_node_vals(line, "test")
                edge = (etype, source, destination, label)
                edges.add(edge)
    print(f"Num test edges: {len(edges)}")
    return edges


def get_scores(node_embeddings: dict, test_edges, poly: bool = False, facets: int = 5):
    # Get array of node vector embeddings
    if poly:
        source_embeddings = [node_embeddings[int(edge[1])] for edge in test_edges]
        destination_embeddings = [node_embeddings[int(edge[2])] for edge in test_edges]
    else:
        source_embeddings = [node_embeddings[edge[1]] for edge in test_edges]
        destination_embeddings = [node_embeddings[edge[2]] for edge in test_edges]

    if poly:
        source_embeddings_combo = []
        destination_embeddings_combo = []
        for i, source_embedding in enumerate(source_embeddings):
            destination_embedding = destination_embeddings[i]
            for source_embedding_item in source_embedding:
                for destination_embedding_item in destination_embedding:
                    source_embeddings_combo.append(source_embedding_item)
                    destination_embeddings_combo.append(destination_embedding_item)
        source_embeddings = source_embeddings_combo
        destination_embeddings = destination_embeddings_combo

    # Convert arrays to pytorch tensors
    source_embeddings = torch.Tensor(source_embeddings)
    destination_embeddings = torch.Tensor(destination_embeddings)

    # Get prediction scores
    scores = torch.sigmoid(
        torch.bmm(
            source_embeddings.unsqueeze(dim=1),
            destination_embeddings.unsqueeze(dim=1).transpose(1, 2),
        )
    )
    return scores


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t["threshold"])


def test_link_prediction(
    test_edges: list, node_embeddings: dict, poly: bool = False, facets: int = 1
):
    true_list = []
    prediction_list = []
    true_num = 0
    n_facet_combos = facets ** 2

    labels = [int(edge[3]) for edge in test_edges]
    true_edge_indexes = [i for i, label in enumerate(labels) if label == 1]
    false_edge_indexes = [i for i, label in enumerate(labels) if label == 0]
    scores = get_scores(node_embeddings, test_edges, poly)

    print(f"Num true edges: {len(true_edge_indexes)}")
    for ti in true_edge_indexes:
        start = ti * n_facet_combos
        end = start + n_facet_combos
        tmp_score = float(max(scores[start:end]))
        true_list.append(1)
        prediction_list.append(tmp_score)
        true_num += 1

    print(f"Num false edges: {len(false_edge_indexes)}")
    for fi in false_edge_indexes:
        start = fi * n_facet_combos
        end = start + n_facet_combos
        tmp_score = float(max(scores[start:end]))
        true_list.append(0)
        prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()

    y_true = np.array(true_list)
    threshold = find_optimal_cutoff(y_true, prediction_list)[0]
    # This was the original threshold approximator
    # threshold2 = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i, _ in enumerate(prediction_list):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    print(f"y_true: {y_true}")
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    # Note we only have one edge_type so we don't have to take the mean
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = auc(rs, ps)
    f1 = f1_score(y_true, y_pred)
    print(f"ROC-AUC: {roc_auc}")
    print(f"PR-AUC: {pr_auc}")
    print(f"F1: {f1}")


def get_poly_embeddings(path: str):
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def avg_poly_embeddings(poly_embeddings):
    poly_embeddings_shape = poly_embeddings.shape
    num_samples = poly_embeddings_shape[0]
    embedding_size = poly_embeddings_shape[2]
    avg_poly_embeddings = np.ndarray((num_samples, embedding_size))
    for n in range(num_samples):
        emb_item = poly_embeddings[n]
        # avg_emb_item = np.average(emb_item, axis=0)
        # min_emb_item = np.amin(emb_item, axis=0)
        # max_emb_item = np.amax(emb_item, axis=0)
        sum_emb_item = np.sum(emb_item, axis=0)
        avg_poly_embeddings[n] = sum_emb_item
    print()
    return avg_poly_embeddings


def link_prediction_eval(
    valid_edges, test_edges, ent2id, rel2id, pretrained_node_embeddings
):
    """
    Args:
        test_edges (List[triple]): list of edges loaded from test.txt
        node_embedding (NodeEmbedding): class implementing NodeEmbedding
                interface

    Returns:
        Precision, Recall, F-Score for link prediction test
    """
    link_prediction = LinkPrediction(
        "dot_product", ent2id, rel2id, pretrained_node_embeddings
    )

    pred_data = {}
    true_data = {}
    for data_type in ["valid", "test"]:
        pred_data[data_type] = []
        true_data[data_type] = []
        if data_type == "valid":
            edges = valid_edges
        else:
            edges = test_edges
        for edge in edges:
            label = edge[3]
            true_data[data_type].append(int(label))
            score = link_prediction(edge)
            pred_data[data_type].append(score)

    valid_true_data = np.array(true_data["valid"])
    threshold = find_optimal_cutoff(valid_true_data, pred_data["valid"])[0]
    run_evaluation_main(
        test_edges, pred_data["test"], true_data["test"], threshold, header="workflow1"
    )
