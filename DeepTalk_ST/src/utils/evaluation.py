from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def run_evaluation(pred_data, true_data):
    """For evaluation."""
    pred_data = np.array(pred_data)
    true_data = np.array(true_data)

    print("pred and true label size", pred_data.shape, true_data.shape)
    pred_seq = []
    true_seq = []
    for ii in range(len(pred_data)):
        pred_seq += [pred_data[ii][jj] for jj in range(len(pred_data[ii]))]
        true_seq += [true_data[ii][jj] for jj in range(len(true_data[ii]))]
    print("pred \n", len(pred_seq))  # , pred_seq)
    # print(pred_seq)
    print("true \n", len(true_seq))  # , true_seq)
    # print(true_seq)

    accu = accuracy_score(true_seq, pred_seq)
    mse = mean_squared_error(true_seq, pred_seq)
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)
    print("Accuracy={}, MSE={}".format(accu, mse))
    return accu, mse


def run_evaluation_graph(pred_data, true_data, test_edges, attr_graph):
    """For evaluation."""
    pred_data = np.array(pred_data)
    true_data = np.array(true_data)

    pred_seq = []
    true_seq = []
    count_correct_by_relation_type = Counter()
    count_error_by_relation_type = Counter()

    for ii in range(len(pred_data)):
        pred_seq += [pred_data[ii][jj] for jj in range(len(pred_data[ii]))]
        true_seq += [true_data[ii][jj] for jj in range(len(true_data[ii]))]

    print("pred \n", len(pred_seq))  # , pred_seq)
    # print(pred_seq)
    print("true \n", len(true_seq))  # , true_seq)
    # print(true_seq)
    # print(test_edges[0][0])
    for ii, subgraph in enumerate(test_edges):
        test_edge = subgraph[0]
        # print(test_edge)
        test_relation = test_edge[1]
        if pred_seq[ii] == true_seq[ii]:
            count_correct_by_relation_type[test_relation] += 1
        else:
            count_error_by_relation_type[test_relation] += 1

    print("Correct counts by relation", count_correct_by_relation_type)
    print("Incorrect counts by relation", count_error_by_relation_type)

    print(
        "Precision , recall, fscore-micro",
        precision_recall_fscore_support(true_seq, pred_seq, average="micro"),
    )
    print(
        "Precision , recall, fscore-macro",
        precision_recall_fscore_support(true_seq, pred_seq, average="macro"),
    )

    print(
        "Precision , recall, fscore-weighted",
        precision_recall_fscore_support(true_seq, pred_seq, average="weighted"),
    )

    accu = accuracy_score(true_seq, pred_seq)
    mse = mean_squared_error(true_seq, pred_seq)
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)

    print("Accuracy={}, MSE={}".format(accu, mse))

    return accu, mse


def run_evaluation_main(test_edges, prediction_data, true_data, threshold, header):
    sorted_pred = prediction_data[:]
    sorted_pred.sort()
    # threshold = sorted_pred[-true_num]
    y_pred = np.zeros(len(prediction_data), dtype=np.int32)

    for i, _ in enumerate(prediction_data):
        if prediction_data[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_data)
    y_scores = np.array(prediction_data)

    ps, rs, _ = precision_recall_curve(y_true, y_scores)

    print("----------------------In run_evaluation_main()------------")
    print(
        f"y_true.shape: {y_true.shape}, y_scores.shape: {y_scores.shape}"
        f", y_pred.shape: {y_pred.shape}"
    )
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        roc_auc = 'UNDEFINED'
    f1 = f1_score(y_true, y_pred)
    auc_value = auc(rs,ps)
    print(
        f"{header} : ROC-AUC: {roc_auc},"
        f" F1: {f1}, AUC: {auc_value}"
    )
    print("\nEvaluation by edge type: ")

    y_pred_dict = {}
    y_true_dict = {}
    y_score_dict = {}
    # print(len(prediction_data), len(true_data), len(test_edges))
    for ii, _ in enumerate(test_edges):
        edge_type = test_edges[ii][0]
        try:
            y_pred_dict[edge_type].append(y_pred[ii])
            y_true_dict[edge_type].append(y_true[ii])
            y_score_dict[edge_type].append(y_scores[ii])
        except KeyError:
            y_pred_dict[edge_type] = [y_pred[ii]]
            y_true_dict[edge_type] = [y_true[ii]]
            y_score_dict[edge_type] = [y_scores[ii]]

    for itm in y_pred_dict:
        print("Performance on edge type: ", itm)
        y_true_dict[itm] = np.array(y_true_dict[itm])
        y_pred_dict[itm] = np.array(y_pred_dict[itm])
        y_score_dict[itm] = np.array(y_score_dict[itm])
        ps, rs, _ = precision_recall_curve(y_true_dict[itm], y_score_dict[itm])
        print(
            f"y_true.shape: {y_true_dict[itm].shape},"
            f" y_scores.shape: {y_score_dict[itm].shape},"
            f" y_pred.shape: {y_pred_dict[itm].shape}"
        )
        try:
            roc_auc = roc_auc_score(y_true_dict[itm], y_score_dict[itm])
        except (KeyError, ValueError):
            roc_auc = 'UNDEFINED'
        f1 = f1_score(y_true_dict[itm], y_pred_dict[itm])
        auc_value = auc(rs,ps)
        print(
                f"{header} : ROC-AUC: "
                f"{roc_auc},"
                f" F1: {f1},"
                f" PR-AUC: {auc_value}"
            )
    print("END")
    return y_true,y_scores

