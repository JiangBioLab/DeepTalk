import json
import os
import pickle
import shutil
import time

import numpy as np
import torch
import torch.nn

import torch.optim as optim
from torch.autograd import Variable

from .ccc_model.ccc_model import CCC, FinetuneLayer
from .ccc_model.process_walks_gcn import Processing_GCN_Walks
from .utils.context_metrics import PathMetrics
from .utils.data_utils import create_finetune_batches2
from .utils.utils import load_pickle, show_progress

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def setup_finetuning_input(args, attr_graph, context_gen):
    # finetune
    print("\n Generate data for finetuning ...")
    num_batches = {}
    finetune_path = args.outdir + "/finetune/"
    if not os.path.exists(finetune_path):
        os.mkdir(finetune_path)

    if os.path.exists(finetune_path + "/finetune_walks.txt"):
        finetune_walks_per_task = json.load(open(finetune_path + "/finetune_walks.txt"))
    else:

        (
            train_edges,
            valid_edges,
            test_edges,
        ) = attr_graph.generate_link_prediction_dataset(
            finetune_path, fr_valid_edges=0.1, fr_test_edges=0.1
        )
        finetune_walks_per_task = context_gen.get_finetune_subgraphs(
            train_edges,
            valid_edges,
            test_edges,
            args.beam_width,
            args.max_length,
            args.path_option,
        )
        json.dump(
            finetune_walks_per_task, open(finetune_path + "/finetune_walks.txt", "w")
        )

        print(
            "Number of train, valid, test",
            len(train_edges),
            len(valid_edges),
            len(test_edges),
        )
    num_batches = create_finetune_batches2(
        finetune_walks_per_task, finetune_path, args.data_name, args.ft_batch_size
    )
    print("No. of batches for finetuning:", num_batches)

    return num_batches


def run_finetuning_wkfl2(
    args, attr_graph, no_batches, pretrained_node_embedding, ent2id, rel2id, epoch=-1
):

    # data_path = args.data_path + args.data_name + "/"
    outdir = args.outdir + "/"
    finetune_path = outdir + "finetune/"
    ft_out_dir = finetune_path + "results"
    try:
        shutil.rmtree(ft_out_dir)
        os.mkdir(ft_out_dir)
    except:  # FIXME - need to replace bare except
        os.mkdir(ft_out_dir)
    relations = attr_graph.relation_to_id
    nodeid2rowid = attr_graph.get_nodeid2rowid()
    walk_processor = Processing_GCN_Walks(
        nodeid2rowid, relations, args.n_pred, args.max_length, args.max_pred
    )

    # process minibatch for finetune(ft)
    print("\n processing minibaches for finetuning:")
    ft_batch_input_file = os.path.join(
        finetune_path, args.data_name + "_ft_batch_input.pickled"
    )
    if os.path.exists(ft_batch_input_file):
        print("loading saved files ...")
        ft_batch_input = load_pickle(ft_batch_input_file)
    else:
        ft_batch_input = {}
        tasks = ["train", "valid", "test"]
        for task in tasks:
            print(task)
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

    print("Begin finetuning")
    # load pre-trained model
    fbest = open(os.path.join(outdir, "best_epoch_id.txt"), "r")
    for line in fbest:
        tmp = line
    best_epoch = int(tmp[:-1])
    print("best_epoch: ", best_epoch)
    fbest.close()
    if epoch == -1:
        fl_ = os.path.join(outdir, "bert_{}.model".format(best_epoch))
    else:
        fl_ = os.path.join(outdir, "bert_{}.model".format(epoch))
    print("LOADING PRE_TRAIN model from ", fl_)
    ccc.load_state_dict(
        torch.load(fl_, map_location=lambda storage, loc: storage)
    )

    # run in fine tuning mode
    ccc.set_fine_tuning()
    # testing
    print("Begin Testing")
    ccc.eval()

    pred_data = {}
    true_data = {}
    path_metrics_graphbert_pre = PathMetrics(
        args.device,
        None,
        attr_graph.G,
        ent2id,
        args.d_model,
        args.ft_d_ff,
        args.ft_layer,
        args.ft_drop_rate,
        attr_graph,
        args.ft_input_option,
        args.n_layers,
    )
    with torch.no_grad():
        for data_type in ["valid", "test"]:
            pred_data[data_type] = []
            true_data[data_type] = []
            for batch_id in range(no_batches[data_type]):
                if batch_id % 100 == 0:
                    print("Evaluating {} batch {}".format(data_type, batch_id))
                subgraphs, all_nodes, labels = ft_batch_input[data_type][batch_id]

                masked_pos = torch.randn(args.batch_size, 2)
                masked_nodes = Variable(
                    torch.LongTensor([[] for ii in range(args.ft_batch_size)])
                )
                output, _, _ = ccc(
                    subgraphs, all_nodes, masked_pos, masked_nodes
                )
                source_embed = output[:, 0, :].unsqueeze(1)
                target_embed = output[:, 1, :].unsqueeze(1).transpose(1, 2)
                score = torch.bmm(source_embed, target_embed).squeeze(1)
                #score = torch.sigmoid(score).data.cpu().numpy().tolist()
                labels = labels.data.cpu().numpy().tolist()
                if data_type == "test":
                    path_metrics_graphbert_pre.update_batch_graphbert(
                        subgraphs, score, labels, output, all_nodes, "graphbert_pre"
                    )

                for ii, _ in enumerate(score):
                    pred_data[data_type].append(score[ii][0])
                    true_data[data_type].append(labels[ii][0])
                # print(len(score), len(labels))

        path_dict = path_metrics_graphbert_pre.finalize()
        '''
        json.dump(
            path_dict, open(outdir + "path_metrics_graphbert_pre.json", "w"), indent=4
        )
        '''

    return pred_data, true_data


def run_finetuning_wkfl3(
    args, attr_graph, no_batches, pretrained_node_embedding, ent2id, rel2id
):

    # data_path = args.data_path + args.data_name + "/"
    outdir = args.outdir + "/"
    finetune_path = outdir + "finetune/"
    ft_out_dir = finetune_path + "results"
    try:
        shutil.rmtree(ft_out_dir)
        os.mkdir(ft_out_dir)
    except:  # FIXME - need to replace bare except
        os.mkdir(ft_out_dir)
    relations = attr_graph.relation_to_id
    nodeid2rowid = attr_graph.get_nodeid2rowid()
    walk_processor = Processing_GCN_Walks(
        nodeid2rowid, relations, args.n_pred, args.max_length, args.max_pred
    )

    # process minibatch for finetune(ft)
    print("\n processing minibaches for finetuning:")
    ft_batch_input_file = os.path.join(
        finetune_path, args.data_name + "_ft_batch_input.pickled"
    )
    if os.path.exists(ft_batch_input_file):
        print("loading saved files ...")
        ft_batch_input = load_pickle(ft_batch_input_file)
    else:
        ft_batch_input = {}
        tasks = ["train", "valid", "test"]
        for task in tasks:
            print(task)
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
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    #criterion = FocalLoss(reduction="mean")
    #criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = optim.Adam(ft_linear.parameters(), args.ft_lr,weight_decay=0.0001)

    print("Begin finetuning")
    # load pre-trained model
    fbest = open(os.path.join(outdir, "best_epoch_id.txt"), "r")
    for line in fbest:
        tmp = line
    best_epoch = int(tmp[:-1])
    print("best_epoch in pretraining: ", best_epoch)
    fbest.close()
    fl_ = os.path.join(outdir, "bert_{}.model".format(best_epoch))
    print("LOADING PRE_TRAIN model from ", fl_)
    ccc.load_state_dict(
        torch.load(fl_, map_location=lambda storage, loc: storage)
    )

    # run in fine tuning mode
    ccc.set_fine_tuning()

    print("\n Begin Training")
    loss_dev_final = []
    for epoch in range(args.ft_n_epochs):
        loss_arr = []
        loss_dev_min = 1e6
        print("\nFinetune epoch: ", epoch)
        start_time = time.time()
        for batch_id in range(no_batches["train"]):
            subgraphs, all_nodes, labels = walk_processor.process_finetune_minibatch(
                finetune_path, args.data_name, "train", batch_id
            )

            with torch.no_grad():
                # random mask
                masked_pos = torch.randn(args.batch_size, 2)
                masked_nodes = Variable(
                    torch.LongTensor([[] for ii in range(args.ft_batch_size)])
                )
                _, layer_output, _ = ccc(
                    subgraphs, all_nodes, masked_pos, masked_nodes
                )
            pred_scores, _, _ = ft_linear(layer_output)
            # print(" Size of ypred, ytrue", pred_scores.size(), labels.size())
            loss = criterion(pred_scores.squeeze(1), labels.squeeze(1).to(args.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.data.cpu().numpy().tolist())

            ccnt = batch_id % args.ft_checkpoint
            if batch_id > 0 and batch_id % args.ft_checkpoint == 0:
                ccnt = args.ft_checkpoint
            '''
            message = "Loss: {}, AvgLoss: {}{}".format(
                np.around(loss.data.cpu().numpy().tolist(), 4),
                np.around(np.average(loss_arr).tolist(), 4),
                " " * 10,
            )
            
            show_progress(
                ccnt, min(args.ft_checkpoint, no_batches["train"]), message
            )
            '''
            if (
                batch_id % args.ft_checkpoint == 0 and batch_id > 0
            ) or batch_id == no_batches["train"] - 1:
                '''
                print(
                    "\nBatch: {}, Loss: {}, AvgLoss: {}".format(
                        batch_id,
                        np.around(loss.data.cpu().numpy().tolist(), 4),
                        np.around(np.average(loss_arr).tolist(), 4),
                    )
                )
                '''
                # validation
                ccc.eval()
                ft_linear.eval()
                pred_data_valid = []
                true_data_valid = []
                with torch.no_grad():
                    loss_dev_arr = []
                    for batch_dev_id in range(no_batches["valid"]):
                        subgraphs, all_nodes, labels = ft_batch_input["valid"][
                            batch_dev_id
                        ]

                        # random mask
                        masked_pos = torch.randn(args.batch_size, 2)
                        masked_nodes = Variable(
                            torch.LongTensor([[] for ii in range(args.ft_batch_size)])
                        )
                        _, layer_output, _ = ccc(
                            subgraphs, all_nodes, masked_pos, masked_nodes
                        )
                        pred_scores, _, _ = ft_linear(layer_output)
                        loss = criterion(
                            pred_scores.squeeze(1), labels.squeeze(1).to(args.device)
                        )
                        loss_dev_arr.append(loss.data.cpu().numpy().tolist())

                        score = pred_scores.data.cpu().numpy().tolist()
                        labels = labels.data.cpu().numpy().tolist()
                        for ii, _ in enumerate(score):
                            pred_data_valid.append(score[ii][0])
                            true_data_valid.append(labels[ii][0])

                    loss_dev_avg = np.average(loss_dev_arr)
                    print(
                        "MinLoss: {}, CurLoss: {}".format(
                            np.around(loss_dev_min, 4), np.around(loss_dev_avg, 4)
                        )
                    )

                    if loss_dev_avg < loss_dev_min:
                        loss_dev_min = loss_dev_avg
                        fmodel = open(
                            os.path.join(
                                ft_out_dir, "finetune_" + str(epoch) + ".model"
                            ),
                            "wb",
                        )
                        torch.save(ft_linear.state_dict(), fmodel)
                        fmodel.close()

                ccc.train()
                ft_linear.train()
        loss_dev_final.append(loss_dev_min)
        print("MinLoss: ", np.around(loss_dev_min, 4))
        end_time = time.time()
        print("epoch time: (s)", (end_time - start_time))
    best_epoch = np.argsort(loss_dev_final)[0]
    print("\nBest Epoch: {}".format(best_epoch))

    np.save(f"{args.outdir}/finetuning_loss.npy", loss_dev_final)
    # testing
    print("Begin Testing")
    ccc.eval()
    ft_linear.eval()

    fl_ = os.path.join(ft_out_dir, "finetune_{}.model".format(best_epoch))
    ft_linear.load_state_dict(
        torch.load(fl_, map_location=lambda storage, loc: storage)
    )

    pred_data = []
    true_data = []
    path_metrics_pretrained = PathMetrics(
        args.device,
        pretrained_node_embedding_tensor,
        attr_graph.G,
        ent2id,
        args.d_model,
        args.ft_d_ff,
        args.ft_layer,
        args.ft_drop_rate,
        attr_graph,
        args.ft_input_option,
        args.n_layers,
    )
    path_metrics_graphbert_ft = PathMetrics(
        args.device,
        None,
        attr_graph.G,
        ent2id,
        args.d_model,
        args.ft_d_ff,
        args.ft_layer,
        args.ft_drop_rate,
        attr_graph,
        args.ft_input_option,
        args.n_layers,
    )
    with torch.no_grad():
        att_weights = {}
        final_emd = {}
        for batch_test_id in range(no_batches["test"]):
            if batch_test_id % 100 == 0:
                print("Evaluating test batch {}".format(batch_test_id))
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

            att_weights[batch_test_id] = {
                "attentions": att_output,
                "subgraphs": subgraphs,
                "nodes_sequence": all_nodes,
                "labels": labels,
            }
            final_emd[batch_test_id] = {
                "src_emd": src_embedding,
                "dst_emd": dst_embedding,
                "subgraphs": subgraphs,
                "nodes_sequence": all_nodes,
                "labels": labels,
            }
            path_metrics_pretrained.update_batch(subgraphs, labels)
            path_metrics_graphbert_ft.update_batch_graphbert(
                subgraphs, score, labels, layer_output, all_nodes, "graphbert_ft"
            )

            for ii, _ in enumerate(score):
                pred_data.append(score[ii][0])
                true_data.append(labels[ii][0])
            # print(len(score), len(labels))

        # dump path metrics
        path_dict = path_metrics_pretrained.finalize()
        json.dump(
            path_dict, open(outdir + "path_metrics_pretrained.json", "w"), indent=4
        )
        path_dict = path_metrics_graphbert_ft.finalize()
        json.dump(
            path_dict, open(outdir + "path_metrics_graphbert_ft.json", "w"), indent=4
        )
        # dump attention weights
        att_weights_file = outdir + "attention_weights.pickled"
        pickle.dump(att_weights, open(att_weights_file, "wb"))
        # dump embeddings for visulization#nodes sequences
        # flag = len(nodes_sequence[ii])
        # for jj in range(len(nodes_sequence[ii])):
        #     if nodes_sequence[ii][jj] == 10099:
        #         flag = jj
        #         break
        # nodes_sequence[ii] = nodes_sequence[ii][:flag]
        final_emd_file = outdir + "final_emd_test.pickled"
        pickle.dump(final_emd, open(final_emd_file, "wb"))

    return pred_data, true_data, pred_data_valid, true_data_valid
