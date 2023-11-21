import json
import os
import pickle
import shutil
import time

import numpy as np
import torch
import torch.nn
import torch.optim as optim

from .ccc_model.ccc_model import CCC
from .ccc_model.process_walks_gcn import (
    Processing_GCN_Walks,
    get_normalized_masked_ids,
)
from .utils.data_utils import (
    create_batches,
    split_data,
)
from .utils.utils import EarlyStopping, load_pickle, show_progress


def setup_pretraining_input(args, attr_graph, context_gen, data_path):
    # split train/validation/test dataset
    tasks = ["train", "validate", "test"]
    split_file = {
        task: os.path.join(data_path, args.data_name + "_pretrain_" + task + ".txt")
        for task in tasks
    }
    if os.path.exists(split_file["train"]):
        print("\n load existing /validate/test data ...")
        for task in tasks:
            fin = open(split_file[task], "r")
            walk_data = []
            for line in fin:
                line = json.loads(line)
                walk_data.append(line)
            fin.close()
            if task == "train":
                walk_train = walk_data
            elif task == "validate":
                walk_validate = walk_data
            else:
                walk_test = walk_data
    else:
        # generate walks
        print("\n Generating subgraphs for pre-training ...")
        all_walks = context_gen.get_pretrain_subgraphs(
            data_path,
            args.data_name,
            args.beam_width,
            args.max_length,
            args.walk_type,
        )
        print("\n split data to train/validate/test and save the files ...")
        walk_train, walk_validate, walk_test = split_data(
            all_walks, data_path, args.data_name
        )
    print(len(walk_train), len(walk_validate), len(walk_test))

    # create batches
    num_batches = {}
    for task in tasks:
        if task == "train":
            walk_data = walk_train
        elif task == "validate":
            walk_data = walk_validate
        else:
            walk_data = walk_test
        cnt = create_batches(
            walk_data, data_path, args.data_name, task, args.batch_size
        )
        num_batches[task] = cnt
    print("number of batches for pre-training: ", num_batches)
    return num_batches


def run_pretraining(
    args, attr_graph, no_batches, pretrained_node_embedding, ent2id, rel2id
):
    earlystopping = EarlyStopping(patience=3, delta=0.001)
    # data_path = args.data_path +'compgcn_output/' + args.data_name + '/'
    data_path = args.data_path +"/" + args.data_name + "/outdir/"
    out_dir = args.outdir + "/"
    try:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    except:
        os.mkdir(out_dir)

    relations = attr_graph.relation_to_id
    #no_relations = attr_graph.get_number_of_relations()
    nodeid2rowid = attr_graph.get_nodeid2rowid()
    print(attr_graph.get_number_of_nodes())
    #no_nodes = attr_graph.get_number_of_nodes()
    walk_processor = Processing_GCN_Walks(
        nodeid2rowid, relations, args.n_pred, args.max_length, args.max_pred
    )

    print("\n processing walks in minibaches before running model:")
    batch_input_file = os.path.join(data_path, args.data_name + "_batch_input.pickled")
    if os.path.exists(batch_input_file):
        print("loading saved files ...")
        batch_input = load_pickle(batch_input_file)
    else:
        batch_input = {}
        tasks = ["train", "validate", "test"]
        for task in tasks:
            print(task)
            batch_input[task] = {}
            for batch_id in range(no_batches[task]):
                (
                    subgraphs_list,
                    all_nodes,
                    masked_nodes,
                    masked_postion,
                ) = walk_processor.process_minibatch(
                    data_path, args.data_name, task, batch_id
                )

                batch_input[task][batch_id] = [
                    subgraphs_list,
                    all_nodes,
                    masked_nodes,
                    masked_postion,
                ]
        pickle.dump(batch_input, open(batch_input_file, "wb"))

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

    # if torch.cuda.is_available():
    #    ccc =  torch.nn.DataParallel(ccc, device_ids=[0,1])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(ccc.parameters(), args.lr)
    node_embeddings = dict()

    loss_dev_final = []
    print("\n Begin Training")
    for epoch in range(args.n_epochs):
        loss_arr = []
        loss_dev_min = 1e6
        print("\nEpoch: {}".format(epoch))
        start_time = time.time()
        for batch_id in range(no_batches["train"]):
            (
                subgraphs_list,
                all_nodes,
                masked_nodes,
                masked_postion,
            ) = walk_processor.process_minibatch(
                data_path, args.data_name, "train", batch_id
            )

            if args.get_bert_encoder_embeddings:
                logits_lm, _ = ccc(
                    subgraphs_list, all_nodes, masked_postion, masked_nodes
                )
                # node_embeddings[subgraphs_list] = bert_subgraph_embedding_output
            else:
                logits_lm = ccc(
                    subgraphs_list, all_nodes, masked_postion, masked_nodes
                )
            # print('check ====', logits_lm.size(), masked_nodes_id.size())
            try:
                normalized_masked_nodes_id = get_normalized_masked_ids(
                    masked_nodes, ent2id
                )

            except:
                normalized_masked_nodes_id = masked_nodes
            # print(logits_lm.size(), normalized_masked_nodes_id.size())
            loss = criterion(
                logits_lm.transpose(1, 2), normalized_masked_nodes_id.to(args.device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.data.cpu().numpy().tolist())

            ccnt = batch_id % args.checkpoint
            if batch_id > 0 and batch_id % args.checkpoint == 0:
                ccnt = args.checkpoint
            '''
            message = "Loss: {}, AvgLoss: {}{}".format(
                np.around(loss.data.cpu().numpy().tolist(), 4),
                np.around(np.average(loss_arr).tolist(), 4),
                " " * 10,
            )
            show_progress(ccnt, min(args.checkpoint, no_batches["train"]), message)
            '''
            if (
                batch_id % args.checkpoint == 0 and batch_id > 0
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

                with torch.no_grad():
                    loss_dev_arr = []
                    for batch_dev_id in range(no_batches["validate"]):
                        (
                            subgraphs_list,
                            all_nodes,
                            masked_nodes,
                            masked_postion,
                        ) = batch_input["validate"][batch_dev_id]

                        if args.get_bert_encoder_embeddings:
                            logits_lm, _ = ccc(
                                subgraphs_list, all_nodes, masked_postion, masked_nodes
                            )
                            # node_embeddings[subgraphs_list] = bert_subgraph_embedding_output
                        else:
                            logits_lm = ccc(
                                subgraphs_list, all_nodes, masked_postion, masked_nodes
                            )

                        try:
                            normalized_masked_nodes_id = get_normalized_masked_ids(
                                masked_nodes, ent2id
                            )
                        except:
                            normalized_masked_nodes_id = masked_nodes
                        # print(normalized_masked_nodes_id.size())
                        loss = criterion(
                            logits_lm.transpose(1, 2), normalized_masked_nodes_id.to(args.device)
                        )

                        loss_dev_arr.append(loss.data.cpu().numpy().tolist())
                    loss_dev_avg = np.average(loss_dev_arr)

                print(
                    "MinLoss: {}, CurLoss: {}".format(
                        np.around(loss_dev_min, 4), np.around(loss_dev_avg, 4)
                    )
                )
                if loss_dev_avg < loss_dev_min:
                    loss_dev_min = loss_dev_avg
                    fmodel = open(
                        os.path.join(out_dir, "bert_" + str(epoch) + ".model"), "wb"
                    )
                    torch.save(ccc.state_dict(), fmodel)
                    fmodel.close()

                ccc.train()
                # if(earlystopping.check_early_stopping(loss_dev_avg) == True):
                #    print("Loss not decreasing over 3 epochs exiting")
                #    epoch = args.n_epochs + 1

        loss_dev_final.append(loss_dev_min)
        print("MinLoss: ", np.around(loss_dev_min, 4))
        end_time = time.time()
        print("epoch time: (s)", (end_time - start_time))

    best_epoch = np.argsort(loss_dev_final)[0]
    print("\nBest Epoch: {}".format(best_epoch))
    fbest = open(os.path.join(out_dir, "best_epoch_id.txt"), "w")
    fbest.write(str(best_epoch) + "\n")
    fbest.close()
    np.save(f"{args.outdir}/pretraining_loss.npy", loss_dev_final)

    print("Begin Testing")
    ccc.eval()
    fl_ = os.path.join(out_dir, "bert_{}.model".format(best_epoch))
    ccc.load_state_dict(
        torch.load(fl_, map_location=lambda storage, loc: storage)
    )

    pred_data = []
    true_data = []
    with torch.no_grad():
        for _ in range(no_batches["test"]):
            subgraphs_list, all_nodes, masked_nodes, masked_postion = batch_input[
                "test"
            ][batch_dev_id]

            if args.get_bert_encoder_embeddings:
                logits, bert_subgraph_embedding_output = ccc(
                    subgraphs_list, all_nodes, masked_postion, masked_nodes
                )
                for i, _ in enumerate(subgraphs_list):
                    str_subgraph = get_str_subgraph(subgraphs_list[i])
                    node_embeddings[str_subgraph] = bert_subgraph_embedding_output[i]
            else:
                logits = ccc(
                    subgraphs_list, all_nodes, masked_postion, masked_nodes
                )
            prob = torch.softmax(logits, dim=2)
            score, index = prob.topk(1, dim=2)

            pred_tmp = index.squeeze(2).data.cpu().numpy().tolist()
            try:
                normalized_masked_nodes_id = get_normalized_masked_ids(
                    masked_nodes, ent2id
                )
            except:
                normalized_masked_nodes_id = masked_nodes
            true_tmp = normalized_masked_nodes_id.data.cpu().numpy().tolist()
            pred_data += pred_tmp
            true_data += true_tmp

    # accu, mse = run_evaluation(pred_data, true_data)
    return pred_data, true_data

    # #Dump node vectors
    # if args.get_node_embedding == False:
    #     print('\n Begin dumping node embeddings ...')
    #     node_embedding = ccc.GetNodeVec()
    #     node_embedding_file = os.path.join(data_path, args.data_name+'_node_embedding.pickled')
    #     pickle.dump(node_embedding, open(node_embedding_file, "wb" ))

    # if(args.get_bert_encoder_embeddings):
    #     torch.save(node_embeddings, "node_embeddings_encoder.txt")


def get_str_subgraph(subgraph):
    str_subgraph = ""
    for edge in subgraph:
        str_subgraph += str(edge[0]) + "_" + str(edge[1]) + "_" + str(edge[2]) + ";"
    return str_subgraph
