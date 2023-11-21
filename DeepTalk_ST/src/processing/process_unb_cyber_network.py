from pprint import pprint
import json
import re
import random
import operator
import collections
from pathlib import Path
random.seed(30)


def get_positive_edges(edge_file_id, path):
    fp = open(path, 'r')
    positive_edges = []
    for line in fp:
        edge = re.split(' ', line[:-1])
        edge = [0, int(edge[0]), int(edge[1])]
        positive_edges.append(edge)
    fp.close()
    print('No. of positive edges in', edge_file_id, len(positive_edges))

    return positive_edges


def get_ent2id(edge_file_id, path):
    # Get node ids
    fp = open(path, 'r')
    ent2id = {}
    for line in fp:
        line = re.split('\t', line[:-1])
        ent2id[line[0]] = int(line[1])
    fp.close()
    print('No. of nodes in', edge_file_id, len(ent2id))

    return ent2id


def split_positive_edges(positive_edges):
    # split positive edges
    no_pos = len(positive_edges)
    no_train = int(no_pos*0.8)
    no_test = int(no_pos*0.1)
    random.shuffle(positive_edges)
    train_edges = positive_edges[:no_train]
    valid_edges = positive_edges[no_train:-no_test]
    test_edges = positive_edges[-no_test:]
    print('No. of edges in training: {}, valid: {}, test: {}'
          .format(len(train_edges), len(valid_edges), len(test_edges)))

    return train_edges, valid_edges, test_edges


def generate_negative_edges(no_neg, ent2id, positive_edges):
    # generate negative edges

    negative_edges = []
    cnt = 0
    while 1:
        node_1, node_2 = random.sample(range(len(ent2id)), 2)
        edge = [0, node_1, node_2]
        reverse_edge = [0, node_2, node_1]
        if edge not in negative_edges and edge not in positive_edges and\
                reverse_edge not in negative_edges and reverse_edge not in positive_edges:
            edge = [0, node_1, node_2, 0]
            negative_edges.append(edge)
            cnt += 1
        if cnt == no_neg:
            break

    return negative_edges


def dump_data(ent2id, output_dir):
    # dump data
    ent2id = sorted(ent2id.items(), key=operator.itemgetter(1))
    ent2id = collections.OrderedDict(ent2id)
    out = json.dump(ent2id, open(
        '{}/ent2id.txt'.format(output_dir), 'w'))

    # dump edges
    tasks = {'train': train_edges, 'valid': valid_edges, 'test': test_edges}
    for task in tasks:
        print(task)
    #     print(tasks[task])
        fout = open('{}/{}.txt'.format(output_dir, task), 'w')
        for ee in tasks[task]:
            ee = [str(itm) for itm in ee]
            if task != 'train' and len(ee) == 3:
                ee.append('1')
            out = ' '.join(ee)
            fout.write(out)
            fout.write('\n')
        fout.close()


def dump_rel2id(path):
    rel2id = {'rel-0': 0}
    out = json.dump(rel2id, open(path, 'w'))


if __name__ == '__main__':
    cyber_data_dir = 'data/cyber'
    # Filter edges
    for edge_file_id in range(11, 18):
        print(edge_file_id)
        edge_file_id = str(edge_file_id)
        # Get Input
        fp = open(
            '{}/raw/testbed-{}jun-aggr.txt.edges'.format(cyber_data_dir, edge_file_id), 'r')
        cnt = 0
        all_edges = []
        for line in fp:
            edge = re.split('\t', line[:-1])
            reverse_edge = [edge[1], edge[0]]
            if edge not in all_edges and reverse_edge not in all_edges:
                all_edges.append(edge)
            cnt += 1

            if cnt % 50e3 == 0:
                print(cnt)

        fp.close()
        print(cnt, len(all_edges))

        # Create output directory
        day_output_dir = f'{cyber_data_dir}/processed/cyber_{edge_file_id}'
        Path(day_output_dir).mkdir(parents=True, exist_ok=True)

        # Dump Output
        fout = open(
            '{}/{}_filtered_edges.txt'.format(day_output_dir, edge_file_id), 'w')
        for ii in range(len(all_edges)):
            out = ' '.join(all_edges[ii])
            fout.write(out+'\n')
        fout.close()

    # Split positive edges and generate negative edges
    for edge_file_id in range(11, 18):
        print(edge_file_id)
        # Create output directory
        day_output_dir = f'{cyber_data_dir}/processed/cyber_{edge_file_id}'
        edge_file_id = str(edge_file_id)

        positive_edges_path = f"{day_output_dir}/{edge_file_id}_filtered_edges.txt"
        positive_edges = get_positive_edges(edge_file_id, positive_edges_path)

        ent2id_input_path = f'{cyber_data_dir}/raw/testbed-{edge_file_id}jun-aggr.txt.ids.tsv'
        ent2id = get_ent2id(edge_file_id, ent2id_input_path)

        train_edges, valid_edges, test_edges = split_positive_edges(
            positive_edges)

        no_neg = len(valid_edges)+len(test_edges)
        negative_edges = generate_negative_edges(
            no_neg, ent2id, positive_edges)
        random.shuffle(negative_edges)
        valid_edges += negative_edges[:-len(test_edges)]
        test_edges += negative_edges[-len(test_edges):]
        print('No. of edges in training: {}, valid: {}, test: {}'
              .format(len(train_edges), len(valid_edges), len(test_edges)))

        dump_data(ent2id, day_output_dir)
        rel2id_path = f"{day_output_dir}/rel2id.txt"
        dump_rel2id(rel2id_path)
