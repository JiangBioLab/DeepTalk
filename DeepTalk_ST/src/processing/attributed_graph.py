import pickle
import random

import networkx as nx
import numpy as np

from .negative_sampling import NegativeSampleGenerator


class AttributedGraph:
    def __init__(self):
        """
        Base class for all graph loaders.
        Assumes derived classes will create a networkx graph object of type
        1) nx.Graph() or nx.DiGraph()
        2) A node id to row id (0 to n-1) dictionary
        3) node_attr_dictionary per node type
        {"nodetype" : pd.dataframe}
        4) set of unique relations
        Note : The random walk methods implemented on top of networkx do not
        work with nx.MultiGraph() and hence MultiGraph object is not supported
        """
        self.G = nx.Graph()
        self.node_attr_dfs = dict()
        self.unique_relations = set()
        self.node_types = dict()
        self.normalized_node_id_map = dict()
        self.train_edges = list()
        self.valid_edges = list()
        self.test_edges = list()
        self.relation_to_id = dict()
        self.id_to_relation = dict()
        self.nodeid2rowid = dict()
        self.rowid2nodeid = dict()
        self.rowid2vocabid = dict()

    def load_graph(self, edgefile, node_attr_file=None):
        """
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.nodeid
        3) self.node_attrs
        4) self.unique_relations
        see SampleParser for example
        """
        return

    def get_normalized_node_id(self, node_id):
        # throw error?
        if str(node_id) not in self.G.nodes():
            print("Could not find node in graph", str(node_id))
            return None
        else:
            return self.normalized_node_id_map[str(node_id)]

    def create_normalized_node_id_map(self):
        cnt = 0
        for node_id in self.G.nodes():
            self.normalized_node_id_map[node_id] = cnt
            cnt += 1
        return

    def dump_graph(self, outdir, fr_valid_edges, fr_test_edges):
        print("Saving  graph at ", outdir)
        self.generate_link_prediction_dataset(outdir, fr_valid_edges, fr_test_edges)
        try:
            with open(outdir + "/normalized.txt", "w") as f:
                for u, v in self.G.edges():
                    relation = self.G.edges[u, v]["label"]
                    # f.write(str(self.normalized_node_id_map[u]) +
                    # " " + str(self.normalized_node_id_map[v]) + " " + relation + "\n")
                    f.write(f"{u}\t{v} {relation}\n")
        except IOError:
            print("Unable to write normalized graph at ", outdir)

        print("Saving test train validation set for link prediction")
        with open(outdir + "/train.txt", "w") as f:
            for edge in self.train_edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")

        with open(outdir + "/valid.txt", "w") as f:
            for edge in self.valid_edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")

        with open(outdir + "/test.txt", "w") as f:
            for edge in self.test_edges:
                f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")
        return

    def get_graph(self):
        return self.G

    def get_nodes(self):
        return self.G.nodes()

    def get_number_of_nodes(self):
        return len(self.G.nodes())

    def get_number_of_edges(self):
        return len(self.G.edges())

    def get_number_of_relations(self):
        return len(self.unique_relations)

    # def get_node_attr(self, node_id):
    #    return self.node_attrs[node_id]

    def get_edge_attr(self, u, v):
        return self.G.edges[u, v]["label"]

    def get_edges(self, node_id):  # FIXME, removed dangerous list init
        if len(node_id) == 0:
            return self.G.edges.data()
        else:
            return list(self.G.neighbors(node_id))

    def get_node_attr_df(self):
        return self.node_attr_dfs

    def get_node_types(self):
        return self.node_types

    def get_node_type(self, node_id):
        if node_id in self.node_types:
            return self.node_types[node_id]
        else:
            return ""

    def get_num_node_types(self):
        return len(self.node_attr_dfs)

    def get_node_attr(self, node_id, attribute_label=None):
        node_type = self.node_types[node_id]
        try:
            if attribute_label:
                return self.node_attr_dfs[node_type].at[node_id, attribute_label]
            else:
                return self.node_attr_dfs[node_type].loc[node_id]
        except KeyError:
            print("Could not find this node attribute", node_id, attribute_label)
            return None

    def get_random_non_edges(self, cnt):
        result = []
        nodes = list(self.G.nodes())
        relations = self.unique_relations
        print("Generating false edges", cnt)
        for _ in range(cnt):
            pair = random.sample(nodes, 2)
            u = pair[0]
            v = pair[1]
            if v not in list(self.G.neighbors(u)):
                result.append((random.sample(relations, 1)[0], u, v, "0"))

        return result

    def generate_link_prediction_dataset(self, outdir, fr_valid_edges, fr_test_edges):
        print("In Default AttributedGraph LinkPrediction generation")
        false_edge_gen = self.false_edge_gen
        all_positive_edges = self.map_positive_edges()
        num_edges = self.G.number_of_edges()
        print(num_edges)
        num_test_edges = int(num_edges * fr_test_edges)
        num_valid_edges = int(num_edges * fr_valid_edges)
        num_train_edges = num_edges - (num_valid_edges + num_test_edges)
        self.train_edges = all_positive_edges[0:num_train_edges]
        self.valid_edges = all_positive_edges[
            num_train_edges : num_train_edges + num_valid_edges
        ]
        self.test_edges = all_positive_edges[num_train_edges + num_valid_edges :]
        if false_edge_gen == "pattern":
            self.train_edges = self.generate_false_edges3(self.train_edges)
            self.valid_edges = self.generate_false_edges3(self.valid_edges)
            self.test_edges = self.generate_false_edges3(self.test_edges)
        elif false_edge_gen == "double":
            self.train_edges = self.generate_false_edges2(self.train_edges)
            self.valid_edges = self.generate_false_edges2(self.valid_edges)
            self.test_edges = self.generate_false_edges2(self.test_edges)
        else:
            self.train_edges = self.generate_false_edges(self.train_edges)
            self.valid_edges = self.generate_false_edges(self.valid_edges)
            self.test_edges = self.generate_false_edges(self.test_edges)

        self.valid_edges, self.test_edges = self.remove_coldstart_valid_test(
            self.train_edges, self.valid_edges, self.test_edges
        )
        # print('Dump link prediction edges ...')
        # self.dump_edges(outdir+ "/train.txt", self.train_edges)
        # self.dump_edges(outdir+ "/valid.txt", self.valid_edges)
        # self.dump_edges(outdir+ "/test.txt", self.test_edges)
        return self.train_edges, self.valid_edges, self.test_edges

    def remove_coldstart_valid_test(self, train_edges, valid_edges, test_edges):
        nodes_in_training_set = []
        for r, u, v, _ in train_edges:
            nodes_in_training_set.append(u)
            nodes_in_training_set.append(v)

        cleaned_valid_edges = []
        for r, u, v, label in valid_edges:
            if u in nodes_in_training_set and v in nodes_in_training_set:
                cleaned_valid_edges.append((r, u, v, label))

        cleaned_test_edges = []
        for r, u, v, label in test_edges:
            if u in nodes_in_training_set and v in nodes_in_training_set:
                cleaned_test_edges.append((r, u, v, label))
        return cleaned_valid_edges, cleaned_test_edges

    def generate_false_edges(self, positive_edge_list):
        """
        This method takes in as input a list of positive edges of form
        (relation, u, v, "1")
        and generates ann equal number of false edges (not present in graph),
        such that for each edge in positive_edge_list:
            (relation,u, v', "0"), here u->v' does not exist in self.G
        returns : random.shuffle(true_edges + false_edges)
        """
        count = len(positive_edge_list)
        # collect nodes of different types
        node_type_to_ids = dict()
        for node_id, node_type in self.node_types.items():
            if node_type in node_type_to_ids:
                node_type_to_ids[node_type].append(node_id)
            else:
                node_type_to_ids[node_type] = [node_id]

        false_edges = set()
        while len(false_edges) < count:
            for relation, source, target, _ in positive_edge_list:
                target_type = self.node_types[target]
                false_target = random.sample(node_type_to_ids[target_type], 1)[0]
                all_source_nbrs = list(set(list(nx.all_neighbors(self.G, source))))
                if false_target not in all_source_nbrs:
                    if len(false_edges) < count:
                        false_edges.add((relation, source, false_target, "0"))
                    else:
                        break

        final_edges = list(set(positive_edge_list + list(false_edges)))
        random.shuffle(final_edges)
        return final_edges
    

    def generate_false_edges1(self, positive_edge_list, false_edge_list):
        """
        This method takes in as input a list of positive edges of form
        (relation, u, v, "1")
        and generates ann equal number of false edges (not present in graph),
        such that for each edge in positive_edge_list:
            (relation,u, v', "0"), here u->v' does not exist in self.G
        returns : random.shuffle(true_edges + false_edges)
        """
        
        final_edges = list(set(positive_edge_list + false_edge_list))
        random.shuffle(final_edges)
        return final_edges


    def generate_false_edges2(self, positive_edge_list):
        """
        This method takes in as input a list of positive edges of form
        (relation, u, v, "1")
        and tries to generates double number of false edges (not present in graph),
        such that for each edge in positive_edge_list:
            (relation,u, v', "0"), here u -> v' does not exist in self.G
            (relation,u', v, "0"), here u'-> v does not exist in self.G
        returns : random.shuffle(true_edges + false_edges)
        """

        # collect nodes of different types
        node_type_to_ids = dict()
        for node_id, node_type in self.node_types.items():
            if node_type in node_type_to_ids:
                node_type_to_ids[node_type].append(node_id)
            else:
                node_type_to_ids[node_type] = [node_id]

        # generate false edges for every positive example
        false_edges = []
        for relation, source, target, _ in positive_edge_list:
            # generate false edges of type (source, relation, false_target) for every
            # (source, relation, target) in positive_edge_list
            target_type = self.node_types[target]
            false_target = random.sample(node_type_to_ids[target_type], 1)[0]
            all_source_nbrs = list(set(list(nx.all_neighbors(self.G, source))))
            if false_target not in all_source_nbrs:
                false_edges.append((relation, source, false_target, "0"))

            # generate false edges of type (false_source, relation, target) for every
            # (source, relation, target) in positive_edge_list
            source_type = self.node_types[source]
            false_source = random.sample(node_type_to_ids[source_type], 1)[0]
            all_target_nbrs = list(set(list(nx.all_neighbors(self.G, target))))
            if false_source not in all_target_nbrs:
                false_edges.append((relation, false_source, target, "0"))

        # generate false edges of type (false_source, relation, target) for every
        final_edges = list(set(positive_edge_list + false_edges))
        random.shuffle(final_edges)
        print(len(final_edges))
        return final_edges

    def generate_false_edges3(
        self, positive_edge_list, num_false_count_per_true_edge=5
    ):
        # generate negative edges of
        # num_false_count_per_true_edge*len(self.G.edges()) for graph object
        # 1) find valid edge types for each node
        # 2) group nodes that have same set of valid edges to determine node types
        # 3) for each node type pair
        #    (relation[i],u', v', "0"), here u'.type -> v'.type does not exist in self.G
        #    relation[i] = self.unique_relations[i]
        # returns : random.shuffle(true_edges + false_edges)
        # get {node -> set(valid edge types)}

        generator = NegativeSampleGenerator()
        false_edges = generator.gen_negative_triples(
            self.G, len(positive_edge_list) * num_false_count_per_true_edge
        )

        final_edges = list(positive_edge_list + false_edges)
        random.shuffle(final_edges)
        return final_edges

    def map_positive_edges(self):
        mappped_edge_list_by_type = []
        for u, v in self.G.edges():
            relation = self.G.edges[u, v]["label"]
            mappped_edge_list_by_type.append((relation, u, v, "1"))
        return mappped_edge_list_by_type

    def dump_edges(self, filename, edges, label=True):
        random.shuffle(edges)
        with open(filename, "w") as f:
            for edge in edges:
                if label:
                    f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")
                else:
                    f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
        return

    def dump_stats(self):
        print("DataSet Stats :")
        print("Number of Nodes", self.get_number_of_nodes())
        print("Number of Edges", self.get_number_of_edges())
        print("Number of Node types", self.get_num_node_types())
        print("Number of relations", self.get_number_of_relations())

    def get_nodeid2rowid(self):
        self.nodeid2rowid = {}
        for node_type in self.node_attr_dfs:
            df = self.node_attr_dfs[node_type]
            node_set = df.index.tolist()
            node2row = {}
            for i, _ in enumerate(node_set):
                node2row[node_set[i]] = i
            # print('\n nodeid2rowid: \n', node2row)
            self.nodeid2rowid[node_type] = node2row

        return self.nodeid2rowid

    def get_rowid2nodeid(self):
        self.rowid2nodeid = {}
        for node_type in self.nodeid2rowid:
            self.rowid2nodeid[node_type] = {}
            for itm in self.nodeid2rowid[node_type]:
                self.rowid2nodeid[node_type][self.nodeid2rowid[node_type][itm]] = itm

        return self.rowid2nodeid

    def get_rowid2vocabid_map(self):
        self.rowid2vocabid = {}
        for node_type in self.rowid2nodeid:
            self.rowid2vocabid[node_type] = {}
            for row_id in self.rowid2nodeid[node_type]:
                graph_node_id = self.rowid2nodeid[node_type][row_id]
                vocabid = self.get_normalized_node_id(graph_node_id)
                self.rowid2vocabid[node_type][row_id] = vocabid

        return self.rowid2vocabid

    def get_rowid2vocabid(self, row_id, node_type):
        # return an interger of vocab id
        return self.rowid2vocabid[node_type][int(row_id)]

    def get_rowid2graphid(self, row_id, node_type):
        try:
            return self.rowid2nodeid[node_type][int(row_id)]
        except KeyError:
            print("Node type and row id not found", row_id, node_type)
            return -1

    def get_graph_node_id(self, node_type_with_row_id):
        """take the string containing nodetype_noderowid."""
        arr = node_type_with_row_id.split("_")
        node_type = arr[0]
        row_id = arr[1]
        graph_node_id = self.rowid2nodeid[node_type][int(row_id)]
        return graph_node_id

    def set_relation_id(self):
        relations = list(self.unique_relations)
        self.relation_to_id = {relations[ii]: ii for ii in range(len(relations))}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}

    def get_relation_id(self, relation):
        try:
            return self.relation_to_id[relation]
        except KeyError:
            print("Could not find relation", relation)
            return -1

    def get_relation_from_id(self, relation_id):
        try:
            return self.id_to_relation[relation_id]
        except KeyError:
            print("Could not find relation", relation_id)
            return -1

    def dump_graph_transformer_networks_dataset(self, dump_dir):
        # Dump features pkl
        #   DBLP: len=334 int32 numpy ndarray per node
        #   ACM: len=1902 int64 numpy ndarray per node
        #   IMDB: len=1256 float32 numpy ndarray per node
        no_features = True
        node_features = []
        for _ in range(self.get_number_of_nodes()):
            # Currently we don't have features so I am just dumping an empty tensor
            if no_features:
                node_feature = np.zeros(shape=(100,))
            else:
                pass
            node_features.append(node_feature)
        node_features_path = f"{dump_dir}/node_features.pkl"
        with open(node_features_path, "wb") as f:
            pickle.dump(node_features, f)

        # Dump edges pkl
        #   All datasets from GTN have 4 relationship types

        # Dump labels pkl
        #   DBLP: 800, 400, 2857
        #   ACM: 600, 300, 2125
        #   IMDB: 300, 300, 2339
