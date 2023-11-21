import random
from collections import defaultdict

import networkx as nx


class NegativeSampleGenerator:
    def __init__(self):
        self.node_role_cluster_map = defaultdict(set)
        self.role_clusters = []
        self.node_type_map = dict()
        self.type_node_map = defaultdict(list)
        self.num_node_types = 0
        self.all_relations = set()
        self.schema_graph = nx.MultiGraph()
        return

    def __update_role_clusters(self, role_cluster):

        overlapping_cluster_indices = []
        for i in range(len(self.role_clusters)):
            overlap = self.role_clusters[i].intersection(role_cluster)
            if len(overlap) > 0:
                overlapping_cluster_indices.append(i)

        if len(overlapping_cluster_indices) == 0:
            self.role_clusters.append(role_cluster)
            return
        else:
            # Merge all overlapping clusters
            super_cluster = set()
            for idx in overlapping_cluster_indices:
                super_cluster.update(self.role_clusters[idx])
            new_role_clusters = [super_cluster]
            for i in range(len(self.role_clusters)):
                if i not in overlapping_cluster_indices:
                    new_role_clusters.append(self.role_clusters[i])
            self.role_clusters = new_role_clusters

        return

    def build_graph_schema(self, graph):

        # graph = self.load_graph(path)

        print("Building node cluster map ...")
        print(
            "Collect roles (relation.src/relation.dst) for each node [O(num_edges)]..."
        )
        for e in graph.edges(data="label"):
            self.node_role_cluster_map[e[0]].add("%s.src" % e[2])
            self.node_role_cluster_map[e[1]].add("%s.dst" % e[2])

        print("Merging role clusters [O(num_nodes*num_node_types)]...")
        for role_cluster in self.node_role_cluster_map.values():
            self.__update_role_clusters(role_cluster)

        print("Assigning node types [O(num_nodes)]...")
        self.num_node_types = len(self.role_clusters)
        for node_id, role_cluster in self.node_role_cluster_map.items():
            overlapping_clusters = []
            for i in range(self.num_node_types):
                if len(self.role_clusters[i].intersection(role_cluster)) > 0:
                    overlapping_clusters.append(i)
            # assert(len(overlapping_clusters) == 1)
            type_id = str(overlapping_clusters[0])
            self.node_type_map[node_id] = type_id
            self.type_node_map[type_id].append(node_id)

        print("Building schema graph [O(E)] ...")
        print("Collecting unique schema edges by storing in set ...")
        schema_edges = set()
        for e in graph.edges(data="label"):
            src_node_type = self.node_type_map[e[0]]
            dst_node_type = self.node_type_map[e[1]]
            relation = e[2]
            schema_edge_key = "%s:%s:%s" % (src_node_type, dst_node_type, relation)
            schema_edges.add(schema_edge_key)

        print("Adding edges in MultiGraph schema_graph")
        for e in schema_edges:
            print("Schema edge : ", e)
            tokens = e.split(":")
            self.schema_graph.add_edge(tokens[0], tokens[1], label=tokens[2])
            self.all_relations.add(tokens[2])

        # return graph

    def get_type_def(self, node_t):
        return self.role_clusters[int(node_t)]

    def get_invalid_neighbor_types(self, node_type_id):
        neighbors = list(self.schema_graph.neighbors(node_type_id))
        all_node_types = set([str(i) for i in range(self.num_node_types)])
        return all_node_types - set([node_type_id] + neighbors)

    def get_invalid_relation_types(self, node_t):
        node_relations = []
        for nbr_t in self.schema_graph.neighbors(node_t):
            multi_edge_data = self.schema_graph.get_edge_data(node_t, nbr_t)
            for i, _ in enumerate(multi_edge_data):
                r = multi_edge_data[i]["label"]
                node_relations.append(r)
        negative_relations = self.all_relations - set(node_relations)
        return negative_relations

    def gen_negative_triples(self, graph, count):

        self.build_graph_schema(graph)
        print("Generating negative triples ...")
        negative_relations = set()
        num_rels = 0
        skip_dup = 0
        print("Generating %d", count)
        for node_id in graph.nodes():

            # print('Iterating over node: %s' % node_id)
            node_t = self.node_type_map[node_id]
            # print('Node type:', self.get_type_def(node_t))
            invalid_node_types = self.get_invalid_neighbor_types(node_t)
            invalid_relation_types = self.get_invalid_relation_types(node_t)

            for invalid_node_t in invalid_node_types:

                # print('Invalid nbr type:', self.get_type_def(invalid_node_t))
                invalid_nbr = random.choice(self.type_node_map[invalid_node_t])
                # print('Invalid nbr: ', invalid_nbr)

                for invalid_relation_t in invalid_relation_types:

                    # print('Invalid relation: %s' % invalid_relation_t)
                    triple = (invalid_relation_t, node_id, invalid_nbr, "0")
                    triple_key = ":".join(triple)

                    if triple_key not in negative_relations:
                        negative_relations.add(triple_key)
                        num_rels += 1
                        if num_rels == count:
                            return [
                                triple_key.split(":")
                                for triple_key in negative_relations
                            ]
                    else:
                        skip_dup += 1

        print(
            "Number of -ve relations: %d skipped duplicates: %d"
            % (len(negative_relations), skip_dup)
        )
        return [triple_key.split(":") for triple_key in negative_relations]

    def load_graph(self, path):
        graph = nx.DiGraph()
        with open(path) as f_in:
            for line in f_in:
                tokens = line.strip().split(" ")
                relation = tokens[0]
                src = tokens[1]
                dst = tokens[2]
                graph.add_edge(src, dst, label=relation)
        return graph
