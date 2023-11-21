import unittest

from ..negative_sampling import NegativeSampleGenerator


class TestNegativeSampleGenerator(unittest.TestCase):
    def test_load_graph(self):
        generator = NegativeSampleGenerator()
        test_path = "test.txt"
        graph = generator.load_graph(test_path)
        print("Loaded test graph: %s" % test_path)
        print("Nodes=%d, Edges=%d" % (graph.number_of_nodes(), graph.number_of_edges()))
        assert graph.number_of_nodes() == 11
        assert graph.number_of_edges() == 11
        return

    def test_build_graph_schema(self):

        generator = NegativeSampleGenerator()
        test_path = "test.txt"
        graph = generator.load_graph(test_path)
        generator.build_graph_schema(graph)

        print("\nPrinting role cluster associated with a sample node:")
        assert len(generator.node_role_cluster_map["richard_feynman"]) == 7

        print("\nPrinting entity role clusters ...")
        print(generator.role_clusters)
        assert len(generator.role_clusters) == 7

        print("\nType -> Node")
        print(generator.type_node_map)
        assert len(generator.type_node_map) == 7

        print("\nNode -> Type")
        print(generator.node_type_map)
        assert len(generator.node_type_map) == 11

        print("\nAll relations")
        print(generator.all_relations)
        assert len(generator.all_relations) == 7
        assert generator.schema_graph.number_of_nodes() == 7
        assert generator.schema_graph.number_of_edges() == 7

    def test_invalid_neighbor_types(self):

        generator = NegativeSampleGenerator()
        test_path = "test.txt"
        graph = generator.load_graph(test_path)
        generator.build_graph_schema(graph)

        query_node = "justin_bieber"
        node_t = generator.node_type_map[query_node]
        invalid_set = generator.get_invalid_neighbor_types(node_t)
        assert len(invalid_set) == 0

        query_node = "atheism"
        node_t = generator.node_type_map[query_node]
        invalid_set = generator.get_invalid_neighbor_types(node_t)
        assert len(invalid_set) == 5

        return

    def test_invalid_relation_types(self):

        generator = NegativeSampleGenerator()
        test_path = "test.txt"
        graph = generator.load_graph(test_path)
        generator.build_graph_schema(graph)

        node_t = generator.node_type_map["justin_bieber"]
        assert len(generator.get_invalid_relation_types(node_t)) == 0

        node_t = generator.node_type_map["atheism"]
        assert len(generator.get_invalid_relation_types(node_t)) == 6

        return

    def test_gen_negative_triples(self):

        generator = NegativeSampleGenerator()
        test_path = "test.txt"
        graph = generator.load_graph(test_path)
        neg_triples = generator.gen_negative_triples(graph, 20)
        for n_t in neg_triples:
            print(n_t)
        assert len(neg_triples) == 20


if __name__ == "__main__":
    unittest.main()
