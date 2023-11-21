import random


class GenerateGraphMask:
    """
    Given a list of subgraphs , where each subgraph is a list of edges, where
    each
    edge.tensor = cat(src_node.tensor, dest_node.tensor, relation.tensor)
    Provides fuctions to generate:
        link preiction task for
        1) subgraph edge prediction
        2) edge.source prediction
        3) edge.relation prediction
        4) edge.target prediction
    """

    def __init__(self, subgraph_sequences, n_pred):
        self.subgraph_sequences = subgraph_sequences
        self.n_pred = n_pred

    def relation_prediction(self):
        """
        generate relation mask
        """
        mask = []
        for subgraph in self.subgraph_sequences:
            num_edges = len(subgraph)
            # print(num_edges, self.n_pred)
            edge_mask_index = random.sample(range(num_edges), self.n_pred)
            subgraph_mask = []
            for i in range(len(subgraph)):
                if i in edge_mask_index:
                    subgraph_mask.append((1, 1, 0))
                else:
                    subgraph_mask.append((1, 1, 1))
            mask.append(subgraph_mask)
        return mask

    def source_prediction(self):
        """
        generate source mask
        """
        mask = []
        for subgraph in self.subgraph_sequences:
            num_edges = len(subgraph)
            edge_mask_index = random.sample(range(num_edges), self.n_pred)
            subgraph_mask = []
            for i in range(len(subgraph)):
                if i in edge_mask_index:
                    subgraph_mask.append((0, 1, 1))
                else:
                    subgraph_mask.append((1, 1, 1))
            mask.append(subgraph_mask)
        return mask

    def target_prediction(self):
        """
        generate target mask
        """
        mask = []
        for subgraph in self.subgraph_sequences:
            num_edges = len(subgraph)
            edge_mask_index = random.sample(range(num_edges), self.n_pred)
            subgraph_mask = []
            for i in range(len(subgraph)):
                if i in edge_mask_index:
                    subgraph_mask.append((1, 0, 1))
                else:
                    subgraph_mask.append((1, 1, 1))
            mask.append(subgraph_mask)
        return mask

    def MaskGeneration(self, mask_type):
        if mask_type == "relation":
            input_mask = self.relation_prediction()
        elif mask_type == "source":
            input_mask = self.source_prediction()
        elif mask_type == "target":
            input_mask = self.target_prediction()
        else:
            print("\n Wrong mask type. No mask is generated!")
            input_mask = []
            for subgraph in self.subgraph_sequences:
                subgraph_mask = [(1, 1, 1) for i in range(len(subgraph))]
                input_mask.append(subgraph_mask)

        return input_mask

    def GCN_MaskGeneration(self):
        mask = []
        masked_nodes = []
        masked_position = []
        for subgraph in self.subgraph_sequences:
            num_nodes = len(subgraph)
            mask_index = random.sample(range(num_nodes), self.n_pred)
            subgraph_mask = []
            subgraph_masked_nodes = []
            subgraph_masked_position = []
            for i, _ in enumerate(subgraph):
                if i in mask_index:
                    subgraph_mask.append(0)
                    subgraph_masked_nodes.append(subgraph[i])
                    subgraph_masked_position.append(i)
                else:
                    subgraph_mask.append(1)
            mask.append(subgraph_mask)
            masked_nodes.append(subgraph_masked_nodes)
            masked_position.append(subgraph_masked_position)

        return mask, masked_nodes, masked_position
