import csv

def load_edges(path, split):
    edges = []
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            rel = int(row[0])
            node_1 = int(row[1])
            node_2 = int(row[2])
            if split != 'train':
                label = int(row[3])
                edge = (rel, node_1, node_2, label)
            else:
                edge = (rel, node_1, node_2)
            edges.append(edge)
    return edges

def write_edges(edges, path, split):
    print(f"Writing {len(edges)} edges to {path}")
    with open(path, 'w') as f:
        for edge in edges:
            rel = edge[0]
            node_1 = edge[1]
            node_2 = edge[2]
            if split != 'train':
                label = edge[3]
                edge = f"{rel} {node_1} {node_2} {label}"
            else:
                edge = f"{rel} {node_1} {node_2}"
            f.write(f"{edge}\n")


def transform_relationship_ids(edges, split):
    transformed_edges = []
    for edge in edges:
        rel = edge[0]
        node_1 = edge[1]
        node_2 = edge[2]

        # transform rel ID
        rel = rel + 1

        if split != 'train':
            label = edge[3]
            edge = (rel, node_1, node_2, label)
        else:
            edge = (rel, node_1, node_2)
        transformed_edges.append(edge)
    return transformed_edges
            

def main():
    data_path = 'data/dblp'
    #data_path = 'data/cyber/processed/cyber_17'
    train_path = f'{data_path}/train.txt'
    valid_path = f'{data_path}/valid.txt'
    test_path = f'{data_path}/test.txt'
    print(f"data_path: {data_path}")

    # Load edges
    train_edges = load_edges(train_path, 'train')
    valid_edges = load_edges(valid_path, 'valid')
    test_edges = load_edges(test_path, 'test')
    print(f"len(train_edges): {len(train_edges)}")
    print(f"len(valid_edges): {len(valid_edges)}")
    print(f"len(test_edges): {len(test_edges)}")

    # Sanity check
    all_relationship_ids = set()
    for edge in train_edges:
        rel = edge[0]
        all_relationship_ids.add(rel)
    
    for edge in valid_edges:
        rel = edge[0]
        all_relationship_ids.add(rel)
    
    for edge in test_edges:
        rel = edge[0]
        all_relationship_ids.add(rel)
    
    print()

    # Transform edges to 0 based relationship index

    train_edges = transform_relationship_ids(train_edges, 'train')
    valid_edges = transform_relationship_ids(valid_edges, 'valid')
    test_edges = transform_relationship_ids(test_edges, 'test')

    transformed_relationship_ids = set()
    for edge in train_edges:
        rel = edge[0]
        transformed_relationship_ids.add(rel)
    
    for edge in valid_edges:
        rel = edge[0]
        transformed_relationship_ids.add(rel)
    
    for edge in test_edges:
        rel = edge[0]
        transformed_relationship_ids.add(rel)


    print("After transformation")
    print(f"len(train_edges): {len(train_edges)}")
    print(f"len(valid_edges): {len(valid_edges)}")
    print(f"len(test_edges): {len(test_edges)}")
    print()

    write_edges(train_edges, train_path, 'train')
    write_edges(valid_edges, valid_path, 'valid')
    write_edges(test_edges, test_path, 'test')

if __name__ == '__main__':
    main()