import json
import random

def get_splits(data_paths_json_path: str):
    """ 
    Simulate splits used during training to get the same splits for the test set.
    """
    data_paths = []
    with open(data_paths_json_path) as file:
        data_paths = json.load(file)

    random.seed(42)
    random.shuffle(data_paths)

    total_items = len(data_paths)
    # we use an 80/10/10 split for all runs - see base_datamodule.py
    train_size = int(total_items * 0.8)
    val_size = int(total_items * 0.1)

    train_paths = data_paths[:train_size]
    val_paths = data_paths[train_size : train_size + val_size]
    test_paths = data_paths[train_size + val_size :]
    
    return train_paths, val_paths, test_paths

if __name__ == "__main__":
    data_path = "data_paths.json"
    train_paths, val_paths, test_paths = get_splits(data_path)
    with open("train_paths.json", "w") as file:
        json.dump(train_paths, file, indent=4)
    with open("val_paths.json", "w") as file:
        json.dump(val_paths, file, indent=4)
    with open("test_paths.json", "w") as file:
        json.dump(test_paths, file, indent=4)