import numpy as np
import json

from typing import Dict

def to_one_hot(basket: list, vocab_size) -> np.ndarray:

    one_hot = np.zeros(vocab_size, dtype=int)
    one_hot[basket] = 1

    return one_hot

def load_json_dataset(path) -> Dict:
    with open(path, 'r') as f:
        dataset = json.load(f)

    return dataset

class DataLoader:

    def __init__(self, 
                 dataset: dict, 
                 vocab_size: int) -> None:
        
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.num_users = len(dataset)
        self.users = list(dataset.keys())
        self.iterated_users = 0

    def __iter__(self):

        return self
    
    def __next__(self):

        if self.iterated_users < self.num_users:
            user_id = self.users[self.iterated_users]
            user_history = self.dataset[user_id]
            encoded_history = np.array([to_one_hot(basket, self.vocab_size) for basket in user_history])
            self.iterated_users += 1
            return encoded_history
        else:
            self.iterated_users = 0 
            raise StopIteration
        
    def __getitem__(self, key):

        if isinstance(key, int):            
            user_id = self.users[key]
            user_history = self.dataset[user_id]
            return np.array([to_one_hot(basket, self.vocab_size) for basket in user_history])

        else:
            raise TypeError("Invalid argument type.")     

    def gen_gt(self) -> np.ndarray:

        gt = []

        for user_hist in self:
              gt.append(user_hist[-1])
        
        return np.array(gt)

class Dataset:

    def __init__(self, name, split) -> None:
        dataset_name = name
        path = f'../DNNTSP/data/{dataset_name}/split_{split}/{dataset_name}.json'

        self.dataset = load_json_dataset(path)
        self.vocab_size = self.get_vocab_size()

    def get_vocab_size(self) -> int:
        
        unique_items = set()

        # Iterate through each subset of the dataset (train, valid, test)
        for subset in self.dataset.values():
            # Iterate through each basket in the subset
            for user in subset.values():
                # Add the unique items of each basket to the set
                for item_list in user:
                    unique_items.update(item_list)

        # The size of the unique_items set is the vocabulary size
        return max(unique_items) + 1
    
    def gen_dataloaders(self) -> list[DataLoader]:

        modes = ['train', 'validate', 'test']

        return [DataLoader(self.dataset[mode], self.vocab_size) for mode in modes ]
    

    
