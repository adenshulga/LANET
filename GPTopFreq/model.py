import numpy as np
from dataloading import DataLoader

class GP:

    def __init__(self, 
                 vocab_size: int) -> None:
        
        self.common_freq = np.zeros(vocab_size, dtype=float)

    def fit(self, dataloader: DataLoader) -> 'GP':

        num_baskets = 0

        for user_history in dataloader:
            
            self.common_freq += np.sum(user_history, axis=0)
            num_baskets += len(user_history)
        
        self.common_freq *= 1/num_baskets
    
    def __call__(self, user_history: np.ndarray) -> np.ndarray:
        
        personal_freq = user_history.mean(axis=0)

        return np.maximum(personal_freq, self.common_freq)
    
    def predict(self, dataloader: DataLoader) -> np.ndarray:

        prediction = []

        for user_hist in dataloader:
            prediction.append(self(user_hist))

        return np.array(prediction)

    def predict_last(self, 
             dataloader: DataLoader) -> float:
        
        prediction = []
        
        for user_hist in dataloader:
            prediction.append(self(user_hist[:-1]))

        return np.array(prediction)