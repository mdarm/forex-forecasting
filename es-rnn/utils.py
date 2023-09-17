import numpy as np 
import pandas as pd
import random
from torch.utils.data import Dataset


class SequenceLabelingDataset(Dataset):
    
    def __init__(self, input_data, max_size=100, sequence_labeling=True, seasonality=12, out_preds=12, augment=True):
        self.data = input_data
        self.max_size = max_size
        self.sequence_labeling = sequence_labeling
        self.seasonality = seasonality
        self.out_preds = out_preds
        self.augment = augment
    
    def __len__(self):
        if self.augment:
            return int(10000)
        else:
            return len(self.data) - self.out_preds + 1
    
    def __getitem__(self, index):
        data_i = self.data
        
        if self.augment:
            # Randomly shift the inputs to create more data
            if len(data_i) > self.max_size:
                max_rand_int = len(data_i) - self.max_size
                # Take a random start integer
                start_int = random.randint(0, max_rand_int)
                data_i = data_i[start_int:(start_int + self.max_size)]
            else:
                start_int = 0
        else:
            # Ensure the sequence doesn't exceed max_size without data augmentation
            if len(data_i) > self.max_size:
                data_i = data_i[-self.max_size:]
            start_int = 0  # No random shifting when augment is False
        
        inp = np.array(data_i[:-self.out_preds])
        
        if self.sequence_labeling:
            out = np.array(data_i[self.out_preds:])
        else:
            out = np.array(data_i[-self.out_preds:])
        
        # Calculate how much to shift the season
        shift_steps = start_int % self.seasonality
        
        return inp, out, shift_steps