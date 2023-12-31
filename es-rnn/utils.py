import os
import random
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_losses(train_losses, val_losses, coin_name, directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(8, 5), dpi=300)
    
    plt.plot(train_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
    
    plt.title(f"Training and Validation Loss for {coin_name}", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    
    plt.savefig(os.path.join(directory, f"{coin_name}_loss_plot.png"), bbox_inches='tight')
    plt.close()

class SequenceLabeling(Dataset):
    def __init__(self, input, max_size=100, sequence_labeling=True, seasonality=12, out_preds=12):     
        self.data = input
        self.max_size = max_size
        self.sequence_labeling = sequence_labeling
        self.seasonality = seasonality
        self.out_preds = out_preds
        
    def __len__(self):
        return int(10000)
    
    def __getitem__(self, index):
        data_i = self.data
        
        # Randomly shift the inputs to create more data
        if len(data_i) > self.max_size:
            max_rand_int = len(data_i)-self.max_size
            start_int = random.randint(0,max_rand_int)
            data_i = data_i[start_int:(start_int+self.max_size)]
        else:
            start_int = 0

        inp = np.array(data_i[:-self.out_preds])
        
        if self.sequence_labeling==True:
            # In case of sequence labeling, shift the input by the range to output
            out = np.array(data_i[self.out_preds:])
        else:
            # In case of sequence classification, return only the last n elements
            # needed for the forecast
            out = np.array(data_i[-self.out_preds:])
            
        # How much to shift the season 
        shift_steps = start_int%self.seasonality
        return inp, out, shift_steps

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
