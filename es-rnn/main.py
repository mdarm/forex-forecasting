"""
Created in Sep 2023; initial github version was around May 2023.
@author: Darmanis Michael 
internal version no: 8, for the new test set (without nulls) and the option of additive-forecast-reconstruction.

The program is meant to save the forecast training-history and evaluations of a simplified version of Slawek's ES-RNN. The daily updated dataset, basic preprocessing and frequency resampling are all handled within.
"""

import os
import sys
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from tqdm import tqdm

from fetch_data import download_zip, unzip_and_rename
from process_data import clean_data, resample_data
from utils import SequenceLabeling, EarlyStopper, plot_losses 
from models import ESRNN 
from evaluation_metrics import mse, smape 


def create_raw_dataset():
    zip_url       = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip?1f54ac4889a7e6d01b17d729b1c02549"
    zip_path      = "eurofxref-hist.zip"
    unzip_dir     = "../dataset"
    original_name = "eurofxref-hist.csv"
    new_name      = "raw_dataset.csv"
    
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)

    if download_zip(zip_url, zip_path):
        unzip_and_rename(zip_path, unzip_dir, original_name, new_name)
        print("Downloaded, unzipped, renamed, and deleted the original forex-zip file successfully.")
    else:
        print("Failed to download the file.")
        sys.exit(1)

def process_dataset():
    unzip_dir             = "../dataset"
    old_name              = "raw_dataset.csv"
    new_name              = "processed_dataset.csv"
    cleaned_csv_file_path = os.path.join(unzip_dir, new_name)

    clean_data(os.path.join(unzip_dir, old_name),
                            cleaned_csv_file_path)
        
def resample_dataset():
    processed_csv_path = "../dataset/processed_dataset.csv"
    ouput_path         = "../dataset"

    resample_data(processed_csv_path, ouput_path)


def training():
    # Sampling frequencies and corresponding prediction horizons    
    frequencies = {
        'daily':    14,
        'weekly':   13,
        'monthly':   18,
        'quarterly': 8,
        'yearly':    6 
    }

    for frequency, horizon in frequencies.items():
        df = pd.read_csv(f"../dataset/{frequency}.csv")
        results = {}
        for coin in df.columns[1:]:
            coin_list = df[coin].tolist()
            test = coin_list[-horizon:]  # testing
            train = coin_list[:-horizon] # training
            val = coin_list              # validation
            sl   = SequenceLabeling(train, len(train), False,
                                    seasonality=horizon, out_preds=horizon)
            sl_v = SequenceLabeling(val, len(val), False,
                                    seasonality=horizon, out_preds=horizon)
            
            train_dl = DataLoader(dataset=sl, batch_size=512, shuffle=False)
            val_dl   = DataLoader(dataset=sl_v, batch_size=512, shuffle=False)
            
            hw = ESRNN(hidden_size=8, slen=horizon, pred_len=horizon, mode='multiplicative')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            hw = hw.to(device)
            
            opti = torch.optim.Adam(hw.parameters(), lr=0.01)
            early_stopper = EarlyStopper(patience=3, min_delta=0.01)

            overall_loss_val = []
            overall_loss_train = []
            print(f"\nTraining model for: {coin}\n{'=' * 30}") 
            for _ in tqdm(range(20)):
                validation_loss = []
                training_loss   = []
                for batch in iter(train_dl):
                    opti.zero_grad()
                    inp    = batch[0].float().to(device)
                    out    = batch[1].float().to(device)
                    shifts = batch[2].numpy()
                    pred   = hw(inp, shifts)
                    loss   = F.l1_loss(pred, out) 
                    training_loss.append(loss.detach().cpu().numpy())
                    loss.backward()
                    opti.step()

                for batch in iter(val_dl):
                    inp    = batch[0].float().to(device)
                    out    = batch[1].float().to(device)
                    shifts = batch[2].numpy()
                    pred   = hw(inp, shifts)
                    loss   = F.l1_loss(pred, out) 
                    validation_loss.append(loss.detach().cpu().numpy())
                    pred = hw(inp, shifts).detach()
                    inp.detach()
                    out.detach()

                print("Mean training loss:", np.mean(training_loss))
                print("Mean validation loss:", np.mean(validation_loss))
                overall_loss_val.append(np.mean(validation_loss))
                overall_loss_train.append(np.mean(training_loss))

                # This should, ideally, change according to sample-frequency
                if early_stopper.early_stop(np.mean(validation_loss)):             
                    break

            plot_losses(overall_loss_train, overall_loss_val,
                        coin, f"./outputs/{frequency}/")

            batch  = next(iter(val_dl))
            inp    = batch[0].float().to(device)
            out    = batch[1].float().to(device)
            shifts = batch[2].numpy()
            pred   = hw(torch.cat([inp, out], dim=1), shifts)
            
            predictions = pred[0:][0].cpu().detach().numpy()
            test        = np.asarray(test) 

            mse_val       = mse(test, predictions)
            smape_val     = smape(test, predictions)
            results[coin] = {'mse': mse_val, 'smape': smape_val}
            print("\n")

        # Save results to text file
        with open(f"./outputs/{frequency}/results.txt", "w") as f:
            for coin, metrics in results.items():
                f.write(f"{coin} - MSE: {metrics['mse']}, SMAPE: {metrics['smape']}\n")
            avg_mse   = np.mean([metrics['mse'] for metrics in results.values()])
            avg_smape = np.mean([metrics['smape'] for metrics in results.values()])
            f.write(f"\nAverage MSE for {frequency} data and a {horizon} prediction horizon: {avg_mse}\n")
            f.write(f"Average SMAPE for {frequency} data and a {horizon} prediction horizon: {avg_smape}\n")

        print("Overall MSE for {} forex frequency and a {} prediction horizon: {:.2f}".format(frequency, horizon, avg_mse))
        print("Overall sMAPE for {} forex frequency and a {} prediciton horizon: {:.2f}".format(frequency, horizon, avg_smape))


if __name__ == "__main__":
    create_raw_dataset()
    process_dataset()
    resample_dataset()
    training()
