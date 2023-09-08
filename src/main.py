import os
import sys

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import random
from torch.utils.data import Dataset,sampler,DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

from fetch_data import download_zip, unzip_and_rename
from process_data import clean_data, resample_data

from utils import *
from models import es_rnn, holt_winters_no_trend


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


def playing_around():
    df = pd.read_csv("../dataset/quarterly.csv")
    usd_list = df['USD'].tolist()
    train = usd_list[:-12]
    test = usd_list

    sl=sequence_labeling_dataset(train,len(train),False)
    sl_t=sequence_labeling_dataset(test,len(test),False)

    train_dl = DataLoader(dataset=sl,
                          batch_size=512,
                          shuffle=False)

    test_dl = DataLoader(dataset=sl_t,
                         batch_size=512,
                         shuffle=False)

    hw = es_rnn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hw = hw.to(device)
    opti = torch.optim.Adam(hw.parameters(), lr=0.01)#,weight_decay=0.0001

    #Initial Prediction 
    overall_loss = []
    batch=next(iter(test_dl))
    inp=batch[0].float().to(device)#.unsqueeze(2)
    out=batch[1].float().to(device)#.unsqueeze(2).float()
    shifts=batch[2].numpy()
    pred=hw(inp, shifts)

    plt.plot(torch.cat([inp[0],out[0,:]]).cpu().numpy(),"g")
    plt.plot(torch.cat([inp[0],pred[0,:]]).cpu().detach().numpy(),"r")
    plt.savefig("../outputs/initial-prediction.png")
    plt.close()

    overall_loss_train=[]
    overall_loss=[]
    for j in tqdm(range(20)):
        loss_list_b=[]
        train_loss_list_b=[]
        #here we use batches of past, and to be forecasted value
        #batches are determined by a random start integer
        for batch in iter(train_dl):

            opti.zero_grad()
            inp=batch[0].float().to(device)#.unsqueeze(2)
            out=batch[1].float().to(device)#.unsqueeze(2).float()
            shifts=batch[2].numpy()
            #it returns the whole sequence atm 
            pred=hw(inp,shifts)
            loss=(torch.mean((pred-out)**2))**(1/2)
            train_loss_list_b.append(loss.detach().cpu().numpy())
            
            loss.backward()
            opti.step()


        #here we use all the available values to forecast the future ones and eval on it
        for batch in iter(test_dl):
            inp=batch[0].float().to(device)#.unsqueeze(2)
            out=batch[1].float().to(device)#.unsqueeze(2).float()
            shifts=batch[2].numpy()
            pred=hw(inp,shifts)
            #loss=torch.mean(torch.abs(pred-out))
            loss=(torch.mean((pred-out)**2))**(1/2)
            loss_list_b.append(loss.detach().cpu().numpy())
        
     
        print(np.mean(loss_list_b))
        print(np.mean(train_loss_list_b))
        overall_loss.append(np.mean(loss_list_b))
        overall_loss_train.append(np.mean(train_loss_list_b))

        # Plot of Train and Validation Loss, we nicely converge
        plt.plot(overall_loss, "g", label="Validation Loss")
        plt.plot(overall_loss_train, "r", label="Training Loss")
        plt.legend()
        plt.savefig("../outputs/training-loss.png")
        plt.close()

        #Forecasting on the Validation set
        batch=next(iter(test_dl))
        inp=batch[0].float().to(device)#.unsqueeze(2)
        out=batch[1].float().to(device)#.unsqueeze(2).float()
        shifts=batch[2].numpy()
        pred=hw(torch.cat([inp,out],dim=1),shifts)

        #plt.plot(torch.cat([inp,out,pred],dim=1)[0].detach().numpy(),"r")

        plt.plot(torch.cat([inp[0],out[0,:]]).cpu().detach().numpy(),"g")
        plt.plot(torch.cat([inp[0],pred[0,:]]).cpu().detach().numpy(),"r")
        plt.savefig("../outputs/validation-forecasting.png")
        plt.close()

        #Forecasting on a 12 period horizon
        batch=next(iter(test_dl))
        inp=batch[0].float().to(device)#.unsqueeze(2)
        out=batch[1].float().to(device)#.unsqueeze(2).float()
        shifts=batch[2].numpy()

        pred=hw(torch.cat([inp,out],dim=1),shifts)
        plt.plot(torch.cat([inp,out],dim=1)[0].cpu().detach().numpy(),"b")
        plt.plot(torch.cat([inp,out,pred],dim=1)[0].cpu().detach().numpy(),"g")
        plt.savefig("../outputs/forecasting.png")
        plt.close()     


if __name__ == "__main__":
    create_raw_dataset()
    process_dataset()
    resample_dataset()
    playing_around()
