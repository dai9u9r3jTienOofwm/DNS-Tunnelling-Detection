import os, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import DNSTunnelingDataset, VOCAB
from .model import DNSTunnelingCNN
from config import *
def load_data(path):
    df=pd.read_csv(path); domain_col=next(c for c in ['domain','query','dns','hostname'] if c in df.columns); label_col=next(c for c in ['label','is_tunnel','tunneling','target'] if c in df.columns); return df[domain_col].astype(str), df[label_col].astype(float)
def train(data_path):
    domains,labels=load_data(data_path); tr_d,va_d,tr_y,va_y=train_test_split(domains,labels,test_size=TEST_SIZE,random_state=SEED,stratify=labels); device='cuda' if torch.cuda.is_available() else 'cpu'; model=DNSTunnelingCNN(len(VOCAB),EMBED_DIM,NUM_FILTERS,KERNEL_SIZE).to(device); opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE); loss_fn=nn.BCEWithLogitsLoss(); best=1e9; wait=0
    for epoch in range(EPOCHS):
        model.train(); total=0
        for x,y in DataLoader(DNSTunnelingDataset(tr_d,tr_y,MAX_LEN),BATCH_SIZE,shuffle=True): opt.zero_grad(); loss=loss_fn(model(x.to(device)),y.to(device)); loss.backward(); opt.step(); total+=loss.item()
        model.eval(); val=sum(loss_fn(model(x.to(device)),y.to(device)).item() for x,y in DataLoader(DNSTunnelingDataset(va_d,va_y,MAX_LEN),BATCH_SIZE))/max(1,len(va_d)//BATCH_SIZE+1); print(f'Epoch {epoch+1}: train={total:.4f} val={val:.4f}')
        if val<best: best=val; wait=0; os.makedirs(WEIGHTS_DIR,exist_ok=True); torch.save(model.state_dict(),os.path.join(WEIGHTS_DIR,'dns_tunneling_cnn.pt'))
        else: wait+=1
        if wait>=PATIENCE: break
if __name__=='__main__': import argparse; p=argparse.ArgumentParser(); p.add_argument('--data',required=True); train(p.parse_args().data)
