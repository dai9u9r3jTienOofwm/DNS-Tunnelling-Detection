import pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from .dataset import DNSTunnelingDataset
from .model import DNSTunnelingCNN
from config import *
def evaluate(data_path, checkpoint=os.path.join(WEIGHTS_DIR,'dns_tunneling_cnn.pt')):
 df=pd.read_csv(data_path); domains=df['domain']; labels=df['label'].astype(int); m=DNSTunnelingCNN(40,EMBED_DIM,NUM_FILTERS,KERNEL_SIZE); m.load_state_dict(torch.load(checkpoint,map_location='cpu')); m.eval();
 with torch.no_grad(): p=torch.sigmoid(m(torch.stack([x for x in DataLoader(DNSTunnelingDataset(domains,None,MAX_LEN),BATCH_SIZE)])))
 pred=(p.numpy()>=.5).astype(int); print({'f1':f1_score(labels,pred),'roc_auc':roc_auc_score(labels,p.numpy()),'confusion_matrix':confusion_matrix(labels,pred).tolist()})
