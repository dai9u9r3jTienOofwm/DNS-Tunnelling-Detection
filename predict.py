import argparse, torch
from src.dataset import encode_domain, VOCAB
from src.model import DNSTunnelingCNN
from config import *
def predict(domain, checkpoint=os.path.join(WEIGHTS_DIR,'dns_tunneling_cnn.pt')):
 m=DNSTunnelingCNN(len(VOCAB),EMBED_DIM,NUM_FILTERS,KERNEL_SIZE); m.load_state_dict(torch.load(checkpoint,map_location='cpu')); m.eval();
 with torch.no_grad(): score=torch.sigmoid(m(torch.tensor([encode_domain(domain,MAX_LEN)]))).item()
 return score, score>=.5
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('domain'); a=p.parse_args(); s,y=predict(a.domain); print({'domain':a.domain,'score':s,'is_tunnel':y})
