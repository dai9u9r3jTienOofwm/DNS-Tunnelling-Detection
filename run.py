import argparse
from src.train import train
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); train(p.parse_args().data)
