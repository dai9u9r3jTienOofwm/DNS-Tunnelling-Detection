import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
SEED = 42
MAX_LEN = 253
EMBED_DIM = 32
NUM_FILTERS = 64
KERNEL_SIZE = 5
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 4
TEST_SIZE = 0.2
