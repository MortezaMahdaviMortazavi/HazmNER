from transformers import AutoTokenizer

ARMAN_TRAIN_FOLD1 = 'datasets/arman/train_fold1.txt'
ARMAN_TRAIN_FOLD2 = 'datasets/arman/train_fold2.txt'
ARMAN_TRAIN_FOLD3 = 'datasets/arman/train_fold3.txt'
ARMAN_TEST_FOLD1 = 'datasets/arman/test_fold1.txt'
ARMAN_TEST_FOLD2 = 'datasets/arman/test_fold2.txt'
ARMAN_TEST_FOLD3 = 'datasets/arman/test_fold3.txt'
ARMAN_TRAIN_CSV_PATH = 'datasets/arman/train.csv'
ARMAN_TEST_CSV_PATH = 'datasets/arman/test.csv'

ARMAN_TRAIN_TOKENS_PKL = 'datasets/train_tokens_arman.pkl'
ARMAN_TRAIN_LABELS_PKL = 'datasets/train_labels_arman.pkl'
ARMAN_TEST_TOKENS_PKL = 'datasets/test_tokens_arman.pkl'
ARMAN_TEST_TOKENS_PKL = 'datasets/test_labels_arman.pkl'
PEYMA_TRAIN = 'datasets/peyma/train.txt'
PEYMA_TEST = 'datasets/peyma/test.txt'
PEYMA_DEV = 'datasets/peyma/dev.txt'
LOGFILE = 'logs/logfile.log'

LABEL2IDX = {'B-event': 0,
 'B-fac': 1,
 'B-loc': 2,
 'B-org': 3,
 'B-pers': 4,
 'B-pro': 5,
 'I-event': 6,
 'I-fac': 7,
 'I-loc': 8,
 'I-org': 9,
 'I-pers': 10,
 'I-pro': 11,
 'O': 12}


# LABEL2IDX = {
#   'O': 0,
#  'I_PER': 1,
#  'B_TIM': 2,
#  'I_MON': 3,
#  'I_PCT': 4,
#  'B_PCT': 5,
#  'I_TIM': 6,
#  'I_LOC': 7,
#  'B_PER': 8,
#  'B_DAT': 9,
#  'B_MON': 10,
#  'I_ORG': 11,
#  'B_ORG': 12,
#  'B_LOC': 13,
#  'I_DAT': 14}

IDX2LABEL = {i:k for k,i in LABEL2IDX.items()}
print(IDX2LABEL)

CLS = [101]
SEP = [102]
VALUE_TOKEN = [0]
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
EPOCHS = 5
NUM_CLASSES = 13
GPU_ID = 0
LEARNING_RATE = 5e-5

MODEL_NAME = 'HooshVareLab/bert-base-parsbert-uncased'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)