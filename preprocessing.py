import spacy
import tqdm
import config
from hazm import Normalizer
from transformers import AutoTokenizer
from typing import Tuple,List,Dict


def clean_text(text:str):
    return Normalizer().normalize(text)

def prepare_conll_data_format(
    path: str,
    sep: str = "\t",
    lower: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prepare data in CoNNL like format.
    Tokens and labels separated on each line.
    Sentences are separated by empty line.
    Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

    Data example:
    token_11    label_11
    token_12    label_12

    token_21    label_21
    token_22    label_22
    token_23    label_23

    ...
    """

    token_seq = []
    label_seq = []
    with open(path, mode="r",encoding="utf-8") as fp:
        tokens = []
        labels = []
        if verbose:
            fp = tqdm(fp)
        for line in fp:
            if line != "\n":
                token, label = line.strip().split(sep)
                if lower:
                    token = token.lower()

                # token = clean_text(token)
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq


def get_label2idx(label_set: List[str]) -> Dict[str, int]:
    """
    Get mapping from labels to indices.
    """

    label2idx: Dict[str, int] = {}

    for label in label_set:
        label2idx[label] = len(label2idx)

    return label2idx

def get_vocab(tokenizer):
    """
        return Token2Index(Dict)
    """
    return tokenizer.vocab


def process_tokens(
    tokens: List[str], token2idx: Dict[str, int], unk: str = "[UNK]"
) -> List[int]:
    """
    Transform list of tokens into list of tokens' indices.
    """

    processed_tokens = [token2idx.get(token, token2idx[unk]) for token in tokens]
    return processed_tokens


def process_labels(labels: List[str], label2idx: Dict[str, int]) -> List[int]:
    """
    Transform list of labels into list of labels' indices.
    """

    processed_labels = [label2idx[label] for label in labels]
    return processed_labels