import torch
import numpy as np
import config
from model import Bert
from transformers import BertTokenizer, BertForTokenClassification,AutoModelForTokenClassification


def predict(model, tokenizer, text):
    model.eval()

    # Tokenize input text
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")
    print(tokens,"\n\n")
    print(inputs,"\n\n")
    # Make predictions
    with torch.no_grad():
        outputs = model(inputs).logits
    print(outputs.shape,"\n\n")
    # Convert output indices to labels
    predicted_labels = [config.IDX2LABEL[idx] for idx in torch.argmax(outputs, dim=2).squeeze().tolist()]
    print(predicted_labels,"\n\n")
    # Filter out 'O' labels and handle 'B' and 'I' labels
    named_entities = []
    current_entity = {"word": "", "tag": None}

    for token, label in zip(tokens, predicted_labels[1:-1]):  # Exclude [CLS] and [SEP]
        if label == 'O':
            continue
        prefix, tag = label.split('-')
        
        if prefix == 'B' or (prefix == 'I' and tag != current_entity["tag"]):
            if current_entity["word"]:
                named_entities.append((current_entity["word"], current_entity["tag"]))
            current_entity = {"word": token, "tag": tag}
        else:
            current_entity["word"] += ' ' + token

    # Add the last entity
    if current_entity["word"]:
        named_entities.append((current_entity["word"], current_entity["tag"]))

    return named_entities


text = "دفتر مرکزی شرکت پارس‌مینو در شهر اراک در استان مرکزی قرار دارد."
model = AutoModelForTokenClassification.from_pretrained(config.MODEL_NAME,num_labels=13)
model.load_state_dict(torch.load('parsbert.pth'))
tokenizer = BertTokenizer.from_pretrained("HooshVareLab/bert-base-parsbert-uncased")
name_entities = predict(model,tokenizer,text)
print(name_entities)