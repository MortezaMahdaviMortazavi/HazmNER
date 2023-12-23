import torch
import config
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from seqeval.metrics import classification_report

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids, mask, token_type_ids, target_tags = batch['ids'], batch['mask'], batch['token_type_ids'], batch['target_tags']
            ids, mask, token_type_ids, target_tags = ids.to(device), mask.to(device), token_type_ids.to(device), target_tags.to(device)

            output = model(ids, mask, token_type_ids)
            active_loss = mask.view(-1) == 1
            active_logits = output.view(-1,len(config.LABEL2IDX))
            active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(criterion.ignore_index).type_as(target_tags))
            loss = criterion(active_logits,active_labels)
            total_loss += loss.item()

            active_loss = mask.view(-1) == 1
            active_logits = output.view(-1, len(config.LABEL2IDX))
            active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(criterion.ignore_index).type_as(target_tags))

            all_predictions.extend(torch.argmax(active_logits, dim=1).cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, precision, recall, f1, accuracy


def convert_to_iob_format(labels):
    iob_labels = []
    current_entity = {"label": None, "start": None, "end": None}

    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_entity["label"]:
                iob_labels.append(current_entity["label"])
            current_entity = {"label": label[2:], "start": i, "end": i}
        elif label.startswith("I-"):
            if current_entity["label"] == label[2:]:
                current_entity["end"] = i
            else:
                iob_labels.append(current_entity["label"])
                current_entity = {"label": None, "start": None, "end": None}
        else:
            if current_entity["label"]:
                iob_labels.append(current_entity["label"])
                current_entity = {"label": None, "start": None, "end": None}

    if current_entity["label"]:
        iob_labels.append(current_entity["label"])

    return iob_labels

if __name__ == "__main__":
    true_labels = ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC']
    predicted_labels = ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG']
    true_iob_labels = convert_to_iob_format(true_labels)
    predicted_iob_labels = convert_to_iob_format(predicted_labels)
    print("True IOB Chunks:", true_iob_labels)
    print("Predicted IOB Chunks:", predicted_iob_labels)
    report = classification_report([true_iob_labels], [predicted_iob_labels])
    print("Classification Report:")
    print(report)
