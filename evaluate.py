import torch
import config
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            ids, mask, token_type_ids, target_tags = batch['ids'], batch['mask'], batch['token_type_ids'], batch['target_tags']
            ids, mask, token_type_ids, target_tags = ids.to(device), mask.to(device), token_type_ids.to(device), target_tags.to(device)

            output, loss = model(ids, mask, token_type_ids)
            
            total_loss += loss.item()

            active_loss = mask.view(-1) == 1
            active_logits = output.view(-1, len(config.LABEL2IDX))
            active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(criterion.ignore_index).type_as(target_tags))

            all_predictions.extend(torch.argmax(active_logits, dim=1).cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)

    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, precision, recall, f1, accuracy
