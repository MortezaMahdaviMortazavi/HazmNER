import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import config

def evaluate_batch(model, criterion, data, targets):
    model.eval()
    logits = model(data)
    loss = criterion(logits, targets)
    return loss.item(), logits

def evaluate_epoch(model, eval_loader, criterion, device, epoch, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Set the filename for the log file
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader)):
            data = batch['input']
            targets = batch['target']
            data, targets = data.to(device), targets.to(device)

            loss, logits = evaluate_batch(model, criterion, data, targets)
            
            total_loss += loss

            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions)

            if batch_idx % 100 == 0:
                logging.info(f'Eval Epoch: {epoch} [{batch_idx}/{len(eval_loader)} '
                             f'({100. * batch_idx / len(eval_loader):.0f}%)]\tLoss: {loss:.6f}')

    average_loss = total_loss / len(eval_loader)

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    logging.info(f'Evaluation:\tAverage Loss: {average_loss:.6f}\tAccuracy: {accuracy:.4f}\t'
                 f'Precision: {precision:.4f}\tRecall: {recall:.4f}\tF1 Score: {f1:.4f}')

    return average_loss

def validate(model, eval_loader, device, log_file=config.logfile):
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Set the filename for the log file
    criterion = nn.CrossEntropyLoss()
    eval_loss = evaluate_epoch(model, eval_loader, criterion, device, 0, log_file)
    logging.info(f'Evaluation:\tAverage Loss: {eval_loss:.6f}')
