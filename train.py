import torch
import torch.nn as nn
import config
from tqdm import tqdm

def optimize_batch(model,batch,criterion,optimizer,device,scheduler=None):
    ids , mask , token_type_ids , target_tags = batch['ids'] , batch['mask'] , batch['token_type_ids'] , batch['target_tags']
    ids , mask , token_type_ids , target_tags = ids.to(device) , mask.to(device) , token_type_ids.to(device) , target_tags.to(device)
    output = model(ids,mask,token_type_ids)
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1,len(config.LABEL2IDX))
    active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(criterion.ignore_index).type_as(target_tags))
    loss = criterion(active_logits,active_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    return loss

def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in tqdm(dataloader):
        loss = optimize_batch(model, batch, criterion, optimizer, device, scheduler)
        total_loss += loss

    average_loss = total_loss / num_batches
    return average_loss


def optimization(model,dataloader,criterion,optimizer,device,scheduler=None):
    for epoch in range(config.EPOCHS):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, scheduler)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}, Average Loss: {avg_loss}")



# def train_fn():
#     #Calculate the loss
#     Critirion_Loss = nn.CrossEntropyLoss()
#     active_loss = mask.view(-1) == 1
#     active_logits = tag.view(-1, self.num_tag)
#     active_labels = torch.where(active_loss, target_tags.view(-1), torch.tensor(Critirion_Loss.ignore_index).type_as(target_tags))
#     loss = Critirion_Loss(active_logits, active_labels)