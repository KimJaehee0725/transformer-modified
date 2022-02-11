from copy import deepcopy
from enum import auto
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import autocast


from scheduler import CosineAnnealingWarmUpRestarts


def train(model, train_dataloader, validation_dataloader, target_tokenizer, args):    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    optimizer = Adam(model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer = optimizer, T_0 = args.t0, T_mult = args.t_mult, eta_max = args.eta_max, T_up = args.T_up, gamma = args.gamma)
    criterion = nn.NLLLoss(ignore_index = args.pad_id)
    len_train_batch = len(train_dataloader)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epochs):

        if (epoch+1)%args.valid_epoch == 0 :
            valid_loss = validation(model, validation_dataloader, target_tokenizer, epoch, args)
            wandb.log({"valid_loss" : valid_loss, "epoch" : epoch})

        print(f"training step 시작 | {epoch + 1} | {round((epoch +1)/args.epochs * 100, 2)}%")
        for num, ((source_tensor, source_attention_mask), (target_tensor, decoder_attention_mask)) in enumerate(train_dataloader):

            if num%100 == 0 :
                print(f"{num} | {len_train_batch} | {round((num/len_train_batch)*100, 2)}%", end = "\r")
            optimizer.zero_grad()

            encoder_input = source_tensor.to(device)
            encoder_attention_mask = source_attention_mask.to(device)
            
            target_input = deepcopy(target_tensor[:, :-1])
            target_label = deepcopy(target_tensor[:, 1:])
            
            decoder_input = target_input.to(device)
            decoder_attention_mask = decoder_attention_mask.to(device)
            target_label = target_label.to(device)


            with autocast():
                output = model(encoder_input, encoder_attention_mask, decoder_input, decoder_attention_mask)
                loss = criterion(
                    output.contiguous().view(-1, output.shape[-1]), 
                    target_label.contiguous().view(-1)
                    )

                lr = scheduler.get_lr()[0]

                wandb.log({"loss" : loss, 'epoch' : epoch, "lr" : lr})
                
                scaler.scale(loss).backward()
                # loss.backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                scheduler.step()

def validation(model, dataloader, tokenizer, epoch, args):
    print("validation step 시작")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    criterion = nn.NLLLoss(ignore_index = args.pad_id)
    loss_total = 0
    batch_len = len(dataloader)

    log_num = 0

    with torch.no_grad():
        for num, ((source_tensor, source_attention_mask), (target_tensor, decoder_attention_mask)) in enumerate(dataloader):
            if num%30 == 0 :
                print(f"{round(num/batch_len * 100, 2)}% 완료")

            encoder_input = source_tensor.to(device)
            encoder_attention_mask = source_attention_mask.to(device)
            
            target_label = deepcopy(target_tensor[:, 1:])
            target_label = target_label.to(device)
            with autocast():
                output = model(encoder_input, encoder_attention_mask)
                loss = criterion(
                    output.contiguous().view(-1, output.shape[-1]), 
                    target_label.contiguous().view(-1)
                    )
            loss_total += loss.sum()

            if num == 0 :
                predicted = (torch.argmax(output, dim = 2).cpu().tolist())
                generated = tokenizer.batch_decode(predicted)

                real = tokenizer.batch_decode(target_label.cpu().tolist())
                text_table = wandb.Table(columns = ["epoch", "loss", "generated", "real"])
                for gener, lab in zip(generated, real) : 
                    text_table.add_data(epoch, loss, gener, lab)
                wandb.log({f"valid_samples_{log_num}" : text_table})
                log_num+=1
    print("validation step 종료")
    model.train()
    return loss_total/batch_len
