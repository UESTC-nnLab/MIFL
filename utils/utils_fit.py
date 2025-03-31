import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    epoch_step = epoch_step // 5       # 每次epoch只随机用训练集合的一部分 防止过拟合

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        #model.backbone.eval()
        #model.neck.eval()
        #model.head.eval()
        images, targets = batch[0], batch[1]
        #print(torch.mean(image))
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            #outputs, motion_loss        = model_train(images)
            outputs      = model_train(images)
           # outputs , old_feat     = model_train(images,old_feat)
            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets) #+ motion_loss

            #----------------------#
            #   反向传播
            #----------------------#
            # torch.autograd.set_detect_anomaly(True)
            # with torch.autograd.detect_anomaly():
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                #----------------------#
                #   计算损失
                #----------------------#
                loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs  = model_train_eval(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss_orin(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        if eval_callback:
            eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        

        #torch.save(model.backbone.state_dict(), "backbone3.pth" )
        if ema:
            state_dict_backbone = ema.ema.backbone.state_dict()
            state_dict_neck     = ema.ema.neck.state_dict()
            state_dict_head     = ema.ema.head.state_dict()
            #state_dict_modify   = ema.ema.modify.state_dict()
        else:
            state_dict_backbone = model.backbone.state_dict()
            state_dict_neck     = model.neck.state_dict()
            state_dict_head     = model.head.state_dict()
            #state_dict_modify   = model.modify.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_dir_ep = os.path.join(save_dir, "ep%03d" % (epoch + 1))
            if not os.path.exists(save_dir_ep):
                os.makedirs(save_dir_ep)
            torch.save(state_dict_backbone, os.path.join(save_dir_ep, "backbone.pth" ))
            torch.save(state_dict_neck, os.path.join(save_dir_ep, "neck.pth" ))
            torch.save(state_dict_head, os.path.join(save_dir_ep, "head.pth" ))
            #torch.save(state_dict_modify, os.path.join(save_dir_ep, "modify.pth" ))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            save_dir_ep = os.path.join(save_dir, "best")
            if not os.path.exists(save_dir_ep):
                os.makedirs(save_dir_ep)
            torch.save(state_dict_backbone, os.path.join(save_dir_ep, "backbone.pth" ))
            torch.save(state_dict_neck, os.path.join(save_dir_ep, "neck.pth"))
            torch.save(state_dict_head, os.path.join(save_dir_ep, "head.pth"))
            #torch.save(state_dict_modify, os.path.join(save_dir_ep, "modify.pth" ))
    
    





    
