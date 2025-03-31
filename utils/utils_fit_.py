import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

def compare_pth(state_dict1, state_dict2):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        print("参数类别不一致")
        print(f"文件1参数：{keys1 - keys2}")
        print(f"文件2参数：{keys2 - keys1}")
        return

    # 如果键一致，逐个比较参数值
    for key in keys1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # 使用torch.allclose检查参数值是否几乎相等
        if not torch.equal(param1, param2):
            print(f"参数 {key} 不一致")

    print("检查完毕") 

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, optimizer_add, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    loss_s_train= 0
    loss_s_var  = 0
    epoch_step = epoch_step // 5       # 每次epoch只随机用训练集合的一部分 防止过拟合

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    #torch.save(model.encoder.state_dict(), "encoder1.pth" )
    #initial_params = model.backbone.state_dict()
    #torch.save(initial_params, "backbone.pth" )
    #torch.save(model.backbone.state_dict(), "backbone12.pth" )
    #compare_pth(initial_params,params2)
    for iteration, batch in enumerate(gen):
        
        if iteration >= epoch_step:
            break
        #model.backbone.eval()
        #model.neck.eval()
        #model.loss_s.eval()
        #model.head.eval()
        #model.encoder.eval()
        images, targets = batch[0], batch[1]
        #print(torch.mean(images))
        #torch.save(model.backbone.state_dict(), "backbone2.pth" )
        with torch.no_grad():
            #torch.save(model.backbone.state_dict(), "backbone2.pth" )
            if cuda:
                images  = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        #----------------------#
        #   清零梯度
        #----------------------#
        #torch.save(model.backbone.state_dict(), "backbone2.pth" )
        optimizer.zero_grad()
        outputs ,  loss_s    = model_train(images)
        loss_value = yolo_loss(outputs, targets) 
        loss_value.backward(retain_graph=True)
        optimizer.step()

        optimizer_add.zero_grad()
        loss_s.backward()
        optimizer_add.step()
        #torch.save(model.backbone.state_dict(), "backbone3.pth" )
        if ema:
            ema.update(model_train)
        #torch.save(model.backbone.state_dict(), "backbone4.pth" )
        loss += loss_value.item()
        loss_s_train += loss_s.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'loss_s'  : loss_s_train / (iteration + 1), 
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
        model.eval()
        torch.save(model.loss_s.state_dict(), "loss1.pth" )
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
            outputs,loss_s = model_train_eval(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets) 

        val_loss += loss_value.item()
        loss_s_var += loss_s.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'loss_s'  : loss_s_var / (iteration + 1)})
            pbar.update(1)
        torch.save(model.loss_s.state_dict(), "loss2.pth" )
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val, loss_s_var / epoch_step_val )
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
            state_dict_loss   = ema.ema.loss_s.state_dict()
        else:
            state_dict_backbone = model.backbone.state_dict()
            state_dict_neck     = model.neck.state_dict()
            state_dict_head     = model.head.state_dict()
            state_dict_loss   = model.loss_s.state_dict()
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            save_dir_ep = os.path.join(save_dir, "ep%03d" % (epoch + 1))
            if not os.path.exists(save_dir_ep):
                os.makedirs(save_dir_ep)
            torch.save(state_dict_backbone, os.path.join(save_dir_ep, "backbone.pth" ))
            torch.save(state_dict_neck, os.path.join(save_dir_ep, "neck.pth" ))
            torch.save(state_dict_head, os.path.join(save_dir_ep, "head.pth" ))
            torch.save(state_dict_loss, os.path.join(save_dir_ep, "loss.pth" ))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            save_dir_ep = os.path.join(save_dir, "best")
            if not os.path.exists(save_dir_ep):
                os.makedirs(save_dir_ep)
            torch.save(state_dict_backbone, os.path.join(save_dir_ep, "backbone.pth" ))
            torch.save(state_dict_neck, os.path.join(save_dir_ep, "neck.pth"))
            torch.save(state_dict_head, os.path.join(save_dir_ep, "head.pth"))
            #torch.save(state_dict_loss, os.path.join(save_dir_ep, "loss.pth" ))

        if len(loss_history.loss_s_var) <= 1 or (loss_s_var / epoch_step_val) <= min(loss_history.loss_s_var):
            print('Save best model_loss to best_epoch_weights.pth')
            save_dir_ep = os.path.join(save_dir, "best")
            torch.save(state_dict_loss, os.path.join(save_dir_ep, "loss.pth" ))



    
