# CoMix Framework training code
import os
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import params
from utils import *
from models.graph_model import *
import time
import datetime
import copy
from torch.autograd import Function

from core.losses import SupConLoss 
from i3d.pytorch_i3d import InceptionI3d

import random

import warnings
warnings.filterwarnings("ignore")


def print_line():
    print('-'*100)

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # Cross Entropy loss after smoothing the labels
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)

        return loss


def simclr_loss_unlab(output_fast,output_slow,normalize=True):
    criterion = torch.nn.CrossEntropyLoss()
    logits, labels = info_nce_loss(torch.cat((output_fast, output_slow), dim=0))
    return criterion(logits, labels)

def simclr_loss(output_fast, output_slow, criterion, labels=None, normalize=True):
    output_fast = torch.unsqueeze(output_fast, dim=1)
    output_slow = torch.unsqueeze(output_slow, dim=1)
    output_new = torch.cat((output_fast, output_slow), dim=1)
    #logits, labels = info_nce_loss(torch.cat((output_fast, output_slow), dim=0))
    return criterion(output_new, labels)



#####...Train CoMix...#####
def train_comix(graph_model, src_data_loader, tgt_data_loader=None, data_loader_eval=None, num_iterations=10000):
    # Trainer function
    
    graph_model.train()
    graph_model = nn.DataParallel(graph_model)

    i3d_online = InceptionI3d(400, in_channels=3)
    i3d_online.load_state_dict(torch.load("./models/rgb_imagenet.pt"))
    
    i3d_online.train()
    i3d_online.cuda()
    i3d_online = nn.DataParallel(i3d_online)

    random.seed(params.manual_seed)

    criterion = nn.CrossEntropyLoss().cuda()
    temperature=0.5
    simclr_loss_criterion = SupConLoss(temperature=temperature)

    optimizer = optim.SGD([ {"params": i3d_online.parameters(), "lr": params.learning_rate * 0.1},
                            {"params": graph_model.parameters(), "lr": params.learning_rate}],
                            lr=params.learning_rate,
                            weight_decay=0.0000001,
                            momentum=params.momentum )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_iterations))

    len_source_data_loader = len(src_data_loader) - 1 
    len_target_data_loader = len(tgt_data_loader) - 1

    print_line()
    print('len_source_data_loader = '+ str(len_source_data_loader))
    print('len_target_data_loader = '+ str(len_target_data_loader))

    best_accuracy_yet = 0.0
    best_itrn = 0
    best_model_wts = copy.deepcopy(graph_model.state_dict())
    best_i3d_model_wts = copy.deepcopy(i3d_online.state_dict())

    print_line()
    print_line()
    print_line()

    start_time = time.process_time()
    running_lr = params.learning_rate
    epoch_number = 0
    start_iter = 0
    if params.dataset_name=="UCF-HMDB":
        num_classes = 12
    elif params.dataset_name=="Jester":
        num_classes = 7
    elif params.dataset_name=="Epic-Kitchens":
        num_classes = 8

    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)

    if params.warmstart_models=='True': #changes needed for i3d online version
        if params.warmstart_graph=='None' or params.warmstart_i3d=='None':
            print('Starting Training for Warmstarting...')
            graph_model, i3d_online = warmstart_models(graph_model, i3d_online, src_data_loader, None, data_loader_eval, params.num_iter_warmstart)            
            print('Warmstarted successfully...')
        else:
            print('Warmstarting...')
            graph_model.load_state_dict(torch.load(params.warmstart_graph))
            i3d_online.load_state_dict(torch.load(params.warmstart_i3d))
            print('Warmstarted successfully...')

    if params.auto_resume=='True':
        checkpoint_path = os.path.join(params.model_root, "Current-Checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_iter = checkpoint['iter']
            epoch_number = checkpoint['epoch_number']
            graph_model.load_state_dict(checkpoint['graph'])
            i3d_online.load_state_dict(checkpoint['i3d'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print('Auto resuming from itrn: ', start_iter)
            print('Best acc yet: ', checkpoint['best_accuracy_yet'])
            best_accuracy_yet = checkpoint['best_accuracy_yet']
            best_itrn = checkpoint['best_itrn']

    for itrn in range(start_iter, num_iterations):
        print("\rRunning Iteration: {}/{}".format(itrn, num_iterations), end='', flush=True)
        if itrn%100 == 0:
            print('Itrn: (T)', itrn+1, 'LR:', scheduler.get_lr())

        if itrn==start_iter:
            iter_source = iter(src_data_loader)
            iter_target = iter(tgt_data_loader)

        if itrn % len_source_data_loader == 0:
            iter_source = iter(src_data_loader)
            epoch_number = epoch_number + 1

        if itrn % len_target_data_loader == 0:            
            iter_target = iter(tgt_data_loader)

        SRC, labels = iter_source.next()
        feat_src = SRC[0]
        bg_src = SRC[1]
        TGT, _ = iter_target.next()
        feat_tgt = TGT[0]
        bg_tgt = TGT[1]

        if params.random_aux=='True':
            random_decider = random.uniform(0, 1)
            if random_decider < 0.33:
                num_slow_nodes = 4
            elif random_decider >= 0.33 and random_decider < 0.67:
                num_slow_nodes = 8
            else:
                num_slow_nodes = 12
        else:
            num_slow_nodes = 8


        mix_ratio = np.random.uniform(0, params.max_gamma)


        src_mix_tgt_bg = (feat_src*(1-mix_ratio)) + (bg_tgt.unsqueeze(1)*mix_ratio)
        tgt_mix_src_bg = (feat_tgt*(1-mix_ratio)) + (bg_src.unsqueeze(1)*mix_ratio)

        src_mix_tgt_bg = src_mix_tgt_bg.float()
        src_mix_tgt_bg = make_variable(src_mix_tgt_bg, gpu_id=params.src_gpu_id)
        
        tgt_mix_src_bg = tgt_mix_src_bg.float()
        tgt_mix_src_bg = make_variable(tgt_mix_src_bg, gpu_id=params.tgt_gpu_id)


        feat_src = make_variable(feat_src, gpu_id=params.src_gpu_id)
        feat_tgt = make_variable(feat_tgt, gpu_id=params.tgt_gpu_id)
        labels = make_variable(labels)

        optimizer.zero_grad()

        bs, num_nodes, num_c, chunk_size, H, W = feat_src.shape

        feat_src = feat_src.reshape(bs*num_nodes, num_c, chunk_size, H, W)
        i3d_feat_src = i3d_online(feat_src)
        feat_tgt = feat_tgt.reshape(bs*num_nodes, num_c, chunk_size, H, W)
        i3d_feat_tgt = i3d_online(feat_tgt)


        src_mix_tgt_bg = src_mix_tgt_bg.reshape(bs*num_nodes, num_c, chunk_size, H, W)
        i3d_src_mix_tgt_bg = i3d_online(src_mix_tgt_bg)
        tgt_mix_src_bg = tgt_mix_src_bg.reshape(bs*num_nodes, num_c, chunk_size, H, W)
        i3d_tgt_mix_src_bg = i3d_online(tgt_mix_src_bg)

        #------Slow range---------------
        fastRange = np.arange(num_nodes)
        splitRange = np.array_split(fastRange, num_slow_nodes)
        slowIds = [np.random.choice(a) for a in splitRange]
        #------Reshaping part-------------
        # required shape for graph is (bs, num_classes, 1024)
        i3d_feat_src = i3d_feat_src.squeeze(3).squeeze(3)
        i3d_feat_src = i3d_feat_src.reshape(bs, num_nodes, -1)
        i3d_feat_src_slow = i3d_feat_src[:,slowIds,:]

        i3d_feat_tgt = i3d_feat_tgt.squeeze(3).squeeze(3)
        i3d_feat_tgt = i3d_feat_tgt.reshape(bs, num_nodes, -1)
        i3d_feat_tgt_slow = i3d_feat_tgt[:,slowIds,:]

        # bs_2 = bs // 2
        i3d_src_mix_tgt_bg = i3d_src_mix_tgt_bg.squeeze(3).squeeze(3)
        i3d_src_mix_tgt_bg = i3d_src_mix_tgt_bg.reshape(bs, num_nodes, -1)
        i3d_src_mix_tgt_bg_slow = i3d_src_mix_tgt_bg[:,slowIds,:]

        i3d_tgt_mix_src_bg = i3d_tgt_mix_src_bg.squeeze(3).squeeze(3)
        i3d_tgt_mix_src_bg = i3d_tgt_mix_src_bg.reshape(bs, num_nodes, -1)
        i3d_tgt_mix_src_bg_slow = i3d_tgt_mix_src_bg[:,slowIds,:]
        #---------------------------------
        preds_src = graph_model(i3d_feat_src)
        preds_src_slow = graph_model(i3d_feat_src_slow)
        preds_tgt = graph_model(i3d_feat_tgt)
        preds_tgt_slow = graph_model(i3d_feat_tgt_slow)

        preds_src_mix = graph_model(i3d_src_mix_tgt_bg)
        preds_src_mix_slow = graph_model(i3d_src_mix_tgt_bg_slow)

        preds_tgt_mix = graph_model(i3d_tgt_mix_src_bg)
        preds_tgt_mix_slow = graph_model(i3d_tgt_mix_src_bg_slow)

        cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, size_average=False)(preds_src, labels).mean()
        


        target_logits_fused = (preds_tgt.data + preds_tgt_slow.data) / 2
        target_logits_softmax = torch.softmax(target_logits_fused, dim=-1)

        max_probs, target_pseudo_labels = torch.max(target_logits_softmax, dim=-1)
        pseudo_mask = max_probs.ge(params.pseudo_threshold).float()

        pseudo_cls_loss = (F.cross_entropy(preds_tgt, target_pseudo_labels, reduction='none') * pseudo_mask).mean()


        virtual_label = torch.arange(0, bs)
        virtual_label = torch.cat((virtual_label, virtual_label), dim=0)
        virtual_label = make_variable(virtual_label)

        sim_clr_loss_src = simclr_loss(torch.softmax(preds_src, dim=-1), torch.softmax(preds_src_slow, dim=-1),  simclr_loss_criterion, labels)
        if pseudo_mask is None : 
            sim_clr_loss_tgt = simclr_loss(preds_tgt, preds_tgt_slow, simclr_loss_criterion)
        else :
            pseudo_mask_new = pseudo_mask > 0
            preds_tgt_masked = torch.masked_select(preds_tgt, pseudo_mask_new.unsqueeze(dim=1).repeat(1, preds_tgt.shape[1]))
            preds_tgt_slow_masked = torch.masked_select(preds_tgt_slow, pseudo_mask_new.unsqueeze(dim=1).repeat(1, preds_tgt_slow.shape[1]))
            target_pseudo_labels_masked = torch.masked_select(target_pseudo_labels, pseudo_mask_new)
            preds_tgt_masked = preds_tgt_masked.reshape(-1, num_classes)
            preds_tgt_slow_masked = preds_tgt_slow_masked.reshape(-1, num_classes)

            if preds_tgt_slow_masked.shape[0] > 0 :
                sim_clr_loss_tgt = simclr_loss(torch.softmax(preds_tgt_masked, dim=-1), torch.softmax(preds_tgt_slow_masked, dim=-1), simclr_loss_criterion, target_pseudo_labels_masked)
            else :
                sim_clr_loss_tgt = torch.tensor(0.0).cuda() 
         
        

        src_fast = torch.cat((preds_src, preds_src_mix), dim=0)
        src_slow = torch.cat((preds_src_slow, preds_src_mix_slow), dim=0)
        simclr_mod_src = simclr_loss(torch.softmax(src_fast, dim=-1), torch.softmax(src_slow, dim=-1), simclr_loss_criterion, virtual_label)

        tgt_fast = torch.cat((preds_tgt, preds_tgt_mix), dim=0)
        tgt_slow = torch.cat((preds_tgt_slow, preds_tgt_mix_slow), dim=0)
        simclr_mod_tgt = simclr_loss(torch.softmax(tgt_fast, dim=-1), torch.softmax(tgt_slow, dim=-1), simclr_loss_criterion, virtual_label)

        simclr_mod_mix = simclr_mod_src + simclr_mod_tgt
        
        pseudo_cls_loss = torch.tensor(0.0).cuda()
        loss = cls_loss + (params.lambda_bgm * (simclr_mod_mix)) + (params.lambda_tpl * (sim_clr_loss_tgt))
        
        loss.backward()
     
        optimizer.step()
        
        scheduler.step()

        # Log updates.
        if ((itrn + 1) % params.log_in_steps == 0):
            print_line()
            print("Iteration [{}/{}]: ce_loss={} bgm_loss={} tpl_loss={} total_loss={}"
                  .format(itrn + 1,
                          num_iterations,
                          cls_loss.item(),
                          simclr_mod_mix.item(),
                          sim_clr_loss_tgt.item(),
                          loss.item()                          
                          ))
            print_line()
            print_line()

        # Evaluate model on the validation set.
        if ((itrn + 1) % params.eval_in_steps == 0):
            loss_val = 0.0
            acc_val = 0.0
            start_time_eval = time.process_time()
            tot_len = 0.0
            graph_model.eval()
            i3d_online.eval()
            for step_val, (feats_val, labels_val) in enumerate(data_loader_eval):
                if step_val % 10 == 0:
                    print("\rEvaluating batch {}/{}".format(step_val, len(data_loader_eval)), end='', flush=True)

                feats_val = feats_val[0]
                feats_val = make_variable(
                    feats_val, gpu_id=params.src_gpu_id, volatile=True)
                labels_val = make_variable(
                    labels_val, gpu_id=params.src_gpu_id)

                bs, num_nodes, num_c, chunk_size, H, W = feats_val.shape

                feats_val = feats_val.reshape(bs*num_nodes, num_c, chunk_size, H, W)
                i3d_feats_eval = i3d_online(feats_val)
                #------reshaping part--------------
                i3d_feats_eval = i3d_feats_eval.squeeze(3).squeeze(3)
                i3d_feats_eval = i3d_feats_eval.reshape(bs, num_nodes, -1)
                #----------------------------------
                preds_val = graph_model(i3d_feats_eval)

                loss_val += criterion(preds_val, labels_val).data.item()

                _, preds_val_val = torch.max(preds_val.data, 1)
                acc_val += torch.sum(preds_val_val == labels_val.data)

                del feats_val, i3d_feats_eval, labels_val, preds_val, preds_val_val

            print('\ncorrect_feats: ', acc_val)
            print('total_feats: ', len(data_loader_eval.dataset))

            avg_loss_val = loss_val / len(data_loader_eval)
            avg_acc_val = float(acc_val.cpu().numpy()) / len(data_loader_eval.dataset)

            checkpoint_path_current = os.path.join(params.model_root, "Current-Checkpoint.pt")
            torch.save({
                        'iter':itrn+1,
                        'epoch_number':epoch_number,
                        'graph':graph_model.state_dict(),
                        'i3d':i3d_online.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'best_accuracy_yet':best_accuracy_yet,
                        'best_itrn':best_itrn
                        },checkpoint_path_current)

            if(best_accuracy_yet <= avg_acc_val):
                best_accuracy_yet = avg_acc_val
                best_model_wts = copy.deepcopy(graph_model.state_dict())
                best_i3d_model_wts = copy.deepcopy(i3d_online.state_dict())

                save_model(graph_model, "Graph-CoMix-Model-Best.pth")
                save_model(i3d_online, "I3D-CoMix-Model-Best.pth")

                best_itrn = itrn + 1
                checkpoint_path_best = os.path.join(params.model_root, "Best-Checkpoint.pt")
                torch.save({
                        'iter':itrn+1,
                        'epoch_number':epoch_number,
                        'graph':graph_model.state_dict(),
                        'i3d':i3d_online.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'best_accuracy_yet':best_accuracy_yet,
                        'best_itrn':best_itrn
                        },checkpoint_path_best)

            print('best_acc_yet: ', best_accuracy_yet, ' ( in itrn:', best_itrn, ')...')
            graph_model.train()
            i3d_online.train()
            print_line()

            end_time_eval = time.process_time()
            print("Avg Loss = {}, Avg Acc = {}".format(avg_loss_val, str(avg_acc_val)))
            print_line()

        del feat_src, labels, preds_src, preds_src_slow, i3d_feat_src, feat_tgt, preds_tgt, preds_tgt_slow, i3d_feat_tgt

    # Load the best models and save them.
    print('Loading the best model weights...')
    graph_model.load_state_dict(best_model_wts)
    i3d_online.load_state_dict(best_i3d_model_wts)

    save_model(graph_model, "Graph-CoMix-Model-Best-{}.pth".format(best_itrn))
    save_model(i3d_online, "I3D-CoMix-Model-Best-{}.pth".format(best_itrn))

    return graph_model



def warmstart_models(graph_model, i3d_online, src_data_loader, tgt_data_loader=None, data_loader_eval=None, num_iterations=10000):
    # Trainer function
    
    optimizer = optim.SGD([ {"params": i3d_online.parameters(), "lr": params.learning_rate_ws * 0.1},
                            {"params": graph_model.parameters(), "lr": params.learning_rate_ws}],
                            lr=params.learning_rate,
                            weight_decay=0.0000001,
                            momentum=params.momentum )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(num_iterations))

    criterion = nn.CrossEntropyLoss().cuda()

    len_source_data_loader = len(src_data_loader) - 1 

    print_line()
    print('len_source_data_loader = '+ str(len_source_data_loader))

    best_accuracy_yet = 0.0
    best_itrn = 0
    best_model_wts = copy.deepcopy(graph_model.state_dict())
    best_i3d_model_wts = copy.deepcopy(i3d_online.state_dict())

    print_line()
    print_line()
    print_line()

    start_time = time.process_time()
    running_lr = params.learning_rate
    epoch_number = 0
    start_iter = 0
    if params.dataset_name=="UCF-HMDB":
        num_classes = 12
    elif params.dataset_name=="Jester":
        num_classes = 7
    elif params.dataset_name=="Epic-Kitchens":
        num_classes = 8

    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)


    for itrn in range(start_iter, num_iterations):
        print("\rRunning Iteration (source-only) : {}/{}".format(itrn, num_iterations), end='', flush=True)
        if itrn%100 == 0:
            print('Itrn: (T)', itrn+1, 'LR:', scheduler.get_lr())

        if itrn==start_iter:
            iter_source = iter(src_data_loader)

        if itrn % len_source_data_loader == 0:
            iter_source = iter(src_data_loader)
            epoch_number = epoch_number + 1

        SRC, labels = iter_source.next()
        feat_src = SRC[0]

        feat_src = make_variable(feat_src, gpu_id=params.src_gpu_id)
        labels = make_variable(labels)

        optimizer.zero_grad()

        bs, num_nodes, num_c, chunk_size, H, W = feat_src.shape

        feat_src = feat_src.reshape(bs*num_nodes, num_c, chunk_size, H, W)
        i3d_feat_src = i3d_online(feat_src)

        i3d_feat_src = i3d_feat_src.squeeze(3).squeeze(3)
        i3d_feat_src = i3d_feat_src.reshape(bs, num_nodes, -1)

        preds_src = graph_model(i3d_feat_src)

        cls_loss = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, size_average=False)(preds_src, labels).mean()
        
        loss = cls_loss
        
        loss.backward()
     
        optimizer.step()
        
        scheduler.step()

        # Log updates.
        if ((itrn + 1) % params.log_in_steps == 0):
            print_line()
            print("Iteration (T) [{}/{}]: cls_loss={}"
                  .format(itrn + 1,
                          num_iterations,
                          cls_loss.item()
                          ))
            print_line()
            print_line()

        # Evaluate model on the validation set.
        if ((itrn + 1) % params.eval_in_steps == 0):
            loss_val = 0.0
            acc_val = 0.0
            start_time_eval = time.process_time()
            tot_len = 0.0
            graph_model.eval()
            i3d_online.eval()
            for step_val, (feats_val, labels_val) in enumerate(data_loader_eval):
                if step_val % 10 == 0:
                    print("\rEvaluating batch {}/{}".format(step_val, len(data_loader_eval)), end='', flush=True)

                feats_val = feats_val[0]
                feats_val = make_variable(
                    feats_val, gpu_id=params.src_gpu_id, volatile=True)
                labels_val = make_variable(
                    labels_val, gpu_id=params.src_gpu_id)

                bs, num_nodes, num_c, chunk_size, H, W = feats_val.shape

                feats_val = feats_val.reshape(bs*num_nodes, num_c, chunk_size, H, W)
                i3d_feats_eval = i3d_online(feats_val)
                #------reshaping part--------------
                i3d_feats_eval = i3d_feats_eval.squeeze(3).squeeze(3)
                i3d_feats_eval = i3d_feats_eval.reshape(bs, num_nodes, -1)
                #----------------------------------
                preds_val = graph_model(i3d_feats_eval)

                loss_val += criterion(preds_val, labels_val).data.item()

                _, preds_val_val = torch.max(preds_val.data, 1)
                acc_val += torch.sum(preds_val_val == labels_val.data)

                del feats_val, i3d_feats_eval, labels_val, preds_val, preds_val_val

            print('\ncorrect_feats: ', acc_val)
            print('total_feats: ', len(data_loader_eval.dataset))

            avg_loss_val = loss_val / len(data_loader_eval)
            avg_acc_val = float(acc_val.cpu().numpy()) / len(data_loader_eval.dataset)

            if(best_accuracy_yet <= avg_acc_val):
                best_accuracy_yet = avg_acc_val
                best_model_wts = copy.deepcopy(graph_model.state_dict())
                best_i3d_model_wts = copy.deepcopy(i3d_online.state_dict())

                save_model(graph_model, "Graph-SourceOnly-Model-Best.pth")
                save_model(i3d_online, "I3D-SourceOnly-Online-Model-Best.pth")

                best_itrn = itrn + 1

            print('best_acc_yet: ', best_accuracy_yet, ' ( in itrn:', best_itrn, ')...')
            graph_model.train()
            i3d_online.train()
            print_line()

            end_time_eval = time.process_time()
            print("Avg Loss = {}, Avg Acc = {}".format(avg_loss_val, str(avg_acc_val)))
            print_line()

        del feat_src, labels, preds_src, i3d_feat_src

    # Load the best models and save them.
    print('Loading the best model weights...')
    graph_model.load_state_dict(best_model_wts)
    i3d_online.load_state_dict(best_i3d_model_wts)

    save_model(graph_model, "Graph-SourceOnly-Model-Best-{}.pth".format(best_itrn))
    save_model(i3d_online, "I3D-SourceOnly-Model-Best-{}.pth".format(best_itrn))

    return graph_model, i3d_online