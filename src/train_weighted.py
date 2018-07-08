#============ Custom tensorboard logging ============#
from utils.TbLogger import Logger

#============ Basic imports ============#
import pickle
import gc
import cv2
import copy
import os
import time
import tqdm
import glob
import shutil
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from skimage.io import imsave,imread

# set no multi-processing for cv2 to avoid collisions with data loader
cv2.setNumThreads(0)

#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn import Sigmoid

#============ Models with presets ============#
from models.model_params import model_presets

#============ Loss ============#
from models.Loss import HardDice
from models.Loss import SemsegLossWeighted as SemsegLoss

#============ Utils and augs ============#
from utils.LRScheduler import CyclicLR
from utils.MapDataset import MapDataset
from utils.Metric import calculate_ap
from utils.Watershed import energy_baseline as wt_seeds
from utils.Watershed import label_baseline as wt_baseline
from aug.AugPresets import TrainAugsIaa,TrainAugs,ValAugs
from utils.Util import str2bool,restricted_float,to_np

#============ Precision, Recall computing and visualizing util ============#
from visualize_util import mix_vis_masks

parser = argparse.ArgumentParser(description='CrowdAI mapping challenge params')

cv2.setNumThreads(0)

# ============ basic params ============#
parser.add_argument('--workers',             default=4,             type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',              default=50,            type=int, help='number of total epochs to run')
parser.add_argument('--epoch_fraction',      default=1.0,           type=float, help='break out of train/val loop on some fraction of the dataset - useful for huge datansets with shuffle')
parser.add_argument('--start-epoch',         default=0,             type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size',          default=64,            type=int, help='mini-batch size (default: 64)')
# add here self-ensembling flags

# ============ data loader and model params ============#
parser.add_argument('--do_augs',             default=False,         type=str2bool, help='Whether to use augs')
parser.add_argument('--do_more_augs',        default=False,         type=str2bool, help='Whether to use heavier augs')
parser.add_argument('--aug_prob',            default=0.25,          type=float, help='Augmentation probability')

parser.add_argument('--arch',                default='linknet34',   type=str, help='text id for saving logs')
parser.add_argument('--ths',                 default=0.5,           type=float, help='mask post-processing threshold')

parser.add_argument('--do_energy_levels',    default=False,         type=str2bool, help='Whether to use mask erosion')
parser.add_argument('--do_boundaries',       default=False,         type=str2bool, help='Whether to predict mask borders')
parser.add_argument('--img_size',            default=300,           type=int, help='Image resize')

# ============ optimization params ============#
parser.add_argument('--lr',                  default=1e-3,          type=float, help='initial learning rate')
parser.add_argument('--m0',                  default=5,             type=int, help='encoder unfreeze milestone')
parser.add_argument('--m1',                  default=5,             type=int, help='lr decay milestone 1')
parser.add_argument('--m2',                  default=20,            type=int, help='lr decay milestone 2')
parser.add_argument('--m3',                  default=30,            type=int, help='dice boost milestone')
parser.add_argument('--optimizer',           default='adam',        type=str, help='model optimizer')
parser.add_argument('--do_running_mean',     default=False,         type=str2bool, help='Whether to use running mean for loss')
parser.add_argument('--bce_weight',          default=0.5,           type=float, help='BCE loss weight')
parser.add_argument('--dice_weight',         default=0.5,           type=float, help='DICE loss weight')

parser.add_argument('--w0',                  default=5.0,           type=float, help='DICE loss weight')
parser.add_argument('--sigma',               default=10.0,          type=float, help='DICE loss weight')

parser.add_argument('--do_remove_small_on_borders',     default=False,         type=str2bool, help='Whether to use running mean for loss')
parser.add_argument('--do_produce_sizes_mask',          default=False,         type=str2bool, help='Whether to use running mean for loss')
parser.add_argument('--do_produce_distances_mask',      default=False,         type=str2bool, help='Whether to use running mean for loss')

# ============ logging params and utilities ============#
parser.add_argument('--print-freq',          default=10,            type=int, help='print frequency (default: 10)')
parser.add_argument('--lognumber',           default='test_model',  type=str, help='text id for saving logs')
parser.add_argument('--tensorboard',         default=False,         type=str2bool, help='Use tensorboard to for loss visualization')
parser.add_argument('--tensorboard_images',  default=False,         type=str2bool, help='Use tensorboard to see images')
parser.add_argument('--resume',              default='',            type=str, help='path to latest checkpoint (default: none)')

# ============ other params ============#
parser.add_argument('--predict',             dest='predict',       action='store_true', help='generate prediction masks')
parser.add_argument('--predict_train',       dest='predict_train', action='store_true', help='generate prediction masks')
parser.add_argument('--evaluate',            dest='evaluate',      action='store_true', help='just evaluate')

train_minib_counter = 0
valid_minib_counter = 0
best_f1_score = 0

args = parser.parse_args()
print(args)

assert args.m0 <= args.m1

# add fold number to the lognumber 
# if not (args.predict or args.predict_train):
#    args.lognumber = args.lognumber + '_fold' + str(args.fold_num)

# Set the Tensorboard logger
if args.tensorboard or args.tensorboard_images:
    if not (args.predict or args.predict_train):
        logger = Logger('./tb_logs/{}'.format(args.lognumber))
    else:
        logger = Logger('./tb_logs/{}'.format(args.lognumber + '_predictions'))


def main():
    global args, best_f1_score
    global logger
  
    # do transfer learning
    model = model_presets[args.arch][0](**model_presets[args.arch][1])
    
    # model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.optimizer.startswith('adam'):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     # Only finetunable params
                                     lr=args.lr)
    elif args.optimizer.startswith('rmsprop'):
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                        # Only finetunable params
                                        lr=args.lr)
    elif args.optimizer.startswith('sgd'):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    # Only finetunable params
                                    lr=args.lr)
    else:
        raise ValueError('Optimizer not supported')        
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_f1_score = checkpoint['best_f1_score']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])            
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            
            loaded_from_checkpoint = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        loaded_from_checkpoint = False
    
    # freeze the encoder
    print('Trainable param groups BEFORE freeze {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))     
    model.module.freeze()
    print('Encoder frozen!')
    print('Trainable param groups AFTER freeze   {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))
    
    if args.predict:
        pass
    
    elif args.evaluate:
        
        val_augs = ValAugs(mean = model.module.mean,
                           std = model.module.std) 
        val_dataset = MapDataset(transforms = val_augs,
                                 mode = 'val',
                                 target_resl = (args.img_size,args.img_size),
                                 do_energy_levels = args.do_energy_levels,
                                 do_boundaries = args.do_boundaries,
                                
                                 w0 = args.w0,
                                 sigma = args.sigma,
                                 do_remove_small_on_borders = args.do_remove_small_on_borders,
                                 do_produce_sizes_mask = args.do_produce_sizes_mask,
                                 do_produce_distances_mask = args.do_produce_distances_mask
                                )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        
        criterion = SemsegLoss(use_running_mean = args.do_running_mean,
                               bce_weight = args.bce_weight,
                               dice_weight = args.dice_weight,
                               use_weight_mask = True).cuda()
        
        hard_dice = HardDice(threshold=args.ths)        

        val_ap,val_ar = evaluate(val_loader,
                                 model,
                                 criterion,
                                 hard_dice)
    else:
        if args.do_augs:
            train_augs = TrainAugs(prob = args.aug_prob,
                                   mean = model.module.mean,
                                   std = model.module.std)
        elif args.do_more_augs:
            train_augs = TrainAugsIaa(prob = args.aug_prob,
                                      mean = model.module.mean,
                                      std = model.module.std)         
        else:
            train_augs = ValAugs(mean = model.module.mean,
                                 std = model.module.std)    

        val_augs = ValAugs(mean = model.module.mean,
                           std = model.module.std)  
            
        train_dataset = MapDataset(transforms = train_augs,
                                   mode = 'train',
                                   target_resl = (args.img_size,args.img_size),
                                   do_energy_levels = args.do_energy_levels,
                                   do_boundaries = args.do_boundaries,
                                   w0 = args.w0,
                                   sigma = args.sigma,
                                   do_remove_small_on_borders = args.do_remove_small_on_borders,
                                   do_produce_sizes_mask = args.do_produce_sizes_mask,
                                   do_produce_distances_mask = args. do_produce_distances_mask                                  
                                  )

        val_dataset = MapDataset(transforms = val_augs,
                                 mode = 'val',
                                 target_resl = (args.img_size,args.img_size),
                                 do_energy_levels = args.do_energy_levels,
                                 do_boundaries = args.do_boundaries,
                                 w0 = args.w0,
                                 sigma = args.sigma,
                                 do_remove_small_on_borders = args.do_remove_small_on_borders,
                                 do_produce_sizes_mask = args.do_produce_sizes_mask,
                                 do_produce_distances_mask = args. do_produce_distances_mask
                                )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size*2,        
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)

        criterion = SemsegLoss(use_running_mean = args.do_running_mean,
                               bce_weight = args.bce_weight,
                               dice_weight = args.dice_weight,
                               use_weight_mask = True).cuda()
        
        hard_dice = HardDice(threshold=args.ths)

        if args.m1<args.m2:
            milestones = [args.m1,args.m2]
        else:
            milestones = [args.m1]
            
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)  

        for epoch in range(args.start_epoch, args.epochs):
            
            if loaded_from_checkpoint == False:
                
                if epoch==args.m0:
                    print('Trainable param groups BEFORE UNfreeze {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))                  
                    model.module.unfreeze()
                    print('Encoder unfrozen!')
                    print('Trainable param groups AFTER UNfreeze {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))              

                    if args.optimizer.startswith('adam'):
                        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                     # Only finetunable params
                                                     lr=args.lr)
                    elif args.optimizer.startswith('rmsprop'):
                        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                                        # Only finetunable params
                                                        lr=args.lr)
                    elif args.optimizer.startswith('sgd'):
                        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                                    # Only finetunable params
                                                    lr=args.lr)
                    else:
                        raise ValueError('Optimizer not supported')

                    # we are assuming that m0 <= m1
                    
                    if args.m1<args.m2:
                        milestones = [args.m1-args.m0,args.m2-args.m0]
                    else:
                        milestones = [args.m1-args.m0]
                    
                    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)                     

                if epoch==args.m3:
                    criterion = SemsegLoss(use_running_mean = args.do_running_mean,
                                           bce_weight = args.bce_weight,
                                           dice_weight = args.dice_weight * 10.0,
                                           use_weight_mask = True).cuda()
            else:
                # if started from checkpoint
                # then unfreeze encoder 
                # then just tune using high dice
                # do not use lr decay anymore

                print('Trainable param groups BEFORE UNfreeze {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))                  
                model.module.unfreeze()
                print('Encoder unfrozen!')
                print('Trainable param groups AFTER UNfreeze {}'.format(len(list(filter(lambda p: p.requires_grad, model.module.parameters())))))      
                
                scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)                     

                criterion = SemsegLoss(use_running_mean = args.do_running_mean,
                                       bce_weight = args.bce_weight,
                                       dice_weight = args.dice_weight * 10.0,
                                       use_weight_mask = True).cuda()
                
                print('Current loss weights: DICE {}, BCE {}'.format(args.bce_weight,args.dice_weight * 10.0))
                
                if args.optimizer.startswith('adam'):
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                 # Only finetunable params
                                                 lr=args.lr)
                elif args.optimizer.startswith('rmsprop'):
                    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                                    # Only finetunable params
                                                    lr=args.lr)
                elif args.optimizer.startswith('sgd'):
                    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                                # Only finetunable params
                                                lr=args.lr)
                else:
                    raise ValueError('Optimizer not supported')           
                
            # adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_loss,train_bce_loss,train_dice_loss,train_hard_dice = train(train_loader,
                                                                              model,
                                                                              criterion,
                                                                              hard_dice,
                                                                              optimizer,
                                                                              epoch)

            # evaluate on validation set
            val_loss,val_bce_loss,val_dice_loss,val_hard_dice,val_ap,val_ar = validate(val_loader,
                                                                                       model,
                                                                                       criterion,
                                                                                       hard_dice)
            
            val_f1 = 2 / (1/val_ap + 1/val_ar)

            scheduler.step()

            #============ TensorBoard logging ============#
            # Log the scalar values        
            if args.tensorboard:
                info = {
                    'eph_tr_loss': train_loss,
                    'eph_tr_bce_loss': train_bce_loss,
                    'eph_tr_dice_loss': train_dice_loss,
                    'eph_tr_hard_dice': train_hard_dice,
                    
                    'eph_val_loss': val_loss,
                    'eph_val_bce_loss': val_bce_loss,
                    'eph_val_dice_loss': val_dice_loss,
                    'eph_val_hard_dice': val_hard_dice,
                    'eph_val_f1_score': val_f1,
                    'eph_val_ap': val_ap,
                    'eph_val_ar': val_ar,                      
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)                     

            # remember best prec@1 and save checkpoint
            is_best = val_f1 > best_f1_score
            best_f1_score = max(val_f1, best_f1_score)
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'best_f1_score': best_f1_score,
                },
                is_best,
                'weights/{}_checkpoint.pth.tar'.format(str(args.lognumber)),
                'weights/{}_best.pth.tar'.format(str(args.lognumber))
            )
   
def train(train_loader,
          model,
          criterion,
          hard_dice,
          optimizer,
          epoch):
                                            
    global train_minib_counter
    global logger
        
    # scheduler.batch_step()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    bce_losses = AverageMeter()
    dice_losses = AverageMeter()
    
    hard_dices = AverageMeter()


    # switch to train mode
    model.train()


    end = time.time()
    
    for i, (input, target, weight, img_ids) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        weight = weight.float().cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        weight_var = torch.autograd.Variable(weight)

        # compute output
        output = model(input_var)
                                            
        loss,bce_loss,dice_loss = criterion(output, target_var, weight_var)
        hard_dice_ = hard_dice(output, target_var)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        bce_losses.update(bce_loss.data[0], input.size(0))
        dice_losses.update(dice_loss.data[0], input.size(0))
        hard_dices.update(hard_dice_.data[0], input.size(0))        
        
        # log the current lr
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                                            
        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'train_loss': losses.val,
                'train_bce_loss': bce_losses.val,
                'train_dice_loss': dice_losses.val,
                'train_hard_dice': hard_dices.val,
                'train_lr': current_lr,      
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, train_minib_counter)                

        train_minib_counter += 1

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'BCE {bce_loss.val:.4f} ({bce_loss.avg:.4f})\t'
                  'DICE {dice_loss.val:.4f} ({dice_loss.avg:.4f})\t'
                  'HDICE    {hard_dice.val:.4f} ({hard_dice.avg:.4f})\t'.format(
                   epoch,i, len(train_loader),
                   batch_time=batch_time,data_time=data_time,
                   loss=losses,bce_loss=bce_losses,dice_loss=dice_losses,
                   hard_dice=hard_dices))
            
        # break out of cycle early if required
        # must be used with Dataloader shuffle = True
        if args.epoch_fraction < 1.0:
            if i > len(train_loader) * args.epoch_fraction:
                print('Proceed to next epoch on {}/{}'.format(i,len(train_loader)))
                break

    print(' * Avg Train Loss  {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg Train HDICE {hard_dice.avg:.4f}'.format(hard_dice=hard_dices))
            
    return losses.avg,bce_losses.avg,dice_losses.avg,hard_dices.avg

def validate(val_loader,
             model,
             criterion,
             hard_dice,
             ):
                                
    global valid_minib_counter
    global logger
    
    # scheduler.batch_step()    
    
    batch_time = AverageMeter()

    losses = AverageMeter()
    bce_losses = AverageMeter()
    dice_losses = AverageMeter()
    
    hard_dices = AverageMeter()    
    
    ap_scores = AverageMeter()
    ar_scores = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    # sigmoid for f1 calculation and illustrations
    m = nn.Sigmoid()      
    
    end = time.time()
    for i, (input, target, weight, img_ids) in enumerate(val_loader):
        
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        weight = weight.float().cuda(async=True)
        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        weight_var = torch.autograd.Variable(weight)
        
        visualize_condition = (i % int(args.print_freq * 2 * args.epoch_fraction + 1) == 0)
       
        # compute output
        output = model(input_var)
                                            
        loss,bce_loss,dice_loss = criterion(output, target_var, weight_var)
        hard_dice_ = hard_dice(output, target_var)
        
        # go over all of the predictions
        # apply the transformation to each mask
        # calculate score for each of the images
        
        averaged_aps_wt = []
        averaged_ars_wt = []
        y_preds_wt = []
            
        for j,pred_output in enumerate(output):
            pred_mask = m(pred_output[0,:,:]).data.cpu().numpy()

            # pred_mask = cv2.resize(pred_mask, (or_h,or_w), interpolation=cv2.INTER_LINEAR)
            # pred_energy = (pred_mask+pred_mask1+pred_mask2+pred_mask3+pred_mask0)/5*255
            
            pred_mask_255 = np.copy(pred_mask) * 255            
            # !!!
            # for baseline - assume that in the ground-truths buildings are not touching
            # otherwise - add additional output to the generator
            gt_labels = wt_baseline(target[j,0,:,:].cpu().numpy()*255,args.ths)
            num_buildings = gt_labels.max()
            gt_masks = []

            for _ in range(1,num_buildings+1):
                gt_masks.append((gt_labels==_)*1) 
                
            if num_buildings==0:
                y_pred_wt = wt_baseline(pred_mask_255, args.ths)
                
                if y_pred_wt.max()==0:
                    averaged_aps_wt.append(1)
                    averaged_ars_wt.append(1)
                else:
                    averaged_aps_wt.append(0)
                    averaged_ars_wt.append(0)                   
            else:
                # simple wt
                y_pred_wt = wt_baseline(pred_mask_255, args.ths)
            
                __ = calculate_ap(y_pred_wt, np.asarray(gt_masks))

                averaged_aps_wt.append(__[1])
                averaged_ars_wt.append(__[3])

            # apply colormap for easier tracking
            
            if visualize_condition:
                y_pred_wt = cv2.applyColorMap((y_pred_wt / y_pred_wt.max() * 255).astype('uint8'), cv2.COLORMAP_JET) 
                y_preds_wt.append(y_pred_wt)

            # print('MAP for sample {} is {}'.format(img_sample[j],m_ap))
            
        if visualize_condition:  
            y_preds_wt = np.asarray(y_preds_wt)
        
        averaged_aps_wt = np.asarray(averaged_aps_wt).mean()
        averaged_ars_wt = np.asarray(averaged_ars_wt).mean()

        #============ TensorBoard logging ============#                                            
        if args.tensorboard_images:
            # if i == 0:
            if visualize_condition:            
                if target.size(1)==1:
                    info = {
                        'images': to_np(input[:4,:,:,:]),
                        'loss_weights': to_np(weight[:4,:,:]),
                        'gt_mask': to_np(target[:4,0,:,:]),
                        'pred_mask': to_np(m(output.data[:4,0,:,:])),
                        'pred_wt': y_preds_wt[:4,:,:],
                    }
                    for tag, images in info.items():
                        logger.image_summary(tag, images, valid_minib_counter)


        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        bce_losses.update(bce_loss.data[0], input.size(0))
        dice_losses.update(dice_loss.data[0], input.size(0))
        hard_dices.update(hard_dice_.data[0], input.size(0))
        ap_scores.update(averaged_aps_wt, input.size(0))
        ar_scores.update(averaged_ars_wt, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'val_loss': losses.val,
                'val_bce_loss': bce_losses.val,
                'val_dice_loss': dice_losses.val,
                'val_hard_dice': hard_dices.val,
                'val_ap_score': ap_scores.val,
                'val_ar_score': ar_scores.val,

            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time  {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss  {loss.val:.4f} ({loss.avg:.4f})\t'
                  'HDICE {hard_dices.val:.4f} ({hard_dices.avg:.4f})\t'
                  'AP    {ap_scores.val:.4f} ({ap_scores.avg:.4f})\t'
                  'AR    {ar_scores.val:.4f} ({ar_scores.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time,
                      loss=losses,hard_dices=hard_dices,
                      ap_scores=ap_scores,ar_scores=ar_scores))
            
        # break out of cycle early if required
        # must be used with Dataloader shuffle = True
        if args.epoch_fraction < 1.0:
            if i > len(val_loader) * args.epoch_fraction:
                print('Proceed to next epoch on {}/{}'.format(i,len(val_loader)))
                break
                
    print(' * Avg Val  Loss  {loss.avg:.4f}'.format(loss=losses))
    print(' * Avg Val  HDICE {hard_dices.avg:.4f}'.format(hard_dices=hard_dices))
    print(' * Avg Val  AP    {ap_scores.avg:.4f}'.format(ap_scores=ap_scores)) 
    print(' * Avg Val  AR    {ar_scores.avg:.4f}'.format(ar_scores=ar_scores)) 

    return losses.avg, bce_losses.avg, dice_losses.avg,hard_dices.avg,ap_scores.avg,ar_scores.avg

def evaluate(val_loader,
              model,
              criterion,
              hard_dice, 
              save_freq = 100 
             ):
    '''
    Validation function with some visualizations and different implementation of target metric
    
    Keyword arguments:
    val_loader -- dataset loader for validation
    model -- model to validate
    criterion -- criterion object
    hard_dice -- hard dice object
    save_freq -- frequency of saving visualized results
    '''
                                
    global valid_minib_counter
    global logger
    
    batch_time = AverageMeter()
    
    ap_scores = AverageMeter()
    ar_scores = AverageMeter()
    
    ap_scores2 = AverageMeter()
    ar_scores2 = AverageMeter()
    
    # switch to evaluate mode
    model.eval() 
    
    m = nn.Sigmoid() 
    
    end = time.time()
    
    prere = pd.DataFrame(columns = ['img_id', 'img_saved_id', 'pre', 're', 'TPs', 'FPs', 'FNs', 'Pre_ave', 'Re_ave'])
    
    for i, (input, target, weight, img_ids) in enumerate(val_loader):
        
        input = input.float().cuda(async=True)
        target = target.float().cuda(async=True)
        weight = weight.float().cuda(async=True)
        
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        weight_var = torch.autograd.Variable(weight)
        
        visualize_condition = (i % int(save_freq) == 0)
       
        # compute output
        output = model(input_var)
        
        averaged_aps_wt = []
        averaged_ars_wt = []
        averaged_aps_wt2 = []
        averaged_ars_wt2 = []
        y_preds_wt = []
            
        for j,pred_output in enumerate(output):
            pred_mask = m(pred_output[0,:,:]).data.cpu().numpy()
            
            pred_mask_255 = np.copy(pred_mask) * 255 #?        
            
            pre, re, TPs, FPs, FNs = mix_vis_masks(gtmask = target[j,0,:,:].cpu().numpy()*255, 
                  genmask = pred_mask_255, 
                  orig = input[j,0,:,:].cpu().numpy()*255, 
                  lbl_treshold = args.ths, 
                  save_path = 'saved_imgs', 
                  save_title = '{}_{}_{}.png'.format(img_ids[j], i, j), 
                  do_vis = visualize_condition,
                  do_save = visualize_condition)
            
            if visualize_condition:
                img_saved_id = '{}_{}_{}.png'.format(img_ids[j], i, j)
            else:
                img_saved_id = np.nan
            
            averaged_aps_wt2.append(pre)
            averaged_ars_wt2.append(re)
            # print('pre: {}, re: {}'.format(pre, re))

            # !!!
            # for baseline - assume that in the ground-truths buildings are not touching
            # otherwise - add additional output to the generator
            gt_labels = wt_baseline(target[j,0,:,:].cpu().numpy()*255,args.ths)
            num_buildings = gt_labels.max()
            gt_masks = []

            for _ in range(1,num_buildings+1):
                gt_masks.append((gt_labels==_)*1) 
            
            aps = 0
            ars = 0
            
            if num_buildings==0:
                y_pred_wt = wt_baseline(pred_mask_255, args.ths)
                
                if y_pred_wt.max()==0:
                    aps = 1
                    ars = 1
                else:
                    aps = 0
                    ars = 0                   
            else:
                # simple wt
                y_pred_wt = wt_baseline(pred_mask_255, args.ths)
            
                __ = calculate_ap(y_pred_wt, np.asarray(gt_masks))
                aps = __[1]
                ars = __[3]

            # apply colormap for easier tracking
            averaged_aps_wt.append(aps)
            averaged_ars_wt.append(ars)
            
            prere_samp = pd.DataFrame(columns = ['img_id', 'img_saved_id', 'pre', 're', 'TPs', 'FPs', 'FNs', 'Pre_ave', 'Re_ave'])
            prere_samp.loc[0] = [img_ids[j], img_saved_id, pre, re, TPs, FPs, FNs, aps, ars]
            prere = pd.concat([prere, prere_samp], ignore_index = True)
            if visualize_condition:
                y_pred_wt = cv2.applyColorMap((y_pred_wt / y_pred_wt.max() * 255).astype('uint8'), cv2.COLORMAP_JET) 
                y_preds_wt.append(y_pred_wt)
                
        # measure accuracy
        
        averaged_aps_wt = np.asarray(averaged_aps_wt).mean()
        averaged_ars_wt = np.asarray(averaged_ars_wt).mean()
        
        averaged_aps_wt2 = np.asarray(averaged_aps_wt2).mean()
        averaged_ars_wt2 = np.asarray(averaged_ars_wt2).mean()
        
        ap_scores.update(averaged_aps_wt, input.size(0))
        ar_scores.update(averaged_ars_wt, input.size(0))
        
        ap_scores2.update(averaged_aps_wt2, input.size(0))
        ar_scores2.update(averaged_ars_wt2, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #============ TensorBoard logging ============#
        # Log the scalar values        
        if args.tensorboard:
            info = {
                'val_ap_score': ap_scores.val,
                'val_ar_score': ar_scores.val,
                'val_ap_score2': ap_scores2.val,
                'val_ar_score2': ar_scores2.val,

            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, valid_minib_counter)            
        
        valid_minib_counter += 1
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time  {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'AP    {ap_scores.val:.4f} ({ap_scores.avg:.4f})\t'
                  'AR    {ar_scores.val:.4f} ({ar_scores.avg:.4f})\t'
                  'AP2    {ap_scores2.val:.4f} ({ap_scores2.avg:.4f})\t'
                  'AR2    {ar_scores2.val:.4f} ({ar_scores2.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time,
                      ap_scores=ap_scores,ar_scores=ar_scores,
                      ap_scores2=ap_scores2,ar_scores2=ar_scores2))
            
        # break out of cycle early if required
        # must be used with Dataloader shuffle = True
        if args.epoch_fraction < 1.0:
            if i > len(val_loader) * args.epoch_fraction:
                print('Proceed to next epoch on {}/{}'.format(i,len(val_loader)))
                break

    print(' * Avg Val  AP    {ap_scores.avg:.4f}'.format(ap_scores=ap_scores))
    print(' * Avg Val  AR    {ar_scores.avg:.4f}'.format(ar_scores=ar_scores))
    print(' * Avg Val  AP2    {ap_scores2.avg:.4f}'.format(ap_scores2=ap_scores2))
    print(' * Avg Val  AR2    {ar_scores2.avg:.4f}'.format(ar_scores2=ar_scores2))
    
    prere.to_csv('prere.csv', index = False)
    
    return pre, re

def predict(val_loader, model):
    pass

def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 50 epochs"""
    lr = args.lr * (0.9 ** ( (epoch+1) // 50))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def measure_hard_mse(output,target,ths):
    _ = torch.abs(output - target)
    _ = (_ < ths) * 1
    items = _.shape[0] * _.shape[1]
    
    return float(_.sum() / items)
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()