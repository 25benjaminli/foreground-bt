#!/usr/bin/env python

import argparse
import time
import random

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR

import matplotlib.pyplot as plt

from models.fewshot import FewShotSeg
from dataloaders.utils import get_train_loader
from utils import *
# import set determinism from monai
from monai.utils import set_determinism
import numpy as np
from tqdm import tqdm
def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    # parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_sv', type=int, required=True)
    # parser.add_argument('--fold', type=int, required=True)

    parser.add_argument('--workers', default=1, type=int)
    # parser.add_argument('--steps', default=60000, type=int) # Setting number of epochs
    # parser.add_argument('--max_iterations', default=2000, type=int) 
    # add epochs
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--n_shot', default=1, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--n_way', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)        
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_gamma', default=0.95, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--bg_wt', default=0.7, type=float)
    parser.add_argument('--t_loss_scaler', default=1.0, type=float)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Deterministic setting for reproducability.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        set_determinism(seed=args.seed)
        np.random.seed(args.seed)
        

    # Set up logging.
    logger = set_logger(args.save_dir, 'train.log')
    logger.info(args)

    # Setup the path to save.
    args.save_model_path = os.path.join(args.save_dir, 'model.pth')

    # Init model.
    model = FewShotSeg(False)
    model = nn.DataParallel(model.cuda())

    # Init optimizer.
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # milestones = [(ii + 1) * 1000 for ii in range(args.steps // 1000 - 1)]
    # scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)

    # Define loss function.
    my_weight = torch.FloatTensor([args.bg_wt, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True

    # Define data set and loader.
    train_loader = get_train_loader(args)
    # train_dataset = TrainDataset(args)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.workers, pin_memory=True, drop_last=False)
    # logger.info('  Training on images not in test fold: ' +
    #             str([elem[len(args.data_dir):] for elem in train_dataset.image_dirs]))

    # Start training.
    # args.epochs = args.steps // args.max_iterations
    logger.info('  Start training ...')

    losses_arr = []
    q_loss_arr = []
    align_loss_arr = []
    t_loss_arr = []
    X_axis = []

    for epoch in range(args.epochs):

        # Train.
        batch_time, data_time, losses, q_loss, align_loss, t_loss = train(train_loader, model, criterion, optimizer,
                                                                          args)

        # Log
        logger.info(f'============== Epoch [{epoch}/{args.epochs}] ==============')
        logger.info('  Batch time: {:6.3f}'.format(batch_time))
        logger.info('  Loading time: {:6.3f}'.format(data_time))
        logger.info('  Total Loss  : {:.5f}'.format(losses))
        logger.info('  Query Loss  : {:.5f}'.format(q_loss))
        logger.info('  Align Loss  : {:.5f}'.format(align_loss))
        logger.info('  Threshold Loss  : {:.5f}'.format(t_loss))
        # print learning rate
        print(f"learning rate {scheduler.get_last_lr()}")

        losses_arr.append(losses)
        q_loss_arr.append(q_loss)
        align_loss_arr.append(align_loss)
        t_loss_arr.append(t_loss)
        X_axis.append(epoch+1)

        scheduler.step() # update LR

        # cut it out after 3 epochs
        if epoch > 3:
            break

    # Save trained model.
    logger.info('Saving model ...')
    torch.save(model.state_dict(), args.save_model_path)

    plt.plot(X_axis, losses_arr, color = 'r', label = 'Total loss')
    plt.plot(X_axis, q_loss_arr, color = 'g', label = 'Query loss')
    plt.plot(X_axis, align_loss_arr, color = 'b', label = 'Align loss')
    plt.plot(X_axis, t_loss_arr, color = 'y', label = 'Threshold loss')   

    plt.show() 

def train(train_loader, model, criterion, optimizer, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    q_loss = AverageMeter('Query loss', ':.4f')
    a_loss = AverageMeter('Align loss', ':.4f')
    t_loss = AverageMeter('Threshold loss', ':.4f')

    # Train mode.
    model.train()  # change learning rate

    end = time.time()
    for i, sample in tqdm(enumerate(train_loader)):
        if i == 0:
            # print the id
            print("id", sample['id'])
        # Extract support and query data.
        support_images = [[shot.float().cuda() for shot in way]
                          for way in sample['support_images']]
        support_fg_mask = [[shot.float().cuda() for shot in way]
                           for way in sample['support_fg_labels']]

        query_images = [query_image.float().cuda() for query_image in sample['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

        # Log loading time.
        data_time.update(time.time() - end)

        # Compute predictions and losses
        query_pred, align_loss, thresh_loss = model(support_images, support_fg_mask, query_images,
                                                    train=True, t_loss_scaler=args.t_loss_scaler)

        query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                     1 - torch.finfo(torch.float32).eps)), query_labels)
        loss = query_loss + align_loss + thresh_loss

        # compute gradient and do SGD step
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()

        # update losses
        losses.update(loss.item(), query_pred.size(0))
        q_loss.update(query_loss.item(), query_pred.size(0))
        a_loss.update(align_loss.item(), query_pred.size(0))
        t_loss.update(thresh_loss.item(), query_pred.size(0))

        # Log elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time.avg, data_time.avg, losses.avg, q_loss.avg, a_loss.avg, t_loss.avg


if __name__ == '__main__':
    main()

