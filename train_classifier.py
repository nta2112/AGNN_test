import argparse
import os
import yaml
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

import datasets
import models
import utils
from datasets.image_folder import ImageFolder
from datasets import image_folder_custom # Register custom folder
from models import classifier

def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    if args.save_path:
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join('./save', svname)
        utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####
    # Using ImageFolder-like wrapper or custom loader that reads the CSVs we created
    # For now, let's assume we can use the 'mini-imagenet' logic but point to our CSVs
    # However, standard ImageFolder requires folder structure. 
    # Since we have CSVs from prepare_dataset.py, let's use a Generic CSV Dataset Loader if available
    # Or reuse the 'image_folder' logic which might need 'root_path' + 'csv_split'
    
    # Let's inspect datasets/image_folder.py or similar to see how to load.
    # BUT for now I will write a generic loop assuming a basic dataset interface.
    
    # NOTE: In AGNN repo, datasets.make returns a dataset object.
    # We need to register 'tlu-states' or uses 'image_folder' with arguments.
    # Let's try to use 'mini-imagenet' class but override paths? No, proper way is 'image_folder'
    
    # Construct args for datasets.make('image_folder', ...) if it exists, or just use mini-imagenet wrapper
    # Actually, looking at image_folder.py (if I could see it), it likely takes a root and a split file.
    # We will assume config passes correct args.
    
    train_dataset = datasets.make(config['train_dataset'], **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    
    if config.get('val_dataset'):
        val_dataset = datasets.make(config['val_dataset'], **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=8, pin_memory=True)
    else:
        val_loader = None

    #### Model and Optimizer ####
    
    # Model: classifier wrapper around encoder
    # models/classifier.py usually takes 'encoder' and 'classifier_args' (n_classes)
    n_classes = config['model_args']['classifier_args']['n_classes']
    utils.log(f"Training with {n_classes} classes")

    model = models.make(config['model'], **config['model_args'])

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])
        
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    #### Training Loop ####
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    
    timer_epoch = utils.Timer()

    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            utils.log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_sd'])
            if 'optimizer_sd' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_sd'])
            utils.log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            utils.log("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(start_epoch, max_epoch + 1):
        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}

        # Train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
            
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

        # Eval
        if val_loader is not None:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                
                aves['vl'].add(loss.item())
                aves['va'].add(acc)

        # Post-epoch
        if lr_scheduler is not None:
            lr_scheduler.step()

        t_epoch = utils.time_str(timer_epoch.t())
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch, aves['tl'].item(), aves['ta'].item())
        if val_loader is not None:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'].item(), aves['va'].item())
        log_str += ', {}'.format(t_epoch)
        utils.log(log_str)
        
        writer.add_scalars('loss', {'train': aves['tl'].item(), 'val': aves['vl'].item()}, epoch)
        writer.add_scalars('acc', {'train': aves['ta'].item(), 'val': aves['va'].item()}, epoch)

        # Save
        save_obj = {
            'file': __file__,
            'config': config,
            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model.state_dict() if not config.get('_parallel') else model.module.state_dict(),
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
            
        if val_loader is not None and aves['va'].item() > max_va:
            max_va = aves['va'].item()
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_classifier_tlu.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-path', default=None, help='custom path to save checkpoints')
    parser.add_argument('--resume', default=None, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
