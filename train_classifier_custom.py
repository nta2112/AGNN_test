import argparse
import os
import yaml
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

import datasets
import models
import utils
from datasets.image_folder import ImageFolder
from datasets import image_folder_custom # Register custom dataset
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
        # If using custom path, likely on Drive, do NOT remove it blindly
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join('./save', svname)
        # Only use ensure_path (which prompts delete) if NOT resuming
        if args.resume is None:
            utils.ensure_path(save_path)
        else:
            utils.ensure_path(save_path, remove=False)

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
    
    # Anti-gravity: Fix for class imbalance
    # Compute the weight for each sample
    utils.log("analyzing dataset for class imbalance...")
    targets = np.array(train_dataset.label) # Convert to numpy for faster indexing
    # We need to map class indices to counts. 
    # Note: targets contains class indices from 0 to n_classes-1
    class_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_count
    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False,
                              sampler=sampler, num_workers=8, pin_memory=True)
    
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

    timer_epoch = utils.Timer()

    start_epoch = 1
    if args.resume is not None:
        if os.path.isfile(args.resume):
            utils.log(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu') # Map to CPU first to avoid GPU mapping issues if different
            if 'training' in checkpoint:
                start_epoch = checkpoint['training']['epoch'] + 1
                optimizer_state = checkpoint['training'].get('optimizer_sd', checkpoint['training'].get('optimizer'))
            else:
                start_epoch = checkpoint.get('epoch', 0) + 1
                optimizer_state = checkpoint.get('optimizer')

            # Load model state
            # Check if model is DataParallel but checkpoint is not (handled by save logic usually, but let's be safe)
            if config.get('_parallel'):
                model.module.load_state_dict(checkpoint['model_sd'])
            else:
                model.load_state_dict(checkpoint['model_sd'])
                
            # Load optimizer state
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            
            # Load scheduler if exists and if saved (Standard code didn't save scheduler state explicitly in 'optimizer' key, 
             # usually it's separate. The save_obj in line 152 doesn't seem to save scheduler state explicitly 
             # but let's check. Ah, it saves 'optimizer' args but not scheduler object state.
             # Ideally we should adding scheduler state load if it's stateful. 
             # Resume logic without scheduler state is acceptable for SGD with StepLR usually if we just recalculated,
             # but let's just do model/optimizer for now as that's the massive part).
            
            utils.log(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            utils.log(f"=> no checkpoint found at '{args.resume}'")

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
        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,
            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model.state_dict() if not config.get('_parallel') else model.module.state_dict(),
            'training': training,
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
    parser.add_argument('--resume', default=None, help='path to checkpoint to resume from')
    parser.add_argument('--save-path', default=None, help='custom path to save checkpoints and logs')
    parser.add_argument('--save-epoch', type=int, default=None, help='frequency of saving checkpoints (in epochs)')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if args.save_epoch is not None:
        config['save_epoch'] = args.save_epoch

    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)
