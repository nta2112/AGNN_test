import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets import image_folder_custom # Register custom loader
from datasets.samplers import CategoriesSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def adjust_learning_rate(optimizers, lr, iter):
    new_lr = lr * (0.1**(int(iter//5000)))
    for param_group in optimizers.param_groups:
        param_group['lr'] = new_lr

def main(config, args):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
                config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    
    if args.save_path:
        save_path = args.save_path
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = os.path.join('./save', svname)
        if args.resume is None:
            utils.ensure_path(save_path)
        else:
            utils.ensure_path(save_path, remove=False)

    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    train_sampler = CategoriesSampler(
            train_dataset.label, config['train_batches'],
            n_train_way, n_train_shot + n_query,
            ep_per_batch=ep_per_batch)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_sampler = CategoriesSampler(
                tval_dataset.label, config['test_batches'],
                n_way, n_shot + n_query,
                ep_per_batch=4)
        tval_loader = DataLoader(tval_dataset, batch_sampler=tval_sampler,
                                 num_workers=8, pin_memory=True)
    else:
        tval_loader = None

    # val
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, config['test_batches'],
            n_way, n_shot + n_query,
            ep_per_batch=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        # Anti-gravity: inject n_way into model_args to support dynamic GNN size
        config['model_args']['n_way'] = config['n_way']
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            # Modified to load on CPU first to allow cross-device loading
            encoder_path = config['load_encoder']
            utils.log(f"Loading encoder from {encoder_path}")
            encoder_checkpoint = torch.load(encoder_path, map_location='cpu')
            
            # Determine if it's a full checkpoint or just model state
            if 'model_sd' in encoder_checkpoint:
                state_dict = encoder_checkpoint['model_sd']
            else:
                state_dict = encoder_checkpoint
                
            # If the checkpoint contains full classifier model, we need to extract 'encoder' part
            # But here models.load expects the full object logic often.
            try:
                # Let's try to mimic original behavior but robustly
                loaded_object = torch.load(encoder_path, map_location='cpu')
                if isinstance(loaded_object, dict) and 'model_sd' in loaded_object:
                     # It's our Training checkpoint
                     # Helper to load model from checkpoint dict
                     temp_model = models.make(loaded_object['model'], **loaded_object['model_args'])
                     temp_model.load_state_dict(loaded_object['model_sd'])
                     encoder = temp_model.encoder
                else:
                     # It might be a direct model dump?
                     temp_model = models.load(loaded_object)
                     encoder = temp_model.encoder
                
                model.encoder.load_state_dict(encoder.state_dict())
                utils.log("Encoder loaded successfully.")
            except Exception as e:
                utils.log(f"Error loading encoder: {e}")
                raise e

    if args.freeze_encoder:
        utils.log("Freezing encoder weights.")
        for param in model.encoder.parameters():
            param.requires_grad = False

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['tl', 'ta', 'tvl', 'tva', 'vl', 'va']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []
    
    start_epoch = 1
    if args.resume is not None:
        if os.path.isfile(args.resume):
            utils.log(f"=> loading checkpoint '{args.resume}'")
            # Load checkpoint
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Check for nested 'training' dictionary (how this script saves it)
            if 'training' in checkpoint:
                start_epoch = checkpoint['training']['epoch'] + 1
                optimizer_state = checkpoint['training'].get('optimizer_sd', checkpoint['training'].get('optimizer'))
            else:
                # Fallback for potential flat format
                start_epoch = checkpoint.get('epoch', 0) + 1
                optimizer_state = checkpoint.get('optimizer_sd', checkpoint.get('optimizer'))

            # Load model state
            if config.get('_parallel'):
                model.module.load_state_dict(checkpoint['model_sd'])
            else:
                model.load_state_dict(checkpoint['model_sd'])
            
            # Load optimizer state
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                 
            utils.log(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch - 1})")
        else:
            utils.log(f"=> no checkpoint found at '{args.resume}'")

    for epoch in range(start_epoch, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        np.random.seed(epoch)
        for data, _ in tqdm(train_loader, desc='train', leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_train_way, n_train_shot, n_query,
                    ep_per_batch=ep_per_batch)
            label = fs.make_nk_label(n_train_way, n_query,
                    ep_per_batch=ep_per_batch).cuda()
            
            optimizer.zero_grad()
            
            # Anti-gravity: GNN model requires support labels (tr_label)
            # Create support labels: (Batch, N_way * N_shot)
            tr_label = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch).cuda()
            
            logits = model(x_shot, x_query, tr_label)
            logits = logits.view(-1, n_train_way)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            total_loss = loss
            total_loss.backward()
            optimizer.step()
            
            aves['tl'].add(total_loss.sum().item())
            aves['ta'].add(acc)

            logits = None; total_loss = None ;loss = None
            
        # eval
        model.eval()

        for name, loader, name_l, name_a in [
                ('tval', tval_loader, 'tvl', 'tva'),
                ('val', val_loader, 'vl', 'va')]:

            if (config.get('tval_dataset') is None) and name == 'tval':
                continue
            np.random.seed(0)
            for data, _ in tqdm(loader, desc=name, leave=False):
                # Anti-gravity: Detect actual ep derived from batch size
                # Because tval_sampler might be hardcoded to 4
                current_batch_size = data.shape[0]
                n_items_per_ep = n_way * (n_shot + n_query)
                real_ep_per_batch = current_batch_size // n_items_per_ep
                
                x_shot, x_query = fs.split_shot_query(
                        data.cuda(), n_way, n_shot, n_query,
                        ep_per_batch=real_ep_per_batch) 

                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=real_ep_per_batch).cuda() 

                with torch.no_grad():
                    # Create support labels for validation
                    tr_label = fs.make_nk_label(n_way, n_shot, ep_per_batch=real_ep_per_batch).cuda()
                    logits = model(x_shot, x_query, tr_label)
                    
                    logits = logits.view(-1, n_way)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                    total_loss = loss

                aves[name_l].add(total_loss.sum().item())
                aves[name_a].add(acc)

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        utils.log('epoch {}, train {:.4f}|{:.4f}, tval {:.4f}|{:.4f}, '
                'val {:.4f}|{:.4f}, {} {}/{} (@{})'.format(
                epoch, aves['tl'], aves['ta'], aves['tvl'], aves['tva'],
                aves['vl'], aves['va'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {
            'train': aves['tl'],
            'tval': aves['tvl'],
            'val': aves['vl'],
        }, epoch)
        writer.add_scalars('acc', {
            'train': aves['ta'],
            'tval': aves['tva'],
            'val': aves['va'],
        }, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

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
            'model_sd': model_.state_dict(),
            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_meta_tlu.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--lamb', type=float, default=0.03)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-path', default=None, help='custom path to save checkpoints')
    parser.add_argument('--resume', default=None, help='path to checkpoint to resume from')
    parser.add_argument('--freeze-encoder', action='store_true', help='freeze encoder weights during training')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config, args)
