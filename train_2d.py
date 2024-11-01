#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import json
import os.path as osp
import random
import numpy as np
import pandas as pd
import torch
from PCGrad.pcgrad import PCGrad

from src.dataset import PointCloud2D
from src.model import SIREN
from src.util import create_output_paths, load_experiment_parameters
from src.loss_functions import loss_siren, loss_medial_axis
from generate_df import generate_df_2D
import time

import wandb

def evaluate_loss( input_data, normals, sdfs, optim, loss_fn, model, config, device, running_loss, pc=False ):
    # zero the parameter gradients
    optim.zero_grad()
    
    # forward + backward + optimize
    #input_data = input_data.to( device )
    #normals = normals.to(device)
    #sdfs = sdfs.to(device)
    
    loss = loss_fn( 
        model, 
        input_data, 
        {'normals': normals, 'sdfs': sdfs},
        config['loss_weights']  
    )

    wandb.log( { 
        name_term : loss_term 
        for name_term, loss_term in loss.items()
    } )

    train_loss = torch.zeros((1, 1), device=device)
    for it, l in loss.items() :
        train_loss += l
        # accumulating statistics per loss term
        if it not in running_loss:
            running_loss[it] = l
        else:
            running_loss[it] += l

    if pc:
        return loss
    else:
        return train_loss
    
def train_model_siren( dataset, model, device, config) -> torch.nn.Module:
    epochs = config["epochs"]
    epochs_til_checkpoint = config.get("epochs_to_checkpoint", 0)

    log_path = config["log_path"]
    optim = config["optimizer"]

    model.to(device)

    losses = dict()
    best_loss = np.inf
    best_weights = None

    loss_fn = loss_medial_axis
    current_lr = config['warmup_lr']

    if config["optimizer_type"][:2] == "pc":
        for g in optim.optimizer.param_groups:
            g['lr'] = current_lr
    elif config["optimizer_type"] != 'lbfgs':
        for g in optim.param_groups:
            g['lr'] = current_lr

    recon_time = 0
    start_ttime = time.time() 
    for epoch in range(epochs):
        if epoch == config['warmup_epochs']:
            current_lr = config['lr']
            if config["optimizer_type"][:2] == "pc":
                for g in optim.optimizer.param_groups:
                    g['lr'] = current_lr
            elif config["optimizer_type"] != 'lbfgs':
                for g in optim.param_groups:
                    g['lr'] = current_lr

        running_loss = dict()
        for input_data, normals, sdfs in iter(dataset):
            
            if config["optimizer_type"][:2] == "pc":
                loss = evaluate_loss( 
                    input_data, normals, sdfs,
                    optim,
                    loss_fn,
                    model,
                    config,
                    device,
                    running_loss,
                    pc=True
                )
                optim.pc_backward( list( loss.values() ) )
            elif config["optimizer_type"] == 'lbfgs':
                optim.step( lambda : evaluate_loss( input_data, normals, sdfs, optim, loss_fn, model, config, device, running_loss ) )
            
            else:
                train_loss = evaluate_loss( 
                    input_data, normals, sdfs,
                    optim,
                    loss_fn,
                    model,
                    config,
                    device,
                    running_loss
                )
                train_loss.backward( )

                optim.step()

        # accumulate statistics
        for it, l in running_loss.items():
            if it in losses:
                losses[it][epoch] = l.detach().cpu().numpy()
            else:
                losses[it] = [0.] * epochs
                losses[it][epoch] = l.detach().cpu().numpy()

        epoch_loss = 0
        for k, v in running_loss.items():
            epoch_loss += v
        epoch_loss /=+ dataset.batchesPerEpoch
        print(f"Epoch: {epoch} - Loss: {epoch_loss} - Learning Rate: {current_lr:.3e}")


        start_rtime = time.time()
        # Saving the best model after warmup.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                best_weights,
                osp.join(log_path, "models", "model_best.pth")
            )

        end_rtime = time.time()
        recon_time += end_rtime - start_rtime

    end_ttime = time.time()
    total_training_time = end_ttime - start_ttime - recon_time

    return losses, best_weights, total_training_time

def setup_train( parameter_dict, cuda_device ):

    if not torch.cuda.is_available():
        print('Utilizing CPU')
        
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    seed = 123 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    full_path = create_output_paths(
        parameter_dict["checkpoint_path"],
        parameter_dict["experiment_name"],
        overwrite=False
    )

    # Saving the parameters to the output path
    with open(osp.join(full_path, "params.json"), "w+") as fout:
        json.dump(parameter_dict, fout, indent=4)

    dataset = PointCloud2D(
        pointCloudPath= parameter_dict["dataset"],
        batchSize= parameter_dict["batch_size"],
        samplingPercentiles=parameter_dict["sampling_percentiles"],
        batchesPerEpoch = parameter_dict["batches_per_epoch"],
        device=device
    )

    network_params = parameter_dict["network"]
    model = SIREN(
        n_in_features= 2,
        n_out_features=1,
        hidden_layer_config=network_params["hidden_layer_nodes"],
        w0=network_params["w0"],
        ww=network_params.get("ww", None),
        activation= network_params.get('activation', 'sine')
    )
    print(model)

    if network_params['pretrained_dict'] != 'None':
        model.load_state_dict(torch.load(network_params['pretrained_dict'], map_location=device))

    opt_params = parameter_dict["optimizer"]

    if opt_params["type"] == "adam":
        
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )
    elif opt_params["type"] == "sgd":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )
    elif opt_params["type"] == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params=model.parameters()
        )
    elif opt_params["type"] == "pc_adam":
        optimizer = PCGrad( torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        ) )
    elif opt_params["type"] == "pc_sgd":
        optimizer = PCGrad( torch.optim.SGD(
            lr=opt_params["lr"],
            params=model.parameters()
        ) )
    else:
        raise ValueError('Unknown optimizer')
    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "log_path": full_path,
        "optimizer": optimizer,
        "optimizer_type": opt_params["type"],
        "warmup_epochs": parameter_dict.get('warmup_epochs',0),
        "warmup_lr": parameter_dict.get('warmup_lr', 1e-4),
        "lr": opt_params["lr"],
        "loss_weights": parameter_dict["loss_weights"],
        "resolution": parameter_dict.get('resolution', 256)
    }
    losses, best_weights, training_time = train_model_siren(
        dataset,
        model,
        device,
        config_dict,
    )
        
    loss_df = pd.DataFrame.from_dict(losses)
    loss_df.to_csv(osp.join(full_path, "losses.csv"), sep=";", index=None)

    # saving the final model.
    torch.save(
        model.state_dict(),
        osp.join(full_path, "models", "model_final.pth")
    )

    print('Generating distance field slices')
    df_options = {
        'device': cuda_device,
        'surf_thresh': 1e-3,
        'width': 512,
        'weight0': network_params["w0"],
        'hidden_layer_nodes': network_params["hidden_layer_nodes"],
        'activation': network_params.get('activation', 'sine')
    }

    generate_df_2D( osp.join(full_path, "models", "model_best.pth"), osp.join(full_path, "reconstructions/"), df_options)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        usage="python main.py path_to_experiments.json cuda_device"
    )

    p.add_argument(
        "experiment_path", type=str,
        help="Path to the JSON experiment description file"
    )
    p.add_argument(
        "device", type=int, help="Cuda device"
    )
    args = p.parse_args()
    parameter_dict = load_experiment_parameters(args.experiment_path)
    if not bool(parameter_dict):
        raise ValueError("JSON experiment not found")
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="SIREN",

        # track hyperparameters and run metadata
        config={
            "learning_rate": parameter_dict['optimizer']['lr'],
            "epochs": parameter_dict['num_epochs'],
            "loss_weights": parameter_dict['loss_weights'],
            "3d-model": parameter_dict['dataset']
        }
    )
    
    setup_train( parameter_dict, args.device )


