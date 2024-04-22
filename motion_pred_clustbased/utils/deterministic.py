import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime 


def create_dirs_anet(cfg, inputs_type):
    """ Create all necessary folders to store results and outputs for auxiliary network
    Args:
        cfg (dict): config file
        inputs (str): inputs type
    Returns:
        [type]: path for the tensorboard runs, path for the outputs
    """
    ts = str(datetime.datetime.now().strftime('%y-%m-%d_%a_%H:%M:%S:%f')) + str(os.getpid())
    ds = cfg["data"]["dataset"]
    test_name = cfg['data']['dataset_target'] # test dataset
    anet_cfg = cfg['anet']['network']
    model_type = anet_cfg['net_type'] 
    mode = "_" + anet_cfg['mode']
    nlayers = "_" + str(anet_cfg['n_layers'])
    check_dir = cfg["save"]["path"]
    runs_path = os.path.join(check_dir, "runs", ds, test_name, "anet_" + model_type + mode + nlayers, inputs_type)
    output_path = os.path.join(check_dir, "outputs", ds, test_name, "anet_" + model_type + mode + nlayers, inputs_type)
    output_path = os.path.join(output_path, ts)
    runs_path = os.path.join(runs_path, ts)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    return runs_path, output_path



def create_dirs(cfg, inputs_type):
    """ Create all necessary folders to store results and outputs for lstm and gru
    Args:
        cfg (dict): config file
        inputs (str): inputs type
    Returns:
        [type]: path for the tensorboard runs, path for the outputs
    """
    ts = str(datetime.datetime.now().strftime('%y-%m-%d_%a_%H:%M:%S:%f')) + str(os.getpid())
    test_name = cfg["data"]["dataset_target"] # test dataset
    model_type = cfg["model"] if cfg["model"] == "gru" else cfg["model"] + f"_{cfg['network']['state']}"
    check_dir = cfg["save"]["path"] # dir to save checkpoints, results
    ds = cfg["data"]["dataset"]
    runs_path = os.path.join(check_dir, "runs", ds, test_name, model_type, inputs_type)
    output_path =  os.path.join(check_dir, "outputs", ds, test_name, model_type, inputs_type)
    output_path = os.path.join(output_path, ts)
    runs_path = os.path.join(runs_path, ts)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    return runs_path, output_path


def det_loss(out_model : torch.Tensor, x : torch.Tensor, target : torch.Tensor, pred_len : int, cs) -> torch.Tensor:
    """ Computes vanilla determninistic loss function
    Args:
        out_model (torch.Tensor): predictor's output
        x (torch.Tensor): input
        target (torch.Tensor): _gt
        pred_len (int): _des
        cs (_type_): coordinates scaler 
    Returns:
        torch.Tensor: _description_
    """
    out = out_model.view(out_model.shape[0], pred_len, -1).clone()
    out_last = cs.denorm_increment(x, out.clone())     
    loss = F.mse_loss(out_last, target, reduction = "none").mean(dim = -1).mean(dim = 1)
    return loss


def save_objects_det(net_type : str, model : nn.Module, test_ep : int, save_dir : str):
    """ Save results
    Args:
        results (tuple): _ade, fde 
        model (nn.Module): deterministic network
        epoch (int): _number of epoch 
        save_dir (str): path to save the results
    """
    torch.save(model.state_dict(), os.path.join(save_dir, f"{net_type}_ep{test_ep}.pth"))
        

def save_checkpoints_det(epoch : int, save_dir : str, test_dsname : str, model_name : str, model : nn.Module, optimizer, lr_sched):
    """Save checkpoints -> model, optimizer, etc.
    Args:
        epoch (int): epoch number
        save_dir (str) : checkpoint dir  
        test_dsname (str): data set name
        model_name (str): model name 
        model (nn.Module): model and its respective weights
        optimizer (_type_): optimizer state
        lr_sched (_type_): learning rate scheduler 
    """
    path = os.path.join(save_dir,  f"checkpoint_{test_dsname}_{model_name}_ep{epoch}.pth")
    checkpoint = { 
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched}
    torch.save(checkpoint, path)