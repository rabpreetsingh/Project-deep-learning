import os
import sys
import glob
import shlex
import pprint
import random
import shutil
import signal
import os.path as osp
import datetime
import platform
import threading
import subprocess

import yaml
import numpy as np
import torch
import scipy.io as sio
from docopt import docopt
import vpd
from vpd.config import Config, ModelConfig
from vpd.datasets import NYUDataset, WireframeDataset, ScanNetDataset, YUDDataset


torch.cuda.empty_cache()

print('Clearing cache')

import gc

gc.collect()
torch.cuda.memory_summary(device=None, abbreviated=False)
print('Clearing cache 2')

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    else:
        shutil.rmtree(directory_name)
        os.makedirs(directory_name)



def generate_output_directory(identifier):
    # load configuration
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % get_git_hash()
    name += "-%s" % identifier
    out_directory = osp.join(osp.expanduser(Config.io.log_directory), name)
    if not osp.exists(out_directory):
        os.makedirs(out_directory)
    Config.io.resume_from = out_directory
    Config.to_yaml(osp.join(out_directory, "config.yaml"))

    return out_directory


def main():
    args = docopt(__doc__)
    configuration_file = args["<yaml-config>"]
    Config.update(Config.from_yaml(filename=configuration_file))
    ModelConfig.update(Config.model)
    pprint.pprint(Config, indent=4)
    resume_from = Config.io.resume_from

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    print('Torch version:', torch.__version__)
    device_name = "cuda"
    num_gpus = args["--devices"].count(",") + 1
    num_gpus = 1
    print('Number of GPUs:', num_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print('CuDNN version:', torch.backends.cudnn.version())
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
        for k in range(0, torch.cuda.device_count()):
            print('kth, device name', k, torch.cuda.get_device_name(k))
    else:
        use_gpu=False
        assert use_gpu is False, "CUDA is not available"
        print("CUDA is not available")
    device = torch.device(device_name)
    print(device)
    print("-------------------------------")
    # 1. dataset
    batch_size = ModelConfig.batch_size * num_gpus
    data_directory = Config.io.data_directory
    num_workers = Config.io.num_workers * num_gpus
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }
    
    if Config.io.dataset.upper() == "WIREFRAME":
        Dataset = WireframeDataset
    elif Config.io.dataset.upper() == "SCANNET":
        Dataset = ScanNetDataset
    elif Config.io.dataset.upper() == "NYU":
        Dataset = NYUDataset
    elif Config.io.dataset.upper() == "YUD":
        Dataset = YUDDataset
    else:
        raise NotImplementedError
    train_loader = torch.utils.data.DataLoader(
        Dataset(data_directory, split="train"), shuffle=True, **kwargs
    )
    validation_loader = torch.utils.data.DataLoader(
        Dataset(data_directory, split="valid"), shuffle=False, **kwargs
    )
    epoch_size = len(train_loader)
    print('Epoch size: train/valid',  len(train_loader), len(validation_loader))

    npzfile = np.load(Config.io.ht_mapping, allow_pickle=True)
    ht_mapping = npzfile['ht_mapping']
    ht_mapping[:,2] = npzfile['rho_res'].item() - np.abs(ht_mapping[:,2])
    ht_mapping[:,2] /= npzfile['rho_res'].item()
    voting_ht_dictionary={}
    voting_ht_dictionary["vote_mapping"]= torch.tensor(ht_mapping, requires_grad=False).float().contiguous()
    voting_ht_dictionary["im_size"]= (npzfile['rows'], npzfile['cols'])
    voting_ht_dictionary["ht_size"]= (npzfile['h'], npzfile['w'])
    print('Voting HT dictionary  memory MB', voting_ht_dictionary["vote_mapping"].size(),
          voting_ht_dictionary["vote_mapping"].element_size() * voting_ht_dictionary["vote_mapping"].nelement() / (1024 * 1024))

    npzfile = np.load(Config.io.sphere_mapping, allow_pickle=True)
    sphere_neighbors = npzfile['sphere_neighbors_weight']
    voting_sphere_dictionary={}
    voting_sphere_dictionary["vote_mapping"]=torch.tensor(sphere_neighbors, requires_grad=False).float().contiguous()
    voting_sphere_dictionary["ht_size"]=(npzfile['h'], npzfile['w'])
    voting_sphere_dictionary["sphere_size"]=npzfile['num_points']
    print('Voting sphere dictionary  memory MB', voting_sphere_dictionary["sphere_size"], voting_sphere_dictionary["vote_mapping"].size(),
          voting_sphere_dictionary["vote_mapping"].element_size() * voting_sphere_dictionary["vote_mapping"].nelement() / (1024 * 1024))

    # 2. model
    if ModelConfig.backbone == "stacked_hourglass":
        backbone = vpd.models.hg(
            planes=128, depth=ModelConfig.depth, num_stacks=ModelConfig.num_stacks, num_blocks=ModelConfig.num_blocks
        )
        print (backbone)
    else:
        raise NotImplementedError

    model = vpd.models.VanishingNet(backbone, voting_ht_dictionary, voting_sphere_dictionary)
    model = model.to(device)
    model = torch.nn.DataParallel(
        model, device_ids=list(range(args["--devices"].count(",") + 1))
    )
    torch.cuda.empty_cache()
    # print('model', model)
    ##### number of parameters in a model
    total_parameters = sum(p.numel() for p in model.parameters())
    print('Number of total parameters', total_parameters)
    ##### number of trainable parameters in a model
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters', trainable_parameters)
    torch.cuda.empty_cache()

    # 3. optimizer
    if Config.optimization.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.optimization.learning_rate * num_gpus,
            weight_decay=Config.optimization.weight_decay,
            amsgrad=Config.optimization.amsgrad,
        )
    elif Config.optimization.name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=Config.optimization.learning_rate * num_gpus,
            weight_decay=Config.optimization.weight_decay,
            momentum=Config.optimization.momentum,
        )
    else:
        raise NotImplementedError

    if resume_from:
        print('Resuming training from', resume_from)
        checkpoint = torch.load(osp.join(resume_from, "checkpoint_latest.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    output_directory = resume_from or generate_output_directory(args["--identifier"])
    print("Output directory:", output_directory)
    torch.cuda.empty_cache()

    try:
        trainer = vpd.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=validation_loader,
            batch_size=batch_size,
            out=output_directory,
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.epoch = checkpoint["epoch"]
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            print('Trainer epoch, iteration, best mean loss: ', trainer.epoch, trainer.iteration, trainer.best_mean_loss )
            del checkpoint
        trainer.train()
        print('Finished training at: ', str(datetime.datetime.now()))
    except BaseException:
        if len(glob.glob(f"{output_directory}/visualization/*")) <= 1:
            shutil.rmtree(output_directory)
        raise


if __name__ == "__main__":
    main()
