import os
import atexit
import random
import shutil
import signal
import os.path as osp
import threading
import subprocess
from timeit import default_timer as timer

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

import vpd.utils as utils
from vpd.config import C, M


class TrainerObject:
    def __init__(
        self, device, model, optimizer, train_loader, val_loader, batch_size, out
    ):
        self.device = device

        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = C.optim.max_epoch
        self.lr_decay_epoch = C.optim.lr_decay_epoch
        self.num_stacks = C.model.num_stacks
        self.mean_loss = self.best_mean_loss = 1e16

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def calculate_loss(self, result):
        losses = result["losses"]
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + list(losses[0].keys())
            self.metrics = np.zeros([self.num_stacks, len(self.loss_labels)])

        total_loss = 0
        for i in range(self.num_stacks):
            for j, name in enumerate(self.loss_labels):
                if name == "sum":
                    continue
                if name not in losses[i]:
                    print('Error: i, j, name', i, j, name)
                    assert i != 0
                    continue
                loss = losses[i][name].mean()
                self.metrics[i, 0] += loss.item()
                self.metrics[i, j] += loss.item()
                total_loss += loss
        return total_loss

    def validate(self):
        print("Running validation...", " " * 75)
        self.model.eval()

        npz = osp.join(self.out, "npz", f"{self.iteration * self.batch_size:09d}")
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        with torch.no_grad():
            for batch_idx, (image, target, gt_vpts) in enumerate(self.val_loader):
                image = image.to(self.device)
                target = target.to(self.device)
                input_dict = {"image": image, "target": target, "eval": True}
                result = self.model(input_dict)
                total_loss += self.calculate_loss(result)

        self.write_metrics(len(self.val_loader), total_loss, "validation", True)
        try:
            self.mean_loss = total_loss / len(self.val_loader)
        except:
            print("Divide by zero error in mean loss calculation")
        del total_loss

        torch.save(
            {
                "iteration": self.iteration,
                "epoch": self.epoch,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth.tar"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth.tar"),
                osp.join(self.out, "checkpoint_best.pth.tar"),
            )

    def train_epoch(self):
        self.model.train()
        time = timer()
        for batch_idx, (image, target, gt_vpts) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            self.metrics[...] = 0

            image = image.to(self.device)
            target = target.to(self.device)
            input_dict = {"image": image, "target": target, "eval": False}
            result = self.model(input_dict)

            loss = self.calculate_loss(result)
            if np.isnan(loss.item()):
                raise ValueError("Loss is NaN while training")
            self.optimizer.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1
            self.iteration += 1
            self.write_metrics(1, loss.item(), "training", do_print=False)

            del loss

            if self.iteration % 400 == 0:
                for k in range(0, torch.cuda.device_count()):
                    memory = torch.cuda.max_memory_allocated(k) / 1024.0 / 1024.0 / 1024.0
                    print('Device, max_memory_allocated', k, memory)
                    
            if self.iteration % 4 == 0:
                print(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()

    def write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                print(prt_str, " " * 7)
        try:
            self.writer.add_scalar(
                f"{prefix}/total_loss", total_loss / size, self.iteration
            )
        except:
            print("Division by zero")
        return total_loss

    def train(self):
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optimizer.param_groups[0]["lr"] /= 10
            print('Learning rate:', self.optimizer.param_groups[0]["lr"])
            self.train_epoch()
            if self.epoch % 2 != 0 and self.epoch != (self.max_epoch - 1): continue
            self.validate()

    def move(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            for name in obj:
                if isinstance(obj[name], torch.Tensor):
                    obj[name] = obj[name].to(self.device)
            return obj
        assert False


def temporary_print(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def permanent_print(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def launch_tensorboard(board_out, port, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = subprocess.Popen(["tensorboard", f"--logdir={board_out}", f"--port={port}"])

    def kill():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(kill)  
