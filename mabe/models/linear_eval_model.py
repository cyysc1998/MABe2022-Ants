from collections import OrderedDict
from copy import deepcopy

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mabe.archs import define_network
from mabe.models.base_model import BaseModel
from mabe.simclr.modules import LARS
from mabe.utils import get_root_logger, master_only
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast


class LinearEvalModel(BaseModel):
    def __init__(self, opt):
        super(LinearEvalModel, self).__init__(opt)

        # define network
        self.net = define_network(deepcopy(opt["network"]))
        self.net = self.model_to_device(self.net)
        # self.print_network(self.net)

        self.init_training_settings()

        self.scaler = GradScaler()

    def init_training_settings(self):
        self.net.train()
        self.setup_optimizers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_type = train_opt["optim"].pop("type")
        if optim_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optim_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optim_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.net.parameters(), **train_opt["optim"]
            )
        elif optimizer == "LARS":
            optimizer = LARS(self.net.parameters(), **train_opt["optim"])
        else:
            raise NotImplementedError(f"optimizer {optim_type} is not supperted yet.")
        self.optimizers.append(self.optimizer)

    def feed_data(self, data):
        self.x = data["x"].to(self.device, non_blocking=True)
        self.label = data["label"].to(self.device, non_blocking=True)

    def optimize_parameters_amp(self):
        self.optimizer.zero_grad()

        with autocast():
            loss_dict = OrderedDict()
            with autocast():
                l_total = 0
                logit = self.net(self.x)
                l_cls = F.cross_entropy(logit, self.label.long())
                l_total += l_cls
                loss_dict["l_cls"] = l_cls

        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def test(self, dataset, dataloader):
        self.net.eval()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logits = []
        labels = []

        for data in dataloader:
            self.feed_data(data)
            with autocast():
                logit = self.net(self.x)
            logits.append(logit)
            labels.append(self.label)

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        # print(1, rank, logits.shape, labels.shape)
        dist.barrier()

        logits_list = [torch.zeros_like(logits) for _ in range(world_size)]
        dist.all_gather(logits_list, logits)
        logits = torch.cat(logits_list)
        labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
        dist.all_gather(labels_list, labels)
        labels = torch.cat(labels_list)
        # print(2, rank, logits.shape, labels.shape)

        self.net.train()

        return cal_metric(logits, labels)


def cal_metric(logits, labels):
    preds = torch.argmax(logits, dim=1)
    TP = ((labels == 1) & (preds == 1)).sum().item()
    FN = ((labels == 1) & (preds == 0)).sum().item()
    FP = ((labels == 0) & (preds == 1)).sum().item()
    TN = ((labels == 0) & (preds == 0)).sum().item()

    acc = P = R = f1 = P0 = 0
    if (TP + FN + FP + TN) != 0:
        acc = (TP + TN) / (TP + FN + FP + TN)
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    if (TP + FN) != 0:
        R = TP / (TP + FN)
    if (P + R) != 0:
        f1 = 2 * P * R / (P + R)
    if (TN + FN) != 0:
        P0 = TN / (TN + FN)
    mP = (P + P0) / 2.0

    # logger = get_root_logger()
    # logger.info(
    #     f"acc: {acc:.4f}\t"
    #     f"P: {P:.4f}\t"
    #     f"R: {R:.4f}\t"
    #     f"f1: {f1:.4f}\t"
    #     f"mP: {mP:.4f}\t"
    # )
    return f1
