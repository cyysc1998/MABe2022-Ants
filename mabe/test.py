import logging
import math

import torch

from mabe.data import create_dataset_dataloader
from mabe.models import create_model
from mabe.train import init_loggers, parse_options
from mabe.utils import MessageLogger


def main():
    opt = parse_options()
    seed = opt["manual_seed"]

    # initialize loggers
    logger, tb_logger = init_loggers(
        opt, prefix="test", log_level=logging.INFO, use_tb_logger=False
    )

    # create train, validation, test datasets and dataloaders
    train_set, train_loader, num_iter_per_epoch = create_dataset_dataloader(
        opt["datasets"]["train"], shuffle=True, seed=seed
    )
    val_set, val_loader, _ = create_dataset_dataloader(
        opt["datasets"]["val"], shuffle=False, seed=seed
    )
    test_set, test_loader, _ = create_dataset_dataloader(
        opt["datasets"]["test"], shuffle=False, seed=seed
    )

    # log training statistics
    total_iters = int(opt["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    logger.info(f"\n\tTotal epochs: {total_epochs}\n\tTotal iters: {total_iters}.")

    # create model
    model = create_model(opt)

    current_iter = 0
    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # training
    logger.info(f"Start testing.")

    # for current_iter in range(
    #     opt["test"]["st_iter"], opt["test"]["ed_iter"] + 1, opt["test"]["test_freq"]
    # ):
    for current_iter in opt["test"]["iter_list"]:
        ckpt_path = f"{opt['path']['models']}/net_{current_iter}.pth"
        model.load_network(model.net, ckpt_path)
        model.test(test_set, test_loader)
        model.save_result(-1, current_iter, "test")

    logger.info(f"End of test.")


if __name__ == "__main__":
    main()
