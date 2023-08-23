import argparse
import logging
import math
import sys

import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .dataio.dataset_input import DataProcessing
from .model.losses import (
    CellCallingLoss,
    NucleiEncapsulationLoss,
    OverlapLoss,
    Oversegmentation,
    PosNegMarkerLoss,
)
from .model.model import SegmentationModel as Network
from .utils.utils import (
    get_experiment_id,
    json_file_to_pyobj,
    make_dir,
    save_fig_outputs,
)


def train(config):
    json_opts = json_file_to_pyobj(config.config_file)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create experiment directories
    if config.resume_epoch is None:
        make_new = True
    else:
        make_new = False
    timestamp = get_experiment_id(make_new, json_opts.experiment_dirs.load_dir)
    experiment_path = "experiments" + "/" + timestamp
    make_dir(experiment_path + "/" + json_opts.experiment_dirs.model_dir)
    make_dir(experiment_path + "/" + json_opts.experiment_dirs.samples_dir)

    # Set up the model
    logging.info("Initialising model")

    atlas_exprs = pd.read_csv(json_opts.data_sources.atlas_fp, index_col=0)
    n_genes = atlas_exprs.shape[1] - 3
    print("Number of genes: %d" % n_genes)

    if json_opts.model_params.name != "custom":
        model = smp.Unet(
            encoder_name=json_opts.model_params.name,
            encoder_weights=None,
            in_channels=n_genes,
            classes=2,
        )
    else:
        model = Network(n_channels=n_genes)

    model = model.to(device)

    # Dataloader
    logging.info("Preparing data")

    train_dataset = DataProcessing(
        json_opts.data_sources,
        json_opts.data_params,
        isTraining=True,
        total_steps=config.total_steps,
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )

    n_train_examples = len(train_loader)
    logging.info("Total number of training examples: %d" % n_train_examples)

    # Loss functions
    criterion_ne = NucleiEncapsulationLoss(json_opts.training_params.ne_weight, device)
    criterion_os = Oversegmentation(json_opts.training_params.os_weight, device)
    criterion_cc = CellCallingLoss(json_opts.training_params.cc_weight, device)
    criterion_ov = OverlapLoss(json_opts.training_params.ov_weight, device)
    criterion_pn = PosNegMarkerLoss(
        json_opts.training_params.pos_weight,
        json_opts.training_params.neg_weight,
        device,
    )

    # Optimiser
    if json_opts.training_params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=json_opts.training_params.learning_rate,
            weight_decay=1e-8,
        )
    elif json_opts.training_params.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=json_opts.training_params.learning_rate,
            betas=(json_opts.training_params.beta1, json_opts.training_params.beta2),
            weight_decay=json_opts.training_params.l2_reg_alpha,
        )
    else:
        sys.exit("Select optimiser from rmsprop or adam")

    global_step = 0

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = (
        lambda x: (
            ((1 + math.cos(x * math.pi / json_opts.training_params.total_epochs)) / 2)
            ** 1.0
        )
        * 0.95
        + 0.05
    )  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step

    # Starting epoch
    if config.resume_epoch is not None:
        initial_epoch = config.resume_epoch
    else:
        initial_epoch = 0

    # Restore saved model
    if config.resume_epoch is not None:
        load_path = (
            experiment_path
            + "/"
            + json_opts.experiment_dirs.model_dir
            + "/epoch_%d_step_%d.pth" % (config.resume_epoch, config.resume_step)
        )
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        assert epoch == config.resume_epoch
        print("Resume training, successfully loaded " + load_path)

    logging.info("Begin training")

    model = model.train()

    lrs = []

    for epoch in range(initial_epoch, json_opts.training_params.total_epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch =", (epoch + 1), " lr =", cur_lr)

        for step_epoch, (
            batch_x313,
            batch_n,
            batch_sa,
            batch_pos,
            batch_neg,
            coords_h1,
            coords_w1,
            nucl_aug,
            expr_aug_sum,
        ) in enumerate(train_loader):
            # Permute channels axis to batch axis
            # torch.Size([1, patch_size, patch_size, 313, n_cells]) to [n_cells, 313, patch_size, patch_size]
            batch_x313 = batch_x313[0, :, :, :, :].permute(3, 2, 0, 1)
            batch_sa = batch_sa.permute(3, 0, 1, 2)
            batch_pos = batch_pos.permute(3, 0, 1, 2)
            batch_neg = batch_neg.permute(3, 0, 1, 2)
            batch_n = batch_n.permute(3, 0, 1, 2)

            if batch_x313.shape[0] == 0:
                if (step_epoch % json_opts.save_freqs.model_freq) == 0:
                    save_path = (
                        experiment_path
                        + "/"
                        + json_opts.experiment_dirs.model_dir
                        + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                    )
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        save_path,
                    )
                    logging.info("Model saved: %s" % save_path)
                continue

            # Transfer to GPU
            batch_x313 = batch_x313.to(device)
            batch_sa = batch_sa.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            batch_n = batch_n.to(device)

            optimizer.zero_grad()

            seg_pred = model(batch_x313)

            # Compute losses
            loss_ne = criterion_ne(seg_pred, batch_n)
            loss_os = criterion_os(seg_pred, batch_n)
            loss_cc = criterion_cc(seg_pred, batch_sa)
            loss_ov = criterion_ov(seg_pred, batch_n)
            loss_pn = criterion_pn(seg_pred, batch_pos, batch_neg)

            loss_ne = loss_ne.squeeze()
            loss_os = loss_os.squeeze()
            loss_cc = loss_cc.squeeze()
            loss_ov = loss_ov.squeeze()
            loss_pn = loss_pn.squeeze()

            loss = loss_ne + loss_os + loss_cc + loss_ov + loss_pn

            # Optimisation
            loss.backward()
            optimizer.step()

            step_ne_loss = loss_ne.detach().cpu().numpy()
            step_os_loss = loss_os.detach().cpu().numpy()
            step_cc_loss = loss_cc.detach().cpu().numpy()
            step_ov_loss = loss_ov.detach().cpu().numpy()
            step_pn_loss = loss_pn.detach().cpu().numpy()

            step_train_loss = loss.detach().cpu().numpy()

            if (global_step % json_opts.save_freqs.sample_freq) == 0:
                coords_h1 = coords_h1.detach().cpu().squeeze().numpy()
                coords_w1 = coords_w1.detach().cpu().squeeze().numpy()
                sample_seg = seg_pred.detach().cpu().numpy()
                sample_n = nucl_aug.detach().cpu().numpy()
                sample_sa = batch_sa.detach().cpu().numpy()
                sample_expr = expr_aug_sum.detach().cpu().numpy()
                patch_fp = (
                    experiment_path
                    + "/"
                    + json_opts.experiment_dirs.samples_dir
                    + "/epoch_%d_%d_%d_%d.png"
                    % (epoch + 1, step_epoch, coords_h1, coords_w1)
                )

                save_fig_outputs(sample_seg, sample_n, sample_sa, sample_expr, patch_fp)

                print(
                    "Epoch[{}/{}], Step[{}], Loss:{:.4f}".format(
                        epoch + 1,
                        json_opts.training_params.total_epochs,
                        step_epoch,
                        step_train_loss,
                    )
                )
                # print('NE:{:.4f}, TC:{:.4f}, CC:{:.4f}, OV:{:.4f}, PN:{:.4f}'.format(step_ne_loss,
                #                                                                     step_os_loss,
                #                                                                     step_cc_loss,
                #                                                                     step_ov_loss,
                #                                                                     step_pn_loss))

            # Save model
            if (step_epoch % json_opts.save_freqs.model_freq) == 0:
                save_path = (
                    experiment_path
                    + "/"
                    + json_opts.experiment_dirs.model_dir
                    + "/epoch_%d_step_%d.pth" % (epoch + 1, step_epoch)
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_path,
                )
                logging.info("Model saved: %s" % save_path)

            global_step += 1

        # Update and append current LR
        scheduler.step()
        lrs.append(cur_lr)

    # Plot lr scheduler
    plt.plot(lrs, ".-", label="LambdaLR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.tight_layout()
    plt.savefig(experiment_path + "/LR.png", dpi=300)

    logging.info("Training finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="configs/config.json",
        type=str,
        help="config file path",
    )
    parser.add_argument(
        "--resume_epoch",
        default=None,
        type=int,
        help="resume training from this epoch, set to None for new training",
    )
    parser.add_argument(
        "--resume_step", default=0, type=int, help="resume training from this step"
    )
    parser.add_argument(
        "--total_steps", default=6000, type=int, help="total number of steps to train"
    )

    config = parser.parse_args()
    train(config)
