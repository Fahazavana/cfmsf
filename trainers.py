import json
import os
import time

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
    _LRScheduler,
)
from matplotlib.ticker import MaxNLocator, ScalarFormatter


class AETrainer:
    def __init__(
        self,
        encoder,
        decoder,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        name,
        device,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.name = name
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_size = len(self.train_loader.dataset)
        self.valid_size = len(self.valid_loader.dataset)
        self.train_loss = []
        self.valid_loss = []
        self.train_lr = []

    def _run_epoch(self, desc):
        lr = self.optimizer.param_groups[-1]["lr"]
        train_loss = 0.0
        valid_loss = 0.0
        self.encoder.train()
        self.decoder.train()
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            loss = self.criterion(decoded, inputs)
            loss.backward()
            nn.utils.clip_grad_value_(self.encoder.parameters(), 1, foreach=True)
            nn.utils.clip_grad_value_(self.decoder.parameters(), 1, foreach=True)
            self.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            batch = f"{batch_idx+1}/{len(self.train_loader)}"
            msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{train_loss / self.train_size:>7.2e} | val_loss:{0.0:>7.2e} | lr:{lr:>7.1e}"
            print(msg, end="")
        train_loss /= self.train_size
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            for inputs, _ in self.valid_loader:
                inputs = inputs.to(self.device)
                encoded = self.encoder(inputs)
                decoded = self.decoder(encoded)
                loss = self.criterion(decoded, inputs)
                valid_loss += loss.item() * inputs.size(0)
                msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{train_loss:>7.2e} | val_loss:{valid_loss / self.valid_size:>7.2e} | lr:{lr:>7.1e}"
                print(msg, end="")

        print()
        return train_loss, valid_loss / self.valid_size

    def train(self, epochs: int, patience=10, delta=0.0, load_best=True):
        os.makedirs(self.name, exist_ok=True)
        no_improvement_counter = 0
        best_valid_loss = float("inf")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience // 2, min_lr=1e-8
        )
        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)

        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            train_loss, valid_loss = self._run_epoch(
                desc=f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            )
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            scheduler.step(valid_loss)
            if valid_loss < (best_valid_loss - delta):
                no_improvement_counter = 0
                best_valid_loss = valid_loss
                torch.save(self.encoder, f"{self.name}/{self.name}_encoder.pth")
                torch.save(self.decoder, f"{self.name}/{self.name}_decoder.pth")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "validation_loss": self.valid_loss,
            "train_loss": self.train_loss,
            "train_lr": self.train_lr,
        }
        with open(f"{self.name}/ae_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.encoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_encoder.pth", weights_only=False
                ).state_dict()
            )
            self.decoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_decoder.pth", weights_only=False
                ).state_dict()
            )
            self.encoder.eval()
            self.decoder.eval()

    def plot_training_loss(self):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("Epochs")
        ax1.plot(self.train_loss, color="tab:blue", label="Training loss")
        ax1.plot(self.valid_loss, color="tab:orange", label="Validation loss")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.yaxis.set_major_formatter(formatter)
        plt.legend()
        ax2 = ax1.twinx()
        color = "tab:gray"
        ax2.set_ylabel("Learning Rate", color=color)
        ax2.plot(self.train_lr, "--", color=color, label="Learning Rate")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.yaxis.set_major_formatter(formatter)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        ax1.legend(lines, labels)
        fig.tight_layout()
        plt.savefig(f"{self.name}/ae_training.pdf")
        plt.show()

class AESTrainer(AETrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train(self, epochs: int, update_freq:int, patience=10, delta=0.0, load_best=True):
        os.makedirs(self.name, exist_ok=True)
        no_improvement_counter = 0
        best_valid_loss = float("inf")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience // 2, min_lr=1e-8
        )
        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)

        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            train_loss, valid_loss = self._run_epoch(
                desc=f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            )
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)
            scheduler.step(valid_loss)
            if (epochs+1)%update_freq ==0:
                self.criterion.feature_extractor.features.load_state_dict(self.encoder.convolutional_features.state_dict())
                for p in self.criterion.parameters():
                    p.requires_grad = False
                    
            if valid_loss < (best_valid_loss - delta):
                no_improvement_counter = 0
                best_valid_loss = valid_loss
                torch.save(self.encoder, f"{self.name}/{self.name}_encoder.pth")
                torch.save(self.decoder, f"{self.name}/{self.name}_decoder.pth")
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "validation_loss": self.valid_loss,
            "train_loss": self.train_loss,
            "train_lr": self.train_lr,
        }
        with open(f"{self.name}/ae_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.encoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_encoder.pth", weights_only=False
                ).state_dict()
            )
            self.decoder.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_decoder.pth", weights_only=False
                ).state_dict()
            )
            self.encoder.eval()
            self.decoder.eval()


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        final_lr: float,
        initial_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.final_lr = final_lr
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr
            param_group["initial_lr"] = self.final_lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                self.initial_lr
                + (self.final_lr - self.initial_lr)
                * self.last_epoch
                / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]
        else:
            return [self.final_lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CFMTrainer:
    def __init__(self, cfm, encoder, optimizer, train_loader, name, device):
        self.cfm = cfm
        self.name = name
        self.device = device
        self.encoder = encoder
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_size = len(self.train_loader.dataset)
        self.criterion = nn.MSELoss()
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.01)
        self.sampler = self.fm.sample_location_and_conditional_flow
        self.train_loss = []
        self.train_lr = []

    def train(
        self,
        epochs,
        min_lr=1e-6,
        high_lr=2e-4,
        warmup=10,
        patience=10,
        delta=0.0,
        load_best=True,
    ):
        os.makedirs(self.name, exist_ok=True)
        self.cfm.train()
        self.cfm = self.cfm.to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        best_loss = float("inf")
        epochs_no_improve = 0

        scheduler_lists = [
            WarmupLR(
                self.optimizer, warmup_steps=warmup, initial_lr=min_lr, final_lr=high_lr
            ),
            CosineAnnealingLR(self.optimizer, T_max=epochs - warmup, eta_min=min_lr),
        ]
        scheduler = SequentialLR(
            self.optimizer, milestones=[warmup], schedulers=scheduler_lists
        )

        start_time = time.time()
        epochstr = str(epochs)
        nbdigit = len(epochstr)
        for epoch in range(epochs):
            self.train_lr.append(scheduler.get_last_lr())
            desc = f"Epoch [{str(epoch +1).zfill(nbdigit)}/{epochstr}]"
            total_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    x1 = self.encoder(inputs)

                self.optimizer.zero_grad()
                x0 = torch.randn_like(x1)
                t, xt, ut = self.sampler(x0, x1)
                if self.cfm.nntype == "mlp":
                    vt = self.cfm((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
                elif self.cfm.nntype == "unet":
                    vt = self.cfm((t, xt))
                else:
                    raise ValueError(f"Unknown model type: {self.cfm.type}")

                loss = self.criterion(ut, vt)
                loss.backward()
                nn.utils.clip_grad_value_(self.cfm.parameters(), 1, foreach=True)
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                batch = f"{batch_idx+1}/{len(self.train_loader)}"
                msg = f"\r{desc:^20} batch:{batch:^10} | train_loss:{total_loss / self.train_size:>7.2e} | lr:{self.train_lr[-1][0]:>7.1e}"
                print(msg, end="")
            total_loss /= self.train_size

            self.train_loss.append(total_loss)

            scheduler.step()
            print()
            if total_loss < (best_loss - delta):
                best_loss = total_loss
                epochs_no_improve = 0
                torch.save(self.cfm, f"{self.name}/{self.name}_cfm.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        end_time = time.time()
        duration = end_time - start_time
        training_data = {
            "start": start_time,
            "end": end_time,
            "duration": duration,
            "train_loss": self.train_loss,
        }
        with open(f"{self.name}/cfm_training_log.json", "w") as f:
            json.dump(training_data, f)
        if load_best:
            self.cfm.load_state_dict(
                torch.load(
                    f"{self.name}/{self.name}_cfm.pth", weights_only=False
                ).state_dict()
            )
        self.cfm.eval()

    def plot_training_loss(self):
        fig, ax1 = plt.subplots()
        color1 = "tab:red"
        ax1.plot(self.train_loss, color=color1, label="Loss")
        ax1.set_ylabel("Loss", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xlabel("Epochs")

        ax2 = ax1.twinx()
        color2 = "tab:gray"
        ax2.plot(self.train_lr, "--", color=color2, label="LR")
        ax2.set_ylabel("LR", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        ax1.legend(lines, labels)
        fig.tight_layout()
        plt.savefig(f"{self.name}/cfm_training.pdf")
        plt.show()

#######################################################################
#### OLD TRAINER and experiments (old folder)
# from dataclasses import dataclass
# from typing import Any, Dict, Optional, Tuple

# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader

# from .utils import WarmupLR


# class AETrainer:
#     def __init__(
#         self,
#         encoder: nn.Module,
#         decoder: nn.Module,
#         model_name: str,
#         optimizer: optim.Optimizer,
#         criterion: nn.Module,
#         train_loader: DataLoader,
#         val_loader: Optional[DataLoader] = None,
#         device: str = "cpu",
#     ):
#         self.device = device
#         self.encoder = encoder.to(self.device)
#         self.decoder = decoder.to(self.device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.train_size = len(self.train_loader.dataset)
#         self.val_size = (
#             len(self.val_loader.dataset) if self.val_loader is not None else 0
#         )
#         self.model_name = model_name
#         self.train_loss = []
#         self.val_loss = []
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode="min", patience=10, min_lr=1e-10
#         )

#     def _run_epoch(self, loader: DataLoader, is_train: bool) -> float:
#         """
#         Runs a single training or validation epoch.

#         Args:
#             loader (DataLoader): The data loader for the epoch.
#             is_train (bool): Whether to train or validate.

#         Returns:
#             float: The average loss for the epoch.
#         """
#         if is_train:
#             self.encoder.train()
#             self.decoder.train()
#             data_size = self.train_size
#         else:
#             self.encoder.eval()
#             self.decoder.eval()
#             data_size = self.val_size

#         total_loss = 0.0
#         for inputs, _ in loader:
#             inputs = inputs.to(self.device)
#             if is_train:
#                 self.optimizer.zero_grad()

#             with torch.set_grad_enabled(is_train):
#                 encoded = self.encoder(inputs)
#                 decoded = self.decoder(encoded)
#                 loss = self.criterion(decoded, inputs)

#                 if is_train:
#                     loss.backward()
#                     self.optimizer.step()

#             total_loss += loss.item() * inputs.size(0)

#         return total_loss / data_size

#     def train(self, epochs: int, patience=10, delta=1e-4):
#         """
#         Trains the autoencoder model.

#         Args:
#             epochs (int): The number of epochs to train for.
#         """
#         best_val_loss = float("inf")
#         epochs_no_improve = 0
#         for epoch in range(epochs):
#             train_loss = self._run_epoch(loader=self.train_loader, is_train=True)
#             self.train_loss.append(train_loss)
#             msg = f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}"
#             if self.val_loader is not None:
#                 val_loss = self._run_epoch(loader=self.val_loader, is_train=False)
#                 self.val_loss.append(val_loss)
#                 self.scheduler.step(val_loss)
#                 msg += f" Val Loss: {val_loss:.4f}"
#             print(msg)
#             # Early stopping
#             if val_loss < (best_val_loss - delta):
#                 best_val_loss = val_loss
#                 epochs_no_improve = 0
#                 torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
#                 torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break


# class AEETrainer:
#     def __init__(
#         self,
#         encoder: nn.Module,
#         decoder: nn.Module,
#         model_name:str,
#         optimizer: optim.Optimizer,
#         criterion: nn.Module,
#         train_loader: DataLoader,
#         val_loader: Optional[DataLoader] = None,
#         device: str = "cpu",
#     ):
#         self.device = device
#         self.encoder = encoder.to(self.device)
#         self.decoder = decoder.to(self.device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.train_size = len(self.train_loader.dataset)
#         self.val_size = (
#             len(self.val_loader.dataset) if self.val_loader is not None else 0
#         )
#         self.model_name = model_name
#         self.train_loss = []
#         self.val_loss = []
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode="min", patience=10, min_lr=1e-10
#         )

#     def _run_epoch(self, loader: DataLoader, is_train: bool) -> float:
#         """
#         Runs a single training or validation epoch.

#         Args:
#             loader (DataLoader): The data loader for the epoch.
#             is_train (bool): Whether to train or validate.

#         Returns:
#             float: The average loss for the epoch.
#         """
#         if is_train:
#             self.encoder.train()
#             self.decoder.train()
#             data_size = self.train_size
#         else:
#             self.encoder.eval()
#             self.decoder.eval()
#             data_size = self.val_size

#         total_loss = 0.0
#         for inputs, labels in loader:
#             inputs = inputs.to(self.device)
#             labels = labels.to(self.device)
#             if is_train:
#                 self.optimizer.zero_grad()

#             with torch.set_grad_enabled(is_train):
#                 encoded = self.encoder(inputs, labels)
#                 decoded = self.decoder(encoded)
#                 loss = self.criterion(decoded, inputs)

#                 if is_train:
#                     loss.backward()
#                     self.optimizer.step()

#             total_loss += loss.item() * inputs.size(0)

#         return total_loss / data_size

#     def train(self, epochs: int, patience=10, delta=1e-4):
#         """
#         Trains the autoencoder model.

#         Args:
#             epochs (int): The number of epochs to train for.
#         """
#         best_val_loss = float("inf")
#         epochs_no_improve = 0
#         for epoch in range(epochs):
#             train_loss = self._run_epoch(loader=self.train_loader, is_train=True)
#             self.train_loss.append(train_loss)
#             msg = f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}"
#             if self.val_loader is not None:
#                 val_loss = self._run_epoch(loader=self.val_loader, is_train=False)
#                 self.val_loss.append(val_loss)
#                 self.scheduler.step(val_loss)
#                 msg += f" Val Loss: {val_loss:.4f}"
#             print(msg)
#             # Early stopping
#             if val_loss < (best_val_loss - delta):
#                 best_val_loss = val_loss
#                 epochs_no_improve = 0
#                 torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
#                 torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break


# class AECTrainer:
#     def __init__(
#         self,
#         encoder: nn.Module,
#         decoder: nn.Module,
#         model_name:str,
#         classifier: nn.Module,
#         optimizer: optim.Optimizer,
#         criterion1: nn.Module,
#         criterion2: nn.Module,
#         train_loader,
#         val_loader=None,
#         device: str = "cpu",
#     ):
#         self.device = device
#         self.encoder = encoder.to(self.device)
#         self.decoder = decoder.to(self.device)
#         self.classifier = classifier.to(self.device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.optimizer = optimizer
#         self.criterion1 = criterion1
#         self.criterion2 = criterion2
#         self.train_size = len(self.train_loader.dataset)
#         self.val_size = (
#             len(self.val_loader.dataset) if self.val_loader is not None else 0
#         )
#         self.model_name = model.name
#         self.train_loss = []
#         self.val_loss = []

#         self.val_accuracy = []
#         self.val_cls_loss = []
#         self.val_rec_loss = []
#         self.val_accuracy = []

#         self.train_accuracy = []
#         self.train_cls_loss = []
#         self.train_rec_loss = []
#         self.train_accuracy = []

#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, mode="min", patience=10, min_lr=1e-10
#         )

#     def _run_epoch(self, loader, is_train: bool) -> float:
#         """
#         Runs a single training or validation epoch.

#         Args:
#             loader (DataLoader): The data loader for the epoch.
#             is_train (bool): Whether to train or validate.

#         Returns:
#             float: The average loss for the epoch.
#         """
#         if is_train:
#             self.encoder.train()
#             self.decoder.train()
#             self.classifier.train()
#             data_size = self.train_size
#         else:
#             self.encoder.eval()
#             self.decoder.eval()
#             self.classifier.eval()
#             data_size = self.val_size

#         total_loss = 0.0
#         total_recloss = 0.0
#         total_clsloss = 0.0
#         correct = 0
#         total = 0
#         for inputs, labels in loader:
#             inputs = inputs.to(self.device)
#             labels = labels.to(self.device)

#             if is_train:
#                 self.optimizer.zero_grad()

#             with torch.set_grad_enabled(is_train):
#                 encoded = self.encoder(inputs)
#                 decoded = self.decoder(encoded)
#                 classified = self.classifier(encoded)

#                 loss1 = self.criterion1(decoded, inputs)
#                 loss2 = self.criterion2(classified, labels)
#                 loss = loss2 + 10 * loss1
#                 if is_train:
#                     loss.backward()
#                     self.optimizer.step()
#                 _, predicted = torch.max(classified.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#             total_recloss += loss1.item() * inputs.size(0)
#             total_clsloss += loss2.item() * inputs.size(0)
#             total_loss += loss.item() * inputs.size(0)

#         accuracy = 100 * correct / total
#         return (
#             total_loss / data_size,
#             total_clsloss / data_size,
#             total_recloss / data_size,
#             accuracy,
#         )

#     def train(self, epochs: int, patience=10, delta=1e-4):
#         """
#         Trains the autoencoder model.

#         Args:
#             epochs (int): The number of epochs to train for.
#         """
#         best_val_loss = float("inf")
#         epochs_no_improve = 0
#         for epoch in range(epochs):
#             train_loss, cls, rec, acc = self._run_epoch(
#                 loader=self.train_loader, is_train=True
#             )
#             self.train_loss.append(train_loss)
#             self.train_cls_loss.append(cls)
#             self.train_rec_loss.append(rec)
#             self.train_accuracy.append(acc)
#             msg = f"Epoch [{epoch + 1}/{epochs}], TL: {train_loss:.4f}, TC:{cls:.4f}, TA:{acc:.4f},  TR={rec:.4f}"
#             if self.val_loader is not None:
#                 val_loss, vcls, vrec, vacc = self._run_epoch(
#                     loader=self.val_loader, is_train=False
#                 )
#                 self.val_loss.append(val_loss)
#                 self.val_cls_loss.append(vcls)
#                 self.val_rec_loss.append(vrec)
#                 self.val_accuracy.append(vacc)
#                 self.scheduler.step(val_loss)
#                 self.scheduler.step(val_loss)
#                 msg += (
#                     f" | VL: {val_loss:.4f}, VC:{vcls:.4f}, VA:{vacc:.4f} VR:{vrec:.4f}"
#                 )
#             print(msg)
#             # Early stopping
#             if val_loss < (best_val_loss - delta):
#                 best_val_loss = val_loss
#                 epochs_no_improve = 0
#                 torch.save(self.encoder.state_dict(), f"{self.model_name}_encoder.pth")
#                 torch.save(self.decoder.state_dict(), f"{self.model_name}_decoder.pth")
#                 torch.save(self.classifier.state_dict(), f"{self.model_name}_classifier.pth")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break



# class CFMTrainer:
#     def __init__(
#         self,
#         cfm_model: nn.Module,
#         sampler: Any,
#         encoder: nn.Module,
#         optimizer: optim.Optimizer,
#         criterion: nn.Module,
#         train_loader: DataLoader,
#         model_name:str,
#         device: str = "cpu",
#     ):
#         self.device = device
#         self.cfm_model = cfm_model.to(device)
#         self.sampler = sampler
#         self.encoder = encoder.to(device)
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.train_loader = train_loader
#         self.data_size = len(self.train_loader.dataset)
#         self.train_loss = []
#         self.train_lr = []
#         self.model_name = model_name
#         self.encoder.requires_grad_(False)
#         self.encoder.eval()

#     def train(
#         self,
#         epochs: int,
#         lr_scheduler: Optional[WarmupLR] = None,
#         patience: int = 10,
#         delta: float = 1e-4,
#     ):
#         """
#         Trains the CFM model.

#         Args:
#             epochs (int): The number of epochs to train for.
#             lr_scheduler (Optional[WarmupLR], optional): The learning rate scheduler to use. Defaults to None.
#             patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
#             delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 1e-4.
#         """

#         self.cfm_model.train()
#         best_loss = float("inf")
#         epochs_no_improve = 0

#         for epoch in range(epochs):
#             if lr_scheduler is not None:
#                 self.train_lr.append(lr_scheduler.get_last_lr())
#             total_loss = 0.0
#             for inputs, _ in self.train_loader:
#                 inputs = inputs.to(self.device)
#                 with torch.no_grad():
#                     x1 = self.encoder(inputs)

#                 self.optimizer.zero_grad()
#                 x0 = torch.randn_like(x1)
#                 t, xt, ut = self.sampler(x0, x1)

#                 if self.cfm_model.model_type == "mlp":
#                     vt = self.cfm_model((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
#                 elif self.cfm_model.model_type == "unet":
#                     vt = self.cfm_model((t, xt))
#                 else:
#                     raise ValueError(f"Unknown model type: {self.cfm_model.model_type}")

#                 loss = self.criterion(ut, vt)
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item() * inputs.size(0)
#             total_loss /= self.data_size

#             self.train_loss.append(total_loss)

#             if lr_scheduler is not None:
#                 lr_scheduler.step()
#                 print(
#                     f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}, lr={self.train_lr[-1][0]:.4f}"
#                 )
#             else:
#                 print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}")

#             # Early stopping
#             if total_loss < (best_loss - delta):
#                 best_loss = total_loss
#                 epochs_no_improve = 0
#                 torch.save(self.encoder.state_dict(), f"{self.model_name}_cfm.pth")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break


# class CFMETrainer:
#     def __init__(
#         self,
#         cfm_model: nn.Module,
#         sampler: Any,
#         encoder: nn.Module,
#         optimizer: optim.Optimizer,
#         criterion: nn.Module,
#         train_loader: DataLoader,
#         model_name:str,
#         device: str = "cpu",
#     ):
#         self.device = device
#         self.cfm_model = cfm_model.to(device)
#         self.sampler = sampler
#         self.encoder = encoder.to(device)
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.train_loader = train_loader
#         self.data_size = len(self.train_loader.dataset)
#         self.train_loss = []
#         self.train_lr = []
#         self.model_name = model_name
#         self.encoder.requires_grad_(False)
#         self.encoder.eval()

#     def train(
#         self,
#         epochs: int,
#         lr_scheduler: Optional[WarmupLR] = None,
#         patience: int = 10,
#         delta: float = 1e-4,
#     ):
#         """
#         Trains the CFM model.

#         Args:
#             epochs (int): The number of epochs to train for.
#             lr_scheduler (Optional[WarmupLR], optional): The learning rate scheduler to use. Defaults to None.
#             patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
#             delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 1e-4.
#         """

#         self.cfm_model.train()
#         best_loss = float("inf")
#         epochs_no_improve = 0

#         for epoch in range(epochs):
#             if lr_scheduler is not None:
#                 self.train_lr.append(lr_scheduler.get_last_lr())
#             total_loss = 0.0
#             for inputs, labels in self.train_loader:
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 with torch.no_grad():
#                     x1 = self.encoder(inputs, labels)

#                 self.optimizer.zero_grad()
#                 x0 = torch.randn_like(x1)
#                 t, xt, ut = self.sampler(x0, x1)

#                 if self.cfm_model.model_type == "mlp":
#                     vt = self.cfm_model((torch.cat([t.unsqueeze(-1), xt], dim=-1),))
#                 elif self.cfm_model.model_type == "unet":
#                     vt = self.cfm_model((t, xt))
#                 else:
#                     raise ValueError(f"Unknown model type: {self.cfm_model.model_type}")

#                 loss = self.criterion(ut, vt)
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item() * inputs.size(0)
#             total_loss /= self.data_size

#             self.train_loss.append(total_loss)

#             if lr_scheduler is not None:
#                 lr_scheduler.step()
#                 print(
#                     f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}, lr={self.train_lr[-1][0]:.4f}"
#                 )
#             else:
#                 print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss:.4f}")

#             # Early stopping
#             if total_loss < (best_loss - delta):
#                 best_loss = total_loss
#                 epochs_no_improve = 0
#                 torch.save(self.encoder.state_dict(), f"{self.model_name}_cfm.pth")
#             else:
#                 epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print(f"Early stopping at epoch {epoch + 1}")
#                     break
