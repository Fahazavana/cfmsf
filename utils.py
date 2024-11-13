import os
import platform
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from pytorch_fid.inception import InceptionV3
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T


def get_device():
    if platform.platform().lower().startswith("mac"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


class FIDDataSet(Dataset):
    def __init__(self, real, generated):
        if len(real) != len(generated):
            raise ValueError("The two dataset must have the same size")
        self.real = real
        self.generated = generated

    def __getitem__(self, idx):
        real = self.real[idx]
        gen = self.generated[idx]
        if isinstance(real, tuple):
            real = real[0]
        if isinstance(gen, tuple):
            gen = gen[0]
        return real, gen

    def __len__(self):
        return len(self.real)


class GeneratedData(torch.utils.data.Dataset):

    def __init__(self, root_dir, resize=False):
        self.root_dir = root_dir
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        self.resize = resize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        images = np.load(self.files[idx])
        images = torch.from_numpy(images)
        if self.resize:
            images = T.functional.resize(images, size=self.resize)
        if images.shape[0] == 1:
            images = images.repeat(3, 1, 1)
        return images


##############################
# Frechet Inception Distance #
# #############################

class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


@torch.no_grad()
def generate(nbr_sample, cfm, decoder, device):
    t, s = cfm.sample(nbr_sample, device)
    return decoder(s)


def save_image(image, output_dir, index):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"image_{index}.npy"), image.numpy())


def generate_and_save(total_samples, batch_size, cfm, decoder, output_dir, device):
    num_full_batches = total_samples // batch_size
    remaining_samples = total_samples % batch_size
    saved = 0

    for _ in range(num_full_batches):
        generated_images = generate(batch_size, cfm, decoder, device).cpu()
        for i in range(generated_images.shape[0]):
            save_image(generated_images[i], output_dir, saved)
            saved += 1
            print(f"\rGenerated images saved at '{output_dir}': {saved}", end="")

    if remaining_samples > 0:
        generated_images = generate(remaining_samples, cfm, decoder, device).cpu()
        for i in range(generated_images.shape[0]):
            save_image(generated_images[i], output_dir, saved)
            saved += 1
            print(f"\rGenerated images saved at '{output_dir}': {saved}", end="")
    print()


@torch.no_grad()
def reconstruct_and_save(data_loader, encoder, decoder, output_dir, device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    saved = 0
    for images, _ in data_loader:
        reconstructed = decoder(encoder(images.to(device))).cpu()
        for i in range(reconstructed.shape[0]):
            save_image(reconstructed[i], output_dir, saved)
            saved += 1
            print(f"\rReconstructed images saved at '{output_dir}': {saved}", end="")
    print()


def compute_fid(dims, data_loader, device):
    def evaluation_step(engine, batch):
        real, fake = batch
        return real.squeeze(0), fake.squeeze(0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx])

    wrapper_model = WrapperInceptionV3(inception_model)
    wrapper_model.eval()
    pytorch_fid_metric = FID(
        num_features=dims, feature_extractor=wrapper_model, device=device
    )

    evaluator = Engine(evaluation_step)
    pytorch_fid_metric.attach(evaluator, "fid")

    evaluator.run(data_loader, max_epochs=1)
    metrics = evaluator.state.metrics

    return metrics


##############################
#  Improved Recall Precision  #
# #############################

Manifold = namedtuple("Manifold", ["features", "radii"])
PrecisionAndRecall = namedtuple("PrecisionAndRecall", ["precision", "recall"])


class PrecisionRecall:
    def __init__(self, k=3, device="cpu", model=None):
        self.manifold_ref = None
        self.device = device
        self.k = k
        if model is None:
            self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.vgg16.classifier = self.vgg16.classifier[:4]
        else:
            self.vgg16 = model
        self.vgg16.eval()
        for p in self.vgg16.parameters():
            p.requires_grad = False
        self.vgg16 = self.vgg16.to(self.device)

    def precision_and_recall(self, subject):
        assert self.manifold_ref is not None, "call compute_manifold_ref() first"
        manifold_subject = self.compute_manifold(subject)
        print(" -> Computing Precision")
        precision = compute_metric(
            self.manifold_ref, manifold_subject.features, "Computing Precision"
        )
        print(" -> Computing Recall")
        recall = compute_metric(
            manifold_subject, self.manifold_ref.features, "Computing Recall"
        )
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, dataloader_ref):
        self.manifold_ref = self.compute_manifold(dataloader_ref)

    def realism(self, image):
        """
        args:
            image: torch.Tensor of 1 x C x H x W
        """
        feat = self.extract_features(image)
        return realism(self.manifold_ref, feat)

    def compute_manifold(self, loader):
        feats = self.extract_features(loader)
        # radii
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

    def extract_features(self, dataloader):
        features = []
        for i, batch in enumerate(dataloader):
            batch = batch[0] if isinstance(batch, list) else batch
            feature = self.vgg16(batch.to(self.device))
            features.append(feature.cpu().data.numpy())
            print(f"\r[{'Feature Extraction':^20}] {i+1}/{len(dataloader)} batches", end=" ")
        print()
        return np.concatenate(features, axis=0)



def compute_pairwise_distances(X, Y=None):
    """
    Compute pairwise distances between points in X and Y.

    Args:
        X: np.array of shape (N, dim)
        Y: np.array of shape (M, dim), optional

    Returns:
        np.array of shape (N, M) if Y is provided, otherwise (N, N)
    """
    print(f" -> Compute Pairwaise Distance")
    X = X.astype(np.float32) 
    X_norm_square = np.sum(X**2, axis=1, keepdims=True)

    if Y is None:
        Y = X
        num_Y = X.shape[0]
        Y_norm_square = X_norm_square
    else:
        Y = Y.astype(np.float32)
        Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
        num_Y = Y.shape[0]

    diff_square = X_norm_square + Y_norm_square.T - 2 * np.dot(X, Y.T)
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print(
            f" * WARNING: fixing {idx.sum()} negative entry"
            )
        
    return np.sqrt(diff_square)
    

def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii



def get_kth_value(np_array, k):
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value



def compute_metric(manifold_ref, feats_subject, desc=""):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in range(num_subjects):
        count += (dist[:, i] < manifold_ref.radii).any()
        print(f"\r[{desc:^20}] {i+1}/{num_subjects} images", end="")
    print()
    return count / num_subjects



def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)



def realism(manifold_real, feat_subject):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    eps = 1e-6
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism


def compute_precision_recall(real_loader, gen_loader, device, k=3):
    ipr = PrecisionRecall(device=device, k=k)
    ipr.compute_manifold_ref(real_loader)
    return ipr.precision_and_recall(gen_loader)
