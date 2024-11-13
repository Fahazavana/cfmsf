import torch
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.features = models.vgg11(weights=models.VGG11_Weights.DEFAULT).features
        self.activation_indices = [4, 9, 15, 20]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, level):
        activation_indices = self.activation_indices[:level]
        max_id = activation_indices[-1] + 1
        outputs = []
        for i, layer in enumerate(self.features[:max_id]):
            x = layer(x)
            if i in activation_indices:
                outputs.append(x)
        return outputs


class VGG11Loss(nn.Module):
    def __init__(self, level, device):
        super(VGG11Loss, self).__init__()
        self.feature_extractor = FeatureExtractor().to(device)
        if level < 1 or level > 4:
            raise ValueError("Level must be between 1 and 4.")
        self.level = level

    def forward(self, I1, I2):
        if I1.shape[1] != 3:
            I1 = I1.repeat(1, 3, 1, 1)
        if I2.shape[1] != 3:
            I2 = I2.repeat(1, 3, 1, 1)

        features1 = self.feature_extractor(I1, self.level)
        features2 = self.feature_extractor(I2, self.level)

        loss = torch.square(I1 - I2).mean()
        for f1, f2 in zip(features1, features2):
            layer_loss = torch.square(f1 - f2).mean()
            loss += layer_loss
        return loss


class SelfFeatureExtractor(nn.Module):
    def __init__(self, encoder):
        super(SelfFeatureExtractor, self).__init__()
        self.features = encoder.convolutional_features
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        
    @torch.no_grad()
    def forward(self, x, level=-1):
        outputs = []
        for i, layer in enumerate(self.features[:level]):
            x = layer(x)
            outputs.append(x)
        return outputs

class SelfLoss(nn.Module):
    def __init__(self, encoder, level, device):
        super(SelfLoss, self).__init__()
        self.level = level
        self.feature_extractor = SelfFeatureExtractor(encoder).to(device)
        
    def forward(self, I1, I2):
        features1 = self.feature_extractor(I1, self.level)
        features2 = self.feature_extractor(I2, self.level)
        loss = torch.square(I1 - I2).mean()
        for f1, f2 in zip(features1, features2):
            layer_loss = torch.square(f1 - f2).mean()
            loss += layer_loss
        return loss
