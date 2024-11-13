import torch
from torch import nn
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE
from torchinfo import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if ("Conv" in classname) or ("Linear" in classname):
                nn.init.normal_(m.weight.data, 0.0, 0.01)

            elif "BatchNorm" in classname:
                nn.init.normal_(m.weight.data, 1.0, 0.01)
                nn.init.constant_(m.bias.data, 0)


def downsampleblock(idx, in_channels, out_channels):
    module = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False
        ),
        nn.GELU(),
    )
    if idx > 0:
        return module
    else:
        return module[1:]


def upsampleblock(idx, in_channels, out_channels, output_padding=1):
    module = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=output_padding,
            bias=False,
        ),
        nn.GELU(),
    )
    if idx < -1:
        return module
    else:
        module[-1] = nn.Sigmoid()
        return module


class Encoder(Net):
    def __init__(self, c_in=1, c_hid=16, c_out=1, dim=2, depth=5):
        super().__init__()
        channels = [1 * c_hid, 2 * c_hid, 4 * c_hid, 8 * c_hid, 16 * c_hid, 32 * c_hid]

        if depth > len(channels):
            raise ValueError("Depth exceeds the maximum number of channels defined.")

        self.convolutional_features = nn.Sequential()

        self.latent_mapping = nn.Sequential(
            nn.Flatten(1, -1), nn.Linear(channels[depth - 1], c_out * dim * dim)
        )
        self.output = nn.Unflatten(1, (c_out, dim, dim))

        for i in range(depth):
            in_channels = c_in if i == 0 else channels[i - 1]
            out_channels = channels[i]
            self.convolutional_features.add_module(
                f"{i}", downsampleblock(i, in_channels, out_channels)
            )
        self.weights_init()

    def forward(self, x):
        f = self.convolutional_features(x)
        z = self.latent_mapping(f)
        return self.output(z)


class Decoder(Net):
    def __init__(self, c_in=1, c_hid=16, c_out=1, dim=16, depth=5, is28=False):
        super().__init__()
        channels = [32 * c_hid, 16 * c_hid, 8 * c_hid, 4 * c_hid, 2 * c_hid, 1 * c_hid]
        if depth > len(channels):
            raise ValueError("Depth exceeds the maximum number of channels defined.")

        self.convolutional_features = nn.Sequential()
        self.upsample = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(in_features=c_in * dim * dim, out_features=channels[-depth]),
            nn.GELU(),
            nn.Unflatten(1, (channels[-depth], 1, 1)),
        )

        for i in range(-depth, 0):
            in_channels = channels[i]
            out_channels = c_out if (i == -1) else channels[i + 1]
            output_padding = 0 if (is28 and i == -3) else 1
            self.convolutional_features.add_module(
                f"{depth+i}",
                upsampleblock(i, in_channels, out_channels, output_padding),
            )
        self.output = nn.Sigmoid()
        self.weights_init()

    def forward(self, x):
        l = self.upsample(x)
        return self.convolutional_features(l)


class CFMModel(Net):

    def __init__(self, feature_size, model, nntype):
        super().__init__()
        self.net = model
        self.feature_size = feature_size
        self.nntype = nntype
        self.node = self._create_node(model)
        self.weights_init()

    def _create_node(self, model):
        node_kwargs = {
            "solver": "dopri5",
            "sensitivity": "adjoint",
            "atol": 1e-4,
            "rtol": 1e-4,
        }
        if self.nntype == "mlp":
            node_model = torch_wrapper(model)
        elif self.nntype == "unet":
            node_model = model
        else:
            raise ValueError(f"Unknown model type: {self.nntype}")
        return NeuralODE(node_model, **node_kwargs)

    def forward(self, inputs):
        return self.net(*inputs)

    @torch.no_grad()
    def sample(self, nbr_sample, device):
        self.net.train()
        self.node = self.node.to(device)
        noise = self._generate_noise(nbr_sample, device)

        trajectory = self.node.trajectory(
            noise,
            t_span=torch.linspace(0, 1, 2, device=device),
        )
        return trajectory

    def _generate_noise(self, nbr_sample, device):
        if self.nntype == "unet":
            return torch.randn((nbr_sample, *self.feature_size), device=device)
        else:
            return torch.randn((nbr_sample, self.feature_size), device=device)
