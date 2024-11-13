import torch
from torch import nn
from math import ceil
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE


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

    def forward(self, x):
        raise NotImplementedError


class MLPEncoder(Net):
    def __init__(self, input_size, unet=False):
        super().__init__()
        self.net = self.__get_net(input_size, unet)
        self.weights_init()

    def __get_net(self, input_size, unet):
        if input_size == 28:
            net = nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(28 * 28, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 64),
            )
            if unet:
                net.add_module("8", nn.Unflatten(1, (1, 8, 8)))

        elif input_size == 32:
            net = nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(3 * 32 * 32, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 128),
            )
            if unet:
                net.add_module("8", nn.Unflatten(1, (2, 8, 8)))

        elif input_size == 64:
            net = nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(3 * 64 * 64, 4096),
                nn.GELU(),
                nn.Linear(4096, 1024),
                nn.GELU(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 256),
            )
            if unet:
                net.add_module("8", nn.Unflatten(1, (1, 16, 16)))
        else:
            raise ValueError(f"input_size = {input_size} not in [28, 32, 64]")
        return net

    def forward(self, x):
        return self.net(x)


class MLPDecoder(Net):
    def __init__(self, input_size, unet=False):
        super().__init__()
        self.net = self.__get_net(input_size, unet)
        self.weights_init()

    def __get_net(self, input_size, unet):
        if input_size == 28:
            net = nn.Sequential(
                nn.Linear(64, 128),
                nn.GELU(),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 1 * 28 * 28),
                nn.Sigmoid(),
                nn.Unflatten(1, (1, 28, 28)),
            )
            if unet:
                net.insert(0, nn.Flatten(1, -1))

        elif input_size == 32:
            net = nn.Sequential(
                nn.Linear(128, 256),
                nn.GELU(),
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Linear(1024, 3 * 32 * 32),
                nn.Sigmoid(),
                nn.Unflatten(1, (3, 32, 32)),
            )
            if unet:
                net.insert(0, nn.Flatten(1, -1))

        elif input_size == 64:
            net = nn.Sequential(
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Linear(1024, 4096),
                nn.GELU(),
                nn.Linear(4096, 3 * 64 * 64),
                nn.Sigmoid(),
                nn.Unflatten(1, (3, 64, 64)),
            )
            if unet:
                net.insert(0, nn.Flatten(1, -1))
        else:
            raise ValueError(f"input_size = {input_size} not in [28, 32, 64]")
        return net

    def forward(self, x):
        return self.net(x)


class CNNEncoder(Net):
    def __init__(self, input_size, c_in=1, c_hid=8, c_out=1, w_out=8, unet=False):
        super().__init__()
        _w_hid = ceil(input_size / 16)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 4 * c_hid, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(4 * c_hid, 8 * c_hid, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(_w_hid * _w_hid * 8 * c_hid, c_out * w_out * w_out),
        )
        if unet:
            self.net.add_module("10", nn.Unflatten(1, (c_out, w_out, w_out)))
        self.weights_init()

    def forward(self, x):
        return self.net(x)


class CNNDecoder(Net):
    def __init__(
        self, input_size, c_in=1, w_in=8, c_hid=8, c_out=1, unet=False,
    ):
        super().__init__()
        _w_hid = ceil(input_size / 16)
        self.net = nn.Sequential(
            nn.Linear(c_in * w_in * w_in, _w_hid * _w_hid * 8 * c_hid),
            nn.GELU(),
            nn.Unflatten(1, (8 * c_hid, _w_hid, _w_hid)),
            nn.ConvTranspose2d(
                8 * c_hid,
                4 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                4 * c_hid,
                2 * c_hid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=0 if (input_size == 28) else 1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                2 * c_hid, c_hid, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                c_hid, c_out, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )
        if unet:
            self.net.insert(0, nn.Flatten(1, -1))
        self.weights_init()

    def forward(self, x):
        return self.net(x)
