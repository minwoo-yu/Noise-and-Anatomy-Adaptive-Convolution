import torch
import torch.nn as nn
from model import common

def make_model(config):  # T: tiny, S: small, B: base
    if config["model"]["level"] == "S":
        n_layers = 3
    elif config["model"]["level"] == "B":
        n_layers = 8
    print("model loaded!: DNCNN-" + config["model"]["level"] + "+" + config["model"]["backbone"])
    return DnCNN(noise_spec=config["N_augment"], backbone=config["model"]["backbone"], num_layers=n_layers)

class DnCNN(nn.Module):
    def __init__(self, noise_spec, backbone, num_layers=10):
        super(DnCNN, self).__init__()
        self.noise_spec = noise_spec
        self.backbone = backbone
        self.num_layers = num_layers
        self.channel = 64
        if self.noise_spec["N_corr"]:
            self.Correlation = common.NoiseCorrelation(kernel_size=3, patch_size=5, stride=1, padding=1)
            cat_ch = 5**2
        else:
            cat_ch = self.noise_spec["N_ch"]

        layers = []
        if self.backbone == "conv":
            layers.append(nn.Conv2d(in_channels=1 + cat_ch, out_channels=self.channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(self.num_layers - 2):
                layers.append(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(self.channel))
                layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            for i in range(self.num_layers - 2):
                if i % 2 == 0:
                    layers.append(common.MalleConv(in_channel=self.channel, out_channel=self.channel, n_ch=cat_ch, bias=False))
                else:
                    layers.append(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(self.channel))
                layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=3, padding=1))
        self.dncnn = nn.ModuleList(layers)

    def forward(self, x, noise=None):
        inp = x
        if noise is not None and self.noise_spec["N_corr"]:
            noise = self.Correlation(noise)
        x = torch.cat((x, noise), dim=1) if noise is not None and self.backbone == "conv" else x
        for layer in self.dncnn:
            if isinstance(layer, common.MalleConv):
                x = layer(x, noise=noise)
            else:
                x = layer(x)
        return x + inp

    def flops(self, H, W):
        flops = 0
        if self.noise_spec["N_corr"]:
            flops += self.Correlation.flops(self.noise_spec["N_ch"], H, W)
        for layer in self.dncnn:
            if isinstance(layer, nn.Conv2d):
                flops += layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels * layer.out_channels * H * W
            elif isinstance(layer, nn.BatchNorm2d):
                flops += 2 * layer.num_features * H * W
            elif isinstance(layer, common.MalleConv):
                flops += layer.flops(H, W)
        return flops


if __name__ == "__main__":
    import yaml
    import os

    with open(os.path.join("/root/code/yaml/dncnn.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_restoration = make_model(config)
    print("# model_restoration parameters: %.2f M" % (sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))
