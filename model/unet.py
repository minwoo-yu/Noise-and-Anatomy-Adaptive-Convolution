import torch.nn as nn
import torch.nn.functional as F
import torch
from model import common


def make_model(config):  # S: Small, B: Base
    if config["model"]["level"] == "S":
        dim = 16
    elif config["model"]["level"] == "B":
        dim = 32
    print("model loaded!: UNet-" + config["model"]["level"] + "+" + config["model"]["backbone"])
    return UNet(noise_spec=config["N_augment"], backbone=config["model"]["backbone"], dim=dim)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, noise_ch=0, strides=1, backbone="conv"):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.noise_ch = noise_ch
        self.backbone = backbone

        self.block = []
        if self.backbone == "conv":
            self.block.append(nn.Conv2d(in_channel + noise_ch, out_channel, kernel_size=3, stride=strides, padding=1))
            self.block.append(nn.LeakyReLU(inplace=True))
            self.block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1))
            self.conv11 = nn.Conv2d(in_channel + noise_ch, out_channel, kernel_size=1, stride=strides, padding=0)
        else:
            self.block.append(common.MalleConv(in_channel, out_channel, n_ch=noise_ch, bias=True))
            self.block.append(nn.LeakyReLU(inplace=True))
            self.block.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1))
            self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

        self.block.append(nn.LeakyReLU(inplace=True))
        self.block = nn.ModuleList(self.block)

    def forward(self, x, noise=None):
        if noise is not None and self.backbone == "conv" and self.noise_ch > 0:
            x = torch.cat((x, noise), dim=1)
        out1 = x
        for layer in self.block:
            if isinstance(layer, common.MalleConv):
                out1 = layer(out1, noise=noise)
            else:
                out1 = layer(out1)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops


class UNet(nn.Module):
    def __init__(self, noise_spec, backbone, dim=32):
        super(UNet, self).__init__()
        self.noise_spec = noise_spec
        self.backbone = backbone
        self.dim = dim

        if self.noise_spec["N_corr"]:
            self.Correlation = common.NoiseCorrelation(kernel_size=3, patch_size=5, stride=1, padding=1)
            cat_ch = 5**2
        else:
            cat_ch = self.noise_spec["N_ch"]

        self.conv0 = nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1)

        self.ConvBlock1 = ConvBlock(dim, dim, noise_ch=cat_ch, strides=1, backbone=backbone)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = ConvBlock(dim, dim * 2, noise_ch=cat_ch, strides=1, backbone=backbone)
        self.pool2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = ConvBlock(dim * 2, dim * 4, noise_ch=cat_ch, strides=1, backbone=backbone)
        self.pool3 = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = ConvBlock(dim * 4, dim * 8, noise_ch=cat_ch, strides=1, backbone=backbone)
        self.pool4 = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = ConvBlock(dim * 8, dim * 16, noise_ch=cat_ch, strides=1, backbone=backbone)

        self.upsample6 = nn.ConvTranspose2d(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = ConvBlock(dim * 16, dim * 8, strides=1, noise_ch=cat_ch if self.backbone != "conv" else 0, backbone=backbone)

        self.upsample7 = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = ConvBlock(dim * 8, dim * 4, strides=1, noise_ch=cat_ch if self.backbone != "conv" else 0, backbone=backbone)

        self.upsample8 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = ConvBlock(dim * 4, dim * 2, strides=1, noise_ch=cat_ch if self.backbone != "conv" else 0, backbone=backbone)

        self.upsample9 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = ConvBlock(dim * 2, dim, strides=1, noise_ch=cat_ch if self.backbone != "conv" else 0, backbone=backbone)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, noise=None):
        if noise is not None:
            if self.noise_spec["N_corr"]:
                noise1 = self.Correlation(noise)
            else:
                noise1 = noise
            noise2 = F.avg_pool2d(noise1, 2, 2)
            noise3 = F.avg_pool2d(noise2, 2, 2)
            noise4 = F.avg_pool2d(noise3, 2, 2)
            noise5 = F.avg_pool2d(noise4, 2, 2)
        else:
            noise1, noise2, noise3, noise4, noise5 = None, None, None, None, None

        conv0 = self.conv0(x)
        conv1 = self.ConvBlock1(conv0, noise=noise1)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1, noise=noise2)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2, noise=noise3)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3, noise=noise4)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4, noise=noise5)

        up6 = self.upsample6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6, noise=noise4)

        up7 = self.upsample7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7, noise=noise3)

        up8 = self.upsample8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8, noise=noise2)

        up9 = self.upsample9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9, noise=noise1)

        conv10 = self.conv10(conv9)
        return conv10 + x

    def flops(self, H, W):  # its MACs!
        flops = 0
        if self.noise_spec["N_corr"]:
            flops += self.Correlation.flops(self.noise_spec["N_ch"], H, W)
        flops += H * W * self.conv0.in_channels * self.conv0.out_channels * 3 * 3
        flops += self.ConvBlock1.flops(H, W)
        flops += H / 2 * W / 2 * self.dim * self.dim * 4 * 4
        flops += self.ConvBlock2.flops(H / 2, W / 2)
        flops += H / 4 * W / 4 * self.dim * 2 * self.dim * 2 * 4 * 4
        flops += self.ConvBlock3.flops(H / 4, W / 4)
        flops += H / 8 * W / 8 * self.dim * 4 * self.dim * 4 * 4 * 4
        flops += self.ConvBlock4.flops(H / 8, W / 8)
        flops += H / 16 * W / 16 * self.dim * 8 * self.dim * 8 * 4 * 4

        flops += self.ConvBlock5.flops(H / 16, W / 16)

        flops += H / 16 * W / 16 * self.dim * 16 * self.dim * 8 * 2 * 2
        flops += self.ConvBlock6.flops(H / 8, W / 8)
        flops += H / 8 * W / 8 * self.dim * 8 * self.dim * 4 * 2 * 2
        flops += self.ConvBlock7.flops(H / 4, W / 4)
        flops += H / 4 * W / 4 * self.dim * 4 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock8.flops(H / 2, W / 2)
        flops += H / 2 * W / 2 * self.dim * 2 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock9.flops(H, W)

        flops += H * W * self.conv10.in_channels * self.conv10.out_channels * 3 * 3
        return flops


if __name__ == "__main__":
    import yaml
    import os

    with open(os.path.join("/root/code/yaml/unet.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_restoration = make_model(config)
    # print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("# model_restoration parameters: %.2f M" % (sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))
