import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import leaptorch
from einops import rearrange
import spatial_correlation_sampler
from flying_conv.flying_conv import _triton_flying_conv2d
from flying_conv.flying_conv import _torch_flying_conv2d


def filtering(data, kernel):
    if kernel is not None:
        batch, ch, row, col = data.shape
        data = data.view(-1, 1, row, col)
        data = F.conv2d(data, kernel, padding=(0, 2))
        return data.view(batch, ch, row, col)
    else:
        return data


def geometry_setting(projector, metadata, num_row):
    projector.set_fanbeam(
        int(metadata[0, 5].item()),  ## num_view
        num_row,
        int(metadata[0, 6].item()),  ## num_det
        1,
        metadata[0, 4].item(),  ## det_interval
        0,
        0.5 * (metadata[0, 6].item() - 1),
        np.linspace(0, 360, int(metadata[0, 5].item()), endpoint=False, dtype="float32"),
        metadata[0, 1].item(),  ## distance source to patient
        metadata[0, 2].item(),  ## distance source to detector
    )
    projector.set_volume(
        512,
        512,
        num_row,
        voxelWidth=metadata[0, 3],  ## recon_interval
        voxelHeight=1,
    )
    projector.allocate_batch_data()
    return projector


class NoiseRecon(nn.Module):
    def __init__(self, config):
        super(NoiseRecon, self).__init__()
        self.config = config
        self.mode = config["dataset"]["mode"]
        self.N_in = float(config["dataset"]["N_in"])
        self.sigma_e = config["dataset"]["sigma_e"]
        if isinstance(config["dataset"]["dose_level"], list):
            self.dose_level = torch.Tensor(config["dataset"]["dose_level"])
        else:
            self.dose_level = torch.Tensor([config["dataset"]["dose_level"]])

        self.projector = leaptorch.Projector(use_gpu=True, gpu_device=torch.device(0), batch_size=1)
        self.device = torch.device(0)
        if self.config["dataset"]["blur"]:
            kernel_q = torch.tensor([0.22, 0.37, 0.73, 0.37, 0.22])
            kernel_e = torch.tensor([0.01, 0.01, 0.98, 0.01, 0.01])
            self.kernel_q = kernel_q.view(1, 1, 1, 5).cuda()
            self.kernel_e = kernel_e.view(1, 1, 1, 5).cuda()
        else:
            self.kernel_q = None
            self.kernel_e = None

    def forward(self, clean_proj, metadata, ld_proj=None, N_ch=0, dose_level=None):
        batch, _, view, col = clean_proj.shape
        clean_proj = rearrange(clean_proj, "b r v c -> b v r c").contiguous()  # [batch, view, row, col]
        # Check if metadata values vary across the batch
        if metadata.shape[0] > 1 and torch.any(metadata[1:] != metadata[0]):
            print("Warning: Metadata values vary across the batch. Using the first value for all samples.")
        u_water = 0.0192867 if int(metadata[0, 0]) == 120 else 0.0205888
        
        if dose_level is None:
            dose_level = self.dose_level[torch.randint(0, len(self.dose_level), (batch,))].view(batch, 1, 1, 1).to(clean_proj.device)
        else:
            dose_level = torch.Tensor([dose_level]).view(1, 1, 1, 1).to(clean_proj.device)

        with torch.no_grad():
            clean_proj = (torch.exp(-clean_proj) * self.N_in).clamp(min=1)
            if ld_proj is None:
                ## noise insertion
                q_normal = filtering(torch.poisson(clean_proj) - clean_proj, self.kernel_q)
                e_normal = filtering(self.sigma_e * torch.randn_like(clean_proj), self.kernel_e)
                nd_proj = (clean_proj + q_normal + e_normal).clamp(min=1)

                std_sino = torch.sqrt(nd_proj)
                q_noise = filtering(std_sino * torch.randn_like(clean_proj), self.kernel_q)
                e_noise = filtering(self.sigma_e * torch.randn_like(clean_proj), self.kernel_e)

                ## dose addition
                a, b = torch.sqrt(1.0 / dose_level - 1.0), torch.sqrt(1.0 / dose_level + 1.0)
                ld_proj = nd_proj + a * (q_noise + b * e_noise)
            else:
                ld_proj = rearrange(ld_proj, "b r v c -> b v r c").contiguous()
                ld_proj = torch.exp(-ld_proj) * self.N_in
            ld_proj = ld_proj.clamp(min=1)

            if N_ch != 0:
                std_ld = torch.sqrt(ld_proj)
                q_lower = filtering(std_ld * torch.randn((batch, view, N_ch, col), device=self.device), self.kernel_q)
                e_lower = filtering(self.sigma_e * torch.randn((batch, view, N_ch, col), device=self.device), self.kernel_e)
                lower_proj = ld_proj + torch.sqrt(1.0 / dose_level) * q_lower + e_lower / dose_level  # for N2N or N2C
                # lower_proj = ld_proj + self.a * (q_lower + self.b * e_lower)  # for N2F
                lower_proj = lower_proj.clamp(min=1)

                ld_proj = torch.cat(((ld_proj / self.N_in), lower_proj / ld_proj), 2)
            else:
                ld_proj = ld_proj / self.N_in

            ## reconstruction
            self.projector = geometry_setting(self.projector, metadata, (2 + N_ch) * batch)
            if self.training and self.mode == "N2N":  ## Noise2Noise
                in_proj = (nd_proj - 1 / a * (q_noise + 1 / b * e_noise)).clamp(min=1)
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(in_proj / self.N_in)), 2)
                data_proj = rearrange(data_proj, "b v r c -> 1 v (b r) c").contiguous()
                recon = self.projector.fbp(data_proj)
            elif self.training and self.mode == "N2F":  ## Noise2Full
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(nd_proj / self.N_in)), 2)
                data_proj = rearrange(data_proj, "b v r c -> 1 v (b r) c").contiguous()
                recon = self.projector.fbp(data_proj)
            else:  # Noise2Clean or validation / test mode
                data_proj = torch.cat((-torch.log(ld_proj), -torch.log(clean_proj / self.N_in)), 2)
                data_proj = rearrange(data_proj, "b v r c -> 1 v (b r) c").contiguous()
                recon = self.projector.fbp(data_proj)
            recon = recon.view(batch, -1, 512, 512).contiguous()
        return (
            recon[:, :1] * 1000 / u_water - 1000,  # LDCT
            None if N_ch == 0 else recon[:, 1:-1] * 1000 / u_water,  # noise
            recon[:, -1:] * 1000 / u_water - 1000,  # GT
        )


class NoiseCorrelation(nn.Module):
    def __init__(self, kernel_size, patch_size, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(NoiseCorrelation, self).__init__()
        self.kernel_size = kernel_size
        self.correlation_sampler = spatial_correlation_sampler.SpatialCorrelationSampler(
            kernel_size=kernel_size,
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            dilation_patch=dilation_patch,
        )
        self.patch_size = patch_size

    def forward(self, noise):
        batch, ch, height, width = noise.shape
        with torch.no_grad():
            noise_corr = self.correlation_sampler(noise, noise) / ch / self.kernel_size**2
        noise_corr = noise_corr.view(batch, -1, height, width)
        return noise_corr

    def flops(self, C, H, W):
        flops = H * W * self.kernel_size**2 * self.patch_size**2 * C
        return flops


def get_recon_patch(input, target, patch_coord, position, noise=None):
    B, H, W, _ = patch_coord.shape
    patch_coord = patch_coord / (512 / H) + position.flip(-1).view(B, 1, 1, 2)
    input = F.grid_sample(input, patch_coord, mode="nearest", align_corners=False)
    target = F.grid_sample(target, patch_coord, mode="nearest", align_corners=False)
    if noise is not None:
        noise = F.grid_sample(noise, patch_coord, mode="nearest", align_corners=False)
    return input, noise, target


class Predictor(nn.Module):
    def __init__(self, channel):
        super(Predictor, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        # self.conv1.weight.data.mul_(1 / init_div)
        # self.conv1.bias.data.mul_(1 / init_div)
        # self.conv2.weight.data.mul_(1 / init_div)
        # self.conv2.bias.data.mul_(1 / init_div)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.act(x)
        x = self.conv2(x)
        return inp + x

    def flops(self, H, W):
        flops = H * W * self.channel * self.channel * 3 * 3 + H * W * self.channel * self.channel * 3 * 3
        return flops


class MalleConv(nn.Module):
    def __init__(self, in_channel, out_channel, n_ch=0, bias=True):
        super(MalleConv, self).__init__()
        self.kernel_size = (3, 3)
        # self.kernel_size = (1, 1)
        self.stride = (1, 1)
        self.padding = (1, 1)
        # self.padding = (0, 0)
        self.dilation = (1, 1)
        self.scale = (8, 8)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_ch = n_ch
        self.bias = bias

        self.kernel_tot = self.kernel_size[0] * self.kernel_size[1] + self.bias
        self.pool = nn.AvgPool2d(4, 4)
        self.block_size = 4

        self.predictor_a = []
        for i in range(self.block_size):
            self.predictor_a.append(Predictor(in_channel))
            if i == self.block_size // 2 - 1:
                self.predictor_a.append(nn.AvgPool2d(2, 2))
        self.predictor_a = nn.Sequential(*self.predictor_a)

        if n_ch != 0:
            self.predictor_n = [nn.Conv2d(n_ch, in_channel, kernel_size=3, padding=1)]
            for i in range(self.block_size):
                self.predictor_n.append(Predictor(in_channel))
                if i == self.block_size // 2 - 1:
                    self.predictor_n.append(nn.AvgPool2d(2, 2))
            self.predictor_n = nn.Sequential(*self.predictor_n)

        self.expander = nn.Conv2d(in_channel, in_channel * self.kernel_tot, kernel_size=1)
        # self.expander.weight.data.mul_(1/2)
        # self.expander.bias.data.mul_(1/2)
        # self.activation = nn.LeakyReLU(inplace=True)
        self.pwconv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, noise=None):
        B, C, H, W = x.shape
        feat_a = self.predictor_a(self.pool(x))
        if noise is not None:
            feat_n = self.predictor_n(self.pool(noise))
            feat_a = feat_a * feat_n
        weights = self.expander(feat_a).view(B, C, -1, H // self.scale[0], W // self.scale[1]).permute(0, 1, 3, 4, 2)  # [B C H W K]
        weights = self.tanh(weights)
        if not self.bias:
            weight, bias = weights.contiguous(), None
        else:
            weight, bias = weights[..., :-1].contiguous(), weights[..., -1].contiguous()
        out = _triton_flying_conv2d.apply(x, weight, bias, self.kernel_size, self.stride, self.padding, self.dilation, self.scale)
        # out = _torch_flying_conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.scale)
        # out = self.activation(out)
        out = self.pwconv(out)
        return out

    def flops(self, H, W):
        flops = 0
        for i in range(self.block_size // 2):
            flops += self.predictor_a[i].flops(H // 4, W // 4)
            flops += self.predictor_a[i + self.block_size // 2 + 1].flops(H // 8, W // 8)
        if self.n_ch != 0:
            flops += self.n_ch * self.in_channel * 3 * 3 * H // 4 * W // 4
            for i in range(self.block_size // 2):
                flops += self.predictor_n[i + 1].flops(H // 4, W // 4)
                flops += self.predictor_n[i + 1 + self.block_size // 2 + 1].flops(H // 8, W // 8)
        flops += self.in_channel * self.in_channel * self.kernel_tot * H // 8 * W // 8  # expander
        flops += self.in_channel * self.kernel_tot * H * W * 5  # flying conv
        flops += self.in_channel * self.out_channel * H * W  # pwconv

        return flops
