import math

import torch
from torch import nn


class EDSGPMIXERLayers(nn.Module):
    def __init__(
        self, feat_dim, clip_len, num_layers=1, ks=3, k=2, k_factor=2, concat=True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.tot_layers = num_layers * 2 + 1
        self._sgp = nn.ModuleList(
            SGPBlock(feat_dim, kernel_size=ks, k=k, init_conv_vars=0.1)
            for _ in range(self.tot_layers)
        )
        self._pooling = nn.ModuleList(
            nn.AdaptiveMaxPool1d(output_size=math.ceil(clip_len / (k_factor ** (i + 1))))
            for i in range(num_layers)
        )
        self._sgpMixer = nn.ModuleList(
            SGPMixer(
                feat_dim,
                kernel_size=ks,
                k=k,
                init_conv_vars=0.1,
                t_size=math.ceil(clip_len / (k_factor**i)),
                concat=concat,
            )
            for i in range(num_layers)
        )

    def forward(self, x):
        store_x = []
        x = x.permute(0, 2, 1)
        for i in range(self.num_layers):
            x = self._sgp[i](x)
            store_x.append(x)
            x = self._pooling[i](x)
        x = self._sgp[self.num_layers](x)
        for i in range(self.num_layers):
            x = self._sgpMixer[-(i + 1)](x=x, z=store_x[-(i + 1)])
            x = self._sgp[self.num_layers + i + 1](x)
        return x.permute(0, 2, 1)


class SGPBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        kernel_size=3,
        k=1.5,
        group=1,
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        init_conv_vars=0.1,
    ):
        super().__init__()
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        n_out = n_embd if n_out is None else n_out
        n_hidden = 4 * n_embd if n_hidden is None else n_hidden
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)
        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        for layer in [self.psi, self.fc, self.convw, self.convkw, self.global_fc]:
            nn.init.normal_(layer.weight, 0, init_conv_vars)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.ln(x)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = self.fc(out) * phi + (self.convw(out) + self.convkw(out)) * self.psi(out) + out
        out = x + out
        return out + self.mlp(self.gn(out))


class SGPMixer(nn.Module):
    def __init__(
        self,
        n_embd,
        kernel_size=3,
        k=1.5,
        group=1,
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        init_conv_vars=0.1,
        t_size=0,
        concat=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.concat = concat
        n_out = n_embd if n_out is None else n_out
        n_hidden = 4 * n_embd if n_hidden is None else n_hidden
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)
        self.psi1 = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.psi2 = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.convw1 = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.convkw1 = nn.Conv1d(n_embd, n_embd, up_size, padding=up_size // 2, groups=n_embd)
        self.convw2 = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.convkw2 = nn.Conv1d(n_embd, n_embd, up_size, padding=up_size // 2, groups=n_embd)
        self.fc1 = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.global_fc1 = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.fc2 = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.global_fc2 = nn.Conv1d(n_embd, n_embd, 1, groups=n_embd)
        self.upsample = nn.Upsample(size=t_size, mode="linear", align_corners=True)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )
        if self.concat:
            self.concat_fc = nn.Conv1d(n_embd * 6, n_embd, 1, groups=group)
        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        for layer in [
            self.psi1, self.psi2, self.convw1, self.convkw1, self.convw2,
            self.convkw2, self.fc1, self.fc2, self.global_fc1, self.global_fc2,
        ]:
            nn.init.normal_(layer.weight, 0, init_conv_vars)
            nn.init.constant_(layer.bias, 0)
        if self.concat:
            nn.init.normal_(self.concat_fc.weight, 0, init_conv_vars)
            nn.init.constant_(self.concat_fc.bias, 0)

    def forward(self, x, z):
        z = self.ln1(z)
        x = self.ln2(x)
        x = self.upsample(x)
        psi1 = self.psi1(z)
        psi2 = self.psi2(x)
        out1 = (self.convw1(z) + self.convkw1(z)) * psi1
        out2 = (self.convw2(x) + self.convkw2(x)) * psi2
        out3 = self.fc1(z) * torch.relu(self.global_fc1(z.mean(dim=-1, keepdim=True)))
        out4 = self.fc2(x) * torch.relu(self.global_fc2(x.mean(dim=-1, keepdim=True)))
        if self.concat:
            out = self.act(self.concat_fc(torch.cat((out1, out2, out3, out4, z, x), dim=1)))
        else:
            out = out1 + out2 + out3 + out4 + z + x
        return out + self.mlp(self.gn(out))


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        factory_kwargs = {"device": device, "dtype": dtype}
        if affine:
            self.weight = nn.Parameter(torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mu = torch.mean(x, dim=1, keepdim=True)
        sigma = torch.mean((x - mu) ** 2, dim=1, keepdim=True)
        out = (x - mu) / torch.sqrt(sigma + self.eps)
        if self.affine:
            out = out * self.weight + self.bias
        return out


class FCLayers(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(self.dropout(x).reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1
        )


class _GSF(nn.Module):
    def __init__(self, fPlane, num_segments=8, gsf_ch_ratio=100):
        super().__init__()
        fPlane_temp = int(fPlane * gsf_ch_ratio / 100)
        if fPlane_temp % 2 != 0:
            fPlane_temp += 1
        self.fPlane = fPlane_temp
        self.conv3D = nn.Conv3d(self.fPlane, 2, (3, 3, 3), padding=(1, 1, 1), groups=2)
        self.tanh = nn.Tanh()
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=self.fPlane)
        self.relu = nn.ReLU()
        self.channel_conv1 = nn.Conv2d(2, 1, (3, 3), padding=(1, 1))
        self.channel_conv2 = nn.Conv2d(2, 1, (3, 3), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_full):
        x = x_full[:, : self.fPlane, :, :]
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = (
            x.reshape(batchSize, self.num_segments, *shape)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        x_bn_relu = self.relu(self.bn(x))
        gate = self.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)

        x_group1 = x[:, : self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2 :]

        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = torch.roll(y_group1, shifts=-1, dims=2)
        y_group2 = torch.roll(y_group2, shifts=1, dims=2)
        y_group1[:, :, -1] = 0
        y_group2[:, :, 0] = 0

        r_1 = r_group1.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).unsqueeze(3)
        r_2 = r_group2.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).unsqueeze(3)
        y_1 = y_group1.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).unsqueeze(3)
        y_2 = y_group2.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False).unsqueeze(3)

        y_r_1 = torch.cat([y_1, r_1], dim=3).permute(0, 3, 1, 2)
        y_r_2 = torch.cat([y_2, r_2], dim=3).permute(0, 3, 1, 2)

        y_1_weights = self.sigmoid(self.channel_conv1(y_r_1)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        y_2_weights = self.sigmoid(self.channel_conv2(y_r_2)).squeeze(1).unsqueeze(-1).unsqueeze(-1)
        y_group1 = y_group1 * y_1_weights + r_group1 * (1 - y_1_weights)
        y_group2 = y_group2 * y_2_weights + r_group2 * (1 - y_2_weights)

        y_group1 = y_group1.view(
            batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]
        ).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(
            batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]
        ).permute(0, 2, 1, 3, 4, 5)
        y = torch.cat(
            (
                y_group1.contiguous().view(
                    batchSize, self.fPlane // 2, self.num_segments, *shape[1:]
                ),
                y_group2.contiguous().view(
                    batchSize, self.fPlane // 2, self.num_segments, *shape[1:]
                ),
            ),
            dim=1,
        )
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize * self.num_segments, *shape)
        return torch.cat([y, x_full[:, self.fPlane :, :, :]], dim=1)


class _GSM(nn.Module):
    def __init__(self, fPlane, num_segments=3):
        super().__init__()
        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.tanh = nn.Tanh()
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        x = x.view(batch_size, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        gate = self.tanh(self.conv3D(self.relu(self.bn(x))))
        y1 = gate[:, 0].unsqueeze(1) * x[:, : self.fPlane // 2]
        y2 = gate[:, 1].unsqueeze(1) * x[:, self.fPlane // 2 :]
        r1 = x[:, : self.fPlane // 2] - y1
        r2 = x[:, self.fPlane // 2 :] - y2
        y1 = torch.cat((y1[:, :, 1:], torch.zeros_like(y1[:, :, :1])), dim=2) + r1
        y2 = torch.cat((torch.zeros_like(y2[:, :, :1]), y2[:, :, :-1]), dim=2) + r2
        y1 = y1.view(batch_size, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y2 = y2.view(batch_size, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y = torch.cat(
            (
                y1.contiguous().view(batch_size, self.fPlane // 2, self.num_segments, *shape[1:]),
                y2.contiguous().view(batch_size, self.fPlane // 2, self.num_segments, *shape[1:]),
            ),
            dim=1,
        )
        return y.permute(0, 2, 1, 3, 4).contiguous().view(batch_size * self.num_segments, *shape)
