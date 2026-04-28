from collections import OrderedDict

import timm
import torch
import torchvision.transforms as T
from torch import nn

from custom_ballspotting.model.layers import EDSGPMIXERLayers, FCLayers
from custom_ballspotting.model.shift import make_temporal_shift


class CustomTDeedModule(nn.Module):
    def __init__(
        self,
        clip_len: int,
        num_actions: int,
        n_layers: int = 2,
        sgp_ks: int = 9,
        sgp_k: int = 4,
        features_model_name: str = "regnety_008",
        temporal_shift_mode: str = "gsf",
        gaussian_blur_ks: int = 5,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.num_actions = num_actions
        self.features_model_name = features_model_name
        self.temporal_shift_mode = temporal_shift_mode
        self.sgp_k = sgp_k
        self.sgp_ks = sgp_ks
        self.n_layers = n_layers

        features = timm.create_model(features_model_name, pretrained=True)
        feat_dim = features.get_classifier().in_features
        features.reset_classifier(0)
        make_temporal_shift(features, clip_len, mode=temporal_shift_mode)

        self._d = feat_dim
        self._features = features
        self.temp_enc = nn.Parameter(
            torch.normal(mean=0, std=1 / clip_len, size=(clip_len, feat_dim))
        )
        self._temp_fine = EDSGPMIXERLayers(
            feat_dim,
            clip_len,
            num_layers=n_layers,
            ks=sgp_ks,
            k=sgp_k,
            concat=True,
        )
        self._pred_fine = FCLayers(feat_dim, num_actions + 1)
        self._pred_displ = FCLayers(feat_dim, 1)
        self.augmentation = T.Compose(
            [
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(gaussian_blur_ks)], p=0.25),
            ]
        )
        self.standardization = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def forward(self, x, inference: bool = False):
        x = x / 255.0
        batch_size, clip_len, channels, height, width = x.shape
        x = x.view(batch_size, clip_len, channels, height, width)
        if not inference:
            for i in range(batch_size):
                x[i] = self.augmentation(x[i])
        for i in range(batch_size):
            x[i] = self.standardization(x[i])

        im_feat = self._features(x.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, self._d
        )
        im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)
        im_feat = self._temp_fine(im_feat)
        return {
            "logits": self._pred_fine(im_feat),
            "displacement": self._pred_displ(im_feat).squeeze(-1),
        }

    def load_backbone(self, model_weight_path: str):
        state = torch.load(model_weight_path, map_location="cpu", weights_only=True)
        features_layers = OrderedDict(
            (k[len("_features.") :], v)
            for k, v in state.items()
            if k.startswith("_features.")
        )
        temp_layers = OrderedDict(
            (k[len("_temp_fine.") :], v)
            for k, v in state.items()
            if k.startswith("_temp_fine.")
        )
        self._features.load_state_dict(features_layers)
        self._temp_fine.load_state_dict(temp_layers)

    def load_all(self, model_weight_path: str):
        state = torch.load(model_weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
