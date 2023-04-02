# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence
import matplotlib.pyplot as plt
import omegaconf
import torch
import torch.nn as nn
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
# from pytorch_wavelets import DTCWTForward, DTCWTInverse
# from solo.data.pretrain_dataloader import DWT3D
from pytorch_wavelets import DWTForward


class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]
        # X = batch[1]
        # Y = batch[2]
        
        xfm = DWTForward(J=1, mode='symmetric', wave='db1').to(self.device)

        Y1, Y2 , Y3, Y4, Y5, Y6, Y7, Y8 = xfm(batch[1][0]), xfm(batch[1][1]), xfm(batch[1][2]), xfm(batch[1][3]), xfm(batch[1][4]), xfm(batch[1][5]), xfm(batch[1][6]), xfm(batch[1][7])
        # print(Y1[0].shape)
        # print(Y2[0].shape)
        # print(Y3[1][0][:,:,0].shape)
        # print(Y4[1][0][:,:,0].shape)
        # print(Y5[1][0][:,:,1].shape)
        # print(Y6[1][0][:,:,1].shape)
        # print(Y7[1][0][:,:,2].shape)
        # print(Y8[1][0][:,:,2].shape)
        # m = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # a = m(Y1[0])
        # b = m(Y2[0])
        # c = m(Y3[1][0][:,:,0])
        # d = m(Y4[1][0][:,:,0])
        # e = m(Y5[1][0][:,:,1])
        # f = m(Y6[1][0][:,:,1])
        # g = m(Y7[1][0][:,:,2])
        # h = m(Y8[1][0][:,:,2])
        # print(a.shape)
        # print(b.shape)  
        # print(c.shape)
        # print(d.shape)
        # print(e.shape)
        # print(f.shape)
        # print(g.shape)
        # print(h.shape)
       
        # mod_batch_a = [batch[0],[Y1[0],Y2[0]],batch[2]]
        # mod_batch_h = [batch[0],[Y1[1][0][:,:,0],Y2[1][0][:,:,0]],batch[2]]
        # mod_batch_v = [batch[0],[Y1[1][0][:,:,1],Y2[1][0][:,:,1]],batch[2]]
        # mod_batch_d = [batch[0],[Y1[1][0][:,:,2],Y2[1][0][:,:,2]],batch[2]]
    
        # mod_batch_a = [batch[0],[Y1[0],Y2[0]],batch[2]]
        # mod_batch_h = [batch[0],[Y3[1][0][:,:,0],Y4[1][0][:,:,0]],batch[2]]
        # mod_batch_v = [batch[0],[Y5[1][0][:,:,1],Y6[1][0][:,:,1]],batch[2]]
        # mod_batch_d = [batch[0],[Y7[1][0][:,:,2],Y8[1][0][:,:,2]],batch[2]]

        batch = [batch[0],[Y1[0],Y2[0],Y3[1][0][:,:,0],Y4[1][0][:,:,0],Y5[1][0][:,:,1],Y6[1][0][:,:,1],Y7[1][0][:,:,2],Y8[1][0][:,:,2]],batch[2]]
        # mod_batch_a = [batch[0],[a,b],batch[2]]
        # mod_batch_h = [batch[0],[c,d],batch[2]]
        # mod_batch_v = [batch[0],[e,f],batch[2]]
        # mod_batch_d = [batch[0],[g,h],batch[2]]

        out = super().training_step(batch, batch_idx)        
        # out_a = super().training_step(mod_batch_a, batch_idx)
        # out_h = super().training_step(mod_batch_h, batch_idx)
        # out_v = super().training_step(mod_batch_v, batch_idx)
        # out_d = super().training_step(mod_batch_d, batch_idx)

        class_loss = out["loss"]
        # print("LOSS",len(out["z"]))
        # z = out["z"]
        z_a = torch.cat((out["z"][0],out["z"][1]))
        z_h = torch.cat((out["z"][2],out["z"][3]))
        z_v = torch.cat((out["z"][4],out["z"][5]))
        z_d = torch.cat((out["z"][6],out["z"][7]))
        # z = torch.cat(out["z"])
        # print("LOSS1",z.shape)
        # z_a = torch.cat(out_a["z"])
        # print(z_a.shape)
        # z_h = torch.cat(out_h["z"])
        # z_v = torch.cat(out_v["z"])
        # z_d = torch.cat(out_d["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        # indexes = indexes.repeat(n_augs)
        indexes = indexes.repeat(2)

        # nce_loss = simclr_loss_func(
        #     z,
        #     indexes=indexes,
        #     temperature=self.temperature,
        # )

        approx_loss = simclr_loss_func(
            z_a,
            indexes=indexes,
            temperature=self.temperature,
        )

        hzt_loss = simclr_loss_func(
            z_h,
            indexes=indexes,
            temperature=self.temperature,
        )

        ver_loss = simclr_loss_func(
            z_v,
            indexes=indexes,
            temperature=self.temperature,
        )

        dia_loss = simclr_loss_func(
            z_d,
            indexes=indexes,
            temperature=self.temperature,
        )

        total = approx_loss + hzt_loss + ver_loss + dia_loss
        final = total + class_loss
        # self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        self.log("train_final_loss", final, on_epoch=True, sync_dist=True)

        return final
