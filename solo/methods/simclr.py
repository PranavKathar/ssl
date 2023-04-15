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
import torchvision
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
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
        # print(len(batch[1]))
        # X = batch[1]
        # xfm = DWTForward(J=1, mode='symmetric', wave='haar').to(self.device)
        # Y1, Y2  = xfm(batch[1][0]), xfm(batch[1][1])
        # print(type(Y1))
        # Y1[0] = trans(Y1[0])
        # Y2[0] = trans(Y2[0])
        # Y1[1][0] = trans(Y1[1][0])
        # Y2[1][0] = trans(Y2[1][0])
        # m = nn.Upsample(scale_factor=2, mode='nearest')
        # a = [X[0],X[1],m(Y1[0]),m(Y2[0]),m(Y1[1][0][:,:,0]),m(Y2[1][0][:,:,0]),m(Y1[1][0][:,:,1]),m(Y2[1][0][:,:,1]),m(Y1[1][0][:,:,2]),m(Y2[1][0][:,:,2])]

        # a = [X[0],X[1],Y1[0],Y2[0],Y1[1][0][:,:,0],Y2[1][0][:,:,0],Y1[1][0][:,:,1],Y2[1][0][:,:,1],Y1[1][0][:,:,2],Y2[1][0][:,:,2]]
        # custom_transforms = [torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))]
        # trans = torchvision.transforms.Compose(custom_transforms)
        # a = [trans(X[0]),trans(X[1]),trans(Y1[0]),trans(Y2[0]),trans(Y1[1][0][:,:,0]),trans(Y2[1][0][:,:,0]),trans(Y1[1][0][:,:,1]),trans(Y2[1][0][:,:,1]),trans(Y1[1][0][:,:,2]),trans(Y2[1][0][:,:,2])]
        # batch = [batch[0],a,batch[2]]
        
        # fig = plt.figure(figsize=(4, 2))
        # ax1 = fig.add_subplot(2, 4, 1)
        # ax2 = fig.add_subplot(2, 4, 2)
        # ax3 = fig.add_subplot(2, 4, 3)
        # ax4 = fig.add_subplot(2, 4, 4)
        # ax5 = fig.add_subplot(2, 4, 5)
        # ax6 = fig.add_subplot(2, 4, 6)
        # ax7 = fig.add_subplot(2, 4, 7)
        # ax8 = fig.add_subplot(2, 4, 8)
        # def convert_to_pil(tensor):
        #     # mt = torchvision.transforms.ToPILImage()
        #     custom_transforms = [torchvision.transforms.Normalize(mean=[-0.4914, -0.4822,-0.4465], std=[1/0.2470, 1/0.2435,1/0.2616])]
        #     inv_trans = torchvision.transforms.Compose(custom_transforms)
        #     tensor = inv_trans(tensor)
        #     # img = mt(tensor)
        #     img = torchvision.transforms.functional.convert_image_dtype(image= tensor,dtype=torch.float64)
        #     return img.numpy().transpose((1,2,0))
        # print(convert_to_pil(a[0][0].cpu()).shape)
        # print(convert_to_pil(a[0][0].cpu()))
        # ax1.imshow(convert_to_pil(a[0][0].cpu()),interpolation='nearest')
        # ax1.axis('off')
        # ax1.set_title("aug1_approx_component")
        # ax2.imshow(convert_to_pil(a[2][0].cpu()),interpolation='nearest')
        # ax2.axis('off')
        # ax2.set_title("aug1_hzt_component")
        # ax3.imshow(convert_to_pil(a[4][0].cpu()),interpolation='nearest')
        # ax3.axis('off')
        # ax3.set_title("aug1_vrt_component")
        # ax4.imshow(convert_to_pil(a[6][0].cpu()),interpolation='nearest')
        # ax4.axis('off')
        # ax4.set_title("aug1_dgn_component")
        # ax5.imshow(convert_to_pil(a[1][0].cpu()),interpolation='nearest')
        # ax5.axis('off')
        # ax5.set_title("aug2_approx_component")
        # ax6.imshow(convert_to_pil(a[3][0].cpu()),interpolation='nearest')
        # ax6.axis('off')
        # ax6.set_title("aug2_hzt_component")
        # ax7.imshow(convert_to_pil(a[5][0].cpu()),interpolation='nearest')
        # ax7.axis('off')
        # ax7.set_title("aug2_vrt_component")
        # ax8.imshow(convert_to_pil(a[7][0].cpu()),interpolation='nearest')
        # ax8.axis('off')
        # ax8.set_title("aug2_dgn_component")
        # plt.savefig('DWT_compare.png',dpi=100) 
        # plt.show()

 
        out = super().training_step(batch, batch_idx)       
        class_loss = out["loss"]
        # z = out["z"]
        # z_da = torch.cat((out["z"][0],out["z"][1]))
        # z_a = torch.cat((out["z"][2],out["z"][3]))
        # z_h = torch.cat((out["z"][4],out["z"][5]))
        # z_v = torch.cat((out["z"][6],out["z"][7]))
        # z_d = torch.cat((out["z"][8],out["z"][9]))

        z_a = torch.cat((out["z"][0],out["z"][1]))
        z_h = torch.cat((out["z"][2],out["z"][3]))
        z_v = torch.cat((out["z"][4],out["z"][5]))
        z_d = torch.cat((out["z"][6],out["z"][7]))

        # print("LOSS1",z.shape)

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        # indexes = indexes.repeat(n_augs)
        indexes = indexes.repeat(2)

        # nce_loss = simclr_loss_func(
        #     z_da,
        #     indexes=indexes,
        #     temperature=self.temperature,
        # )
        nce_loss = 0

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
        l = 1 #lambda
        total = approx_loss + hzt_loss + ver_loss + dia_loss
        total_avg = total/4
        final = total*l + nce_loss
        # avg = final/5
        avg = final/4
        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
        self.log("train_class_loss", class_loss, on_epoch=True, sync_dist=True)
        self.log("Approx_comp_loss", approx_loss, on_epoch=True, sync_dist=True)
        self.log("Horizontal_comp_loss", hzt_loss, on_epoch=True, sync_dist=True)
        self.log("Vertical_comp_loss",ver_loss, on_epoch=True, sync_dist=True)
        self.log("Diagonal_comp_loss", dia_loss, on_epoch=True, sync_dist=True)
        self.log("Total_comp_loss", total, on_epoch=True, sync_dist=True)
        self.log("Total/4_comp_loss", total_avg, on_epoch=True, sync_dist=True)
        self.log("Total*lambda_comp_loss", total*l, on_epoch=True, sync_dist=True)
        self.log("train_final_loss", final, on_epoch=True, sync_dist=True)
        self.log("train_final/5_loss", avg, on_epoch=True, sync_dist=True)
        self.log("final/5+class_loss", avg + class_loss, on_epoch=True, sync_dist=True)

        return avg + class_loss
    
##########################ORIGINAL#####################################

        # indexes = batch[0]

        # out = super().training_step(batch, batch_idx)
        # class_loss = out["loss"]
        # z = torch.cat(out["z"])

        # # ------- contrastive loss -------
        # n_augs = self.num_large_crops + self.num_small_crops
        # indexes = indexes.repeat(n_augs)

        # nce_loss = simclr_loss_func(
        #     z,
        #     indexes=indexes,
        #     temperature=self.temperature,
        # )

        # self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        # return nce_loss + class_loss

