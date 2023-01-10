### Dependencies
# Base Dependencies
import os
import pickle
import sys
import torch
import torch.nn.functional as F
from HIPT_4K.hipt_4k import HIPT_4K

from models.model_utils import *
from models.model_hierarchical_mil import HIPT_LGP_FC

print("device:", torch.cuda.get_device_name())

print(torch.__version__)

def get_vitWSI(pretrained_weights, device=torch.device('cuda:0')):

    state_dict = torch.load(pretrained_weights)

    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # try to load in vit wsi
    vitWSI = HIPT_LGP_FC(freeze_4k=True, pretrain_4k=True, freeze_WSI=True, pretrain_WSI=True, n_classes=2)
    vitWSI.load_state_dict(state_dict, strict=False)
    vitWSI.eval()
    vitWSI.to(device)

    print(vitWSI)

    return vitWSI

class HIPT(torch.nn.Module):
    def __init__(self, 
		model256_path: str = 'HIPT_4K/Checkpoints/vit256_small_dino.pth',
		model4k_path: str = 'HIPT_4K/Checkpoints/vit4k_xs_dino.pth',
		modelWSI_path: str = "2-Weakly-Supervised-Subtyping/results/model.pt",
        device="cuda:0"):
        
        super().__init__()
        # self.model256 = get_vit256(pretrained_weights=model256_path).to(device256)
        # self.model4k = get_vit4k(pretrained_weights=model4k_path).to(device4k)
        self.hipt_4k = HIPT_4K(model256_path=model256_path, model4k_path=model4k_path, device256=device, device4k=device)
        self.hipt_vit_WSI = get_vitWSI(pretrained_weights=modelWSI_path, device=device).to(device)
        self.device = device


    def forward(self, h_4096):
        """_summary_

        Args:
            x (_type_): embeddings [M, 192]
        """

        h_4096 = self.hipt_vit_WSI.global_phi(h_4096)
        h_4096 = self.hipt_vit_WSI.global_transformer(h_4096.unsqueeze(1)).squeeze(1)
        A_4096, h_4096 = self.hipt_vit_WSI.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_WSI = self.hipt_vit_WSI.global_rho(h_path)

        logits = self.hipt_vit_WSI.classifier(h_WSI)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        return F.softmax(logits, dim=1), Y_hat





