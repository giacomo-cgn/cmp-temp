import torch
from torch import nn
import torch.nn.functional as F
import copy

from .abstract_ssl_model import AbstractSSLModel
from .emp import TotalCodingRate, cal_TCR
from ..utils import update_ema_params

class BYOLCMP(nn.Module, AbstractSSLModel):

    def __init__(self, base_encoder, dim_backbone_features,
                  dim_proj=2048, dim_pred=512,
                  byol_momentum=0.9, return_momentum_encoder=True, 
                  n_patches=2,
                  tcr_strength=0.005,
                  alpha_multipatch=200,
                  save_pth=None):
        
        super(BYOLCMP, self).__init__()
        self.save_pth = save_pth
        self.model_name = 'byol_cmp'
        self.dim_projector = dim_proj
        self.dim_predictor = dim_pred
        self.n_patches = n_patches
        self.tcr_strength = tcr_strength
        self.alpha_multipatch = alpha_multipatch

        self.byol_momentum = byol_momentum
        self.return_momentum_encoder = return_momentum_encoder

        # Online encoder
        self.online_encoder = base_encoder
        # Online projector
        self.online_projector = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_proj),
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.online_projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # Momentum encoder
        self.momentum_encoder = copy.deepcopy(self.online_encoder)
        self.momentum_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradient in momentum network
        self.momentum_encoder.requires_grad_(False)
        self.momentum_projector.requires_grad_(False)

        # Build predictor for online network
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        def loss_byol(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * (x * y).sum(dim=-1)
        self.criterion = loss_byol

        self.criterion_tcr = TotalCodingRate(eps=0.2)

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'Byol_Momentum: {self.byol_momentum}\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'dim_predictor: {dim_pred}\n')
                f.write(f'n_patches: {n_patches}\n')
                f.write(f'tcr_strength: {tcr_strength}\n')
                f.write(f'alpha_multipatch: {alpha_multipatch}\n')

    def forward(self, x_views_list):
        # Concat all tensors in the list in a single tensor
        x_views = torch.cat(x_views_list, dim=0)

        # Both augmentations are passed in both momentum and online nets 
        with torch.no_grad():
            z_mom = self.momentum_projector(self.momentum_encoder(x_views))
        
        e_onl = self.online_encoder(x_views)
        z_onl = self.online_projector(e_onl)
        p = self.predictor(z_onl)

        # Subdivide e projections in patches from same sample
        e_list = e_onl.chunk(self.n_patches, dim=0)
        # Subdivide z projections in patches from same sample
        z_onl_list = z_onl.chunk(self.n_patches, dim=0)
        z_mom_list = z_mom.chunk(self.n_patches, dim=0)
        # Subdivide p predictions in patches from same sample
        p_list = p.chunk(self.n_patches, dim=0)

        num_patch = len(z_onl_list)
        p = torch.stack(list(p_list), dim=0)
        z_mom = torch.stack(list(z_mom_list), dim=0)
        z_mom_avg = z_mom.mean(dim=0).detach()
        
        loss = 0
        for i in range(num_patch):
            loss += self.criterion(p_list[i], z_mom_avg).mean()
            
        loss = -loss/num_patch
        loss_TCR = cal_TCR(z_onl_list, self.criterion_tcr, num_patch)
        loss = self.alpha_multipatch*loss + self.tcr_strength * loss_TCR

        return loss, z_onl_list, e_list
     
    @torch.no_grad()
    def update_momentum(self):
        # Update encoder
        update_ema_params(
            self.online_encoder.parameters(), self.momentum_encoder.parameters(), self.byol_momentum) 
        
        # Update projector
        update_ema_params(
           self.online_projector.parameters(), self.momentum_projector.parameters(), self.byol_momentum)

    def get_encoder_for_eval(self):
        if self.return_momentum_encoder:
            return self.momentum_encoder
        else:
            return self.online_encoder
    
    def get_encoder(self):
        return self.online_encoder
        
    def get_projector(self):
        return self.online_projector
            
    def get_embedding_dim(self):
        return self.online_projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_projector
    
    def get_criterion(self):
        return self.criterion, True
    
    def get_name(self):
        return self.model_name

    def after_backward(self):
        self.update_momentum()

    def get_params(self):
        return list(self.get_encoder().parameters()) + list(self.get_projector().parameters()) + list(self.predictor.parameters())