import torch
from torch import nn
import torch.nn.functional as F

from .abstract_ssl_model import AbstractSSLModel

class EMP(nn.Module, AbstractSSLModel):
    """
    Build an EMP SSL model.
    """
    def __init__(self,
                 base_encoder,
                 dim_backbone_features: int,
                 dim_proj: int = 2048,
                 save_pth: str  = None,
                 n_patches: int = 20,
                 emp_tcr_param: float = 1,
                 emp_tcr_eps: float = 0.2,
                 emp_patch_sim: float = 200,
                 ):


        super(EMP, self).__init__()
        
        self.model_name = 'emp'
        self.encoder = base_encoder

        self.save_pth = save_pth
        self.dim_projector = dim_proj
        self.n_patches = n_patches
        self.emp_tcr_param = emp_tcr_param
        self.emp_tcr_eps = emp_tcr_eps
        self.emp_patch_sim = emp_patch_sim

        # Set up loss definitions
        self.contractive_loss = Similarity_Loss()
        self.criterion = TotalCodingRate(eps=self.emp_tcr_eps)

        # Build a 3-layer projector
        self.projector = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_proj),
                                        nn.BatchNorm1d(dim_proj, affine=False)) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN 

        def emp_loss(z_list):
            loss_contract, _ = self.contractive_loss(z_list)
            loss_TCR = cal_TCR(z_list, self.criterion, self.n_patches)
            
            loss = self.emp_patch_sim*loss_contract + self.emp_tcr_param*loss_TCR
            return loss

        self.emp_loss = emp_loss
        
        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write ssl model hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'num_views (num patches): {self.n_patches}\n')
                f.write(f'emp_tcr_param: {self.emp_tcr_param}\n')
                f.write(f'emp_tcr_eps: {self.emp_tcr_eps}\n')
                f.write(f'emp_patch_sim: {self.emp_patch_sim}\n')

    
    def forward(self, x_views_list):
        # Concat all tensors in the list in a single tensor
        x_views = torch.cat(x_views_list, dim=0)

        # Forward pass for all views
        e = self.encoder(x_views)
        z = self.projector(e)

        # Subdivide e projections in patches from same sample
        e_list = e.chunk(self.n_patches, dim=0)
        # Subdivide z projections in patches from same sample
        z_list = z.chunk(self.n_patches, dim=0)

        
        loss = self.emp_loss(z_list)
    
        return loss, z_list, e_list
    
    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_projector
    
    def get_criterion(self):
        return self.emp_loss, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())
    

# Additional classes and functions for EMP

class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list):
        z_sim = 0
        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out
    

def cal_TCR(z_list, criterion, num_patches):
    loss = 0
    for i in range(num_patches):
        loss += criterion(z_list[i])
    loss = loss/num_patches
    return loss

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)
    

class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)