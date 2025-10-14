import torch
from torch import nn
import torch.nn.functional as F
from .abstract_ssl_model import AbstractSSLModel
from .emp import TotalCodingRate, cal_TCR

class SimSiamCMP(nn.Module, AbstractSSLModel):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim_backbone_features, dim_proj=2048, dim_pred=512, 
                 n_patches=2, tcr_strength=0.005, alpha_multipatch=200, save_pth=None):
        super(SimSiamCMP, self).__init__()
        self.encoder = base_encoder
        self.save_pth = save_pth
        self.model_name = 'simsiam_cmp'
        self.dim_projector = dim_proj
        self.dim_predictor = dim_pred
        self.n_patches = n_patches
        self.tcr_strength = tcr_strength
        self.alpha_multipatch = alpha_multipatch


        # Set up criterion
        self.criterion = nn.CosineSimilarity(dim=1)
        self.criterion_tcr = TotalCodingRate(eps=0.2)


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


        # Build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim_proj, dim_pred, bias=False),
                                        nn.BatchNorm1d(dim_pred),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim_pred, dim_proj)) # output layer
        
        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'dim_predictor: {dim_pred}\n')
                f.write(f'n_patches: {n_patches}\n')
                f.write(f'tcr_strength: {tcr_strength}\n')
                f.write(f'alpha_multipatch: {alpha_multipatch}\n')

    def forward(self, x_views_list):

        x_views = torch.cat(x_views_list, dim=0)

        # Forward pass for all views
        e = self.encoder(x_views)
        z = self.projector(e)
        p = self.predictor(z)

        # Subdivide e projections in patches from same sample
        e_list = e.chunk(self.n_patches, dim=0)
        # Subdivide z projections in patches from same sample
        z_list = z.chunk(self.n_patches, dim=0)
        # Subdivide p predictions in patches from same sample
        p_list = p.chunk(self.n_patches, dim=0)

        num_patch = len(z_list)
        p_list = torch.stack(list(p_list), dim=0)
        z_list = torch.stack(list(z_list), dim=0)
        z_avg = z_list.mean(dim=0).detach()
        
        loss = 0
        for i in range(num_patch):
            loss += F.cosine_similarity(p_list[i], z_avg, dim=1).mean()
            
        loss = -loss/num_patch
        
        loss_TCR = cal_TCR(z_list, self.criterion_tcr, num_patch)
        loss = self.alpha_multipatch*loss + self.tcr_strength*loss_TCR

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
        return lambda x, y: -self.criterion(x,y), True
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())