import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from .abstract_ssl_model import AbstractSSLModel
from .emp import TotalCodingRate, cal_TCR


class SimCLRCMP(nn.Module, AbstractSSLModel):
    """
    Build a SimCLR model.
    """
    def __init__(self, base_encoder, dim_backbone_features, dim_proj=2048, save_pth=None, temperature=0.5,
                 n_patches=2, tcr_strength=0.005, alpha_multipatch=200):
        super(SimCLRCMP, self).__init__()
        self.encoder = base_encoder
        self.save_pth = save_pth
        self.model_name = 'simclr_cmp'
        self.dim_projector = dim_proj
        self.temperature = temperature
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


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'SimCLR temperature: {self.temperature}\n')
                f.write(f'n_patches: {n_patches}\n')
                f.write(f'tcr_strength: {tcr_strength}\n')
                f.write(f'alpha_multipatch: {alpha_multipatch}\n')

    def simclr_cmp_loss(self, z):
        # #Helper: try to show matplotlib plots non-blocking; if that fails (headless), save to disk.
        # def _plot_matrix(mat, title, save_dir=None, fname_prefix=None, cmap='viridis'):
        #     try:
        #         arr = mat.detach().cpu().numpy() if hasattr(mat, 'detach') else np.asarray(mat)
        #     except Exception:
        #         try:
        #             arr = np.asarray(mat)
        #         except Exception:
        #             return

        #     try:
        #         plt.figure(figsize=(6, 4))
        #         if arr.ndim == 1:
        #             plt.plot(arr)
        #             plt.ylabel('value')
        #         else:
        #             # Use square pixels so the matrix' row/column ratio is preserved
        #             plt.imshow(arr, aspect='equal', cmap=cmap, interpolation='nearest')
        #             plt.colorbar()
        #         plt.title(title)
        #         plt.tight_layout()
        #         # Show in blocking mode so the user can observe the plot (will block until window is closed)
        #         plt.show(block=True)
        #         plt.close()
        #     except Exception:
        #         # In headless environments or if showing fails, silently skip plotting
        #         try:
        #             plt.close()
        #         except Exception:
        #             pass

        # Normalize projection feature vectors
        z_norm = F.normalize(z, dim=-1) # [B*n_patches, dim_proj]

        # Subdivide normalized projections in patches from same sample
        z_norm_list = z_norm.chunk(self.n_patches, dim=0)
        z_norm_list = torch.stack(list(z_norm_list), dim=0)
        # Compute average normalized projection
        z_avg_norm = z_norm_list.mean(dim=0).detach() # [B, dim_proj]
        batch_size = z_avg_norm.size(0)

        # Compute similarity matrix between all normalized projections and average projections (no mask needed)
        sim_matrix_avg = torch.exp(torch.mm(z_norm, z_avg_norm.t().contiguous()) / self.temperature) # [B*n_patches, B]
        # Compute similarity matrix between all normalized projections (will need mask)
        sim_matrix = torch.exp(torch.mm(z_norm, z_norm.t().contiguous()) / self.temperature) # [B*n_patches, B*n_patches]

        # Mask the diagonal of each [B, B] block
        mask = torch.ones_like(sim_matrix).bool()
        eye_mask = ~torch.eye(batch_size, device=sim_matrix.device).bool()
        for i in range(self.n_patches):
            for j in range(self.n_patches):
                mask[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = eye_mask
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size * self.n_patches, -1) # [B*n_patches, B*n_patches - n_patches]
        # Concat the sim_matrix_avg and the masked sim_matrix
        sim_matrix = torch.cat([sim_matrix, sim_matrix_avg], dim=1) # [B*n_patches, B*n_patches - n_patches + B]

        # Positive similarity (between each projection and the average projection of the corresponding sample)
        pos_sim = torch.exp(torch.sum(z_norm * z_avg_norm.repeat(self.n_patches, 1), dim=-1) / self.temperature) # [B*n_patches]

        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def forward(self, x_views_list):

        x_views = torch.cat(x_views_list, dim=0)

        # Forward pass for all views
        e = self.encoder(x_views)
        z = self.projector(e)
        
        # Subdivide e projections in patches from same sample
        e_list = e.chunk(self.n_patches, dim=0)
        # Subdivide z projections in patches from same sample
        z_list = z.chunk(self.n_patches, dim=0)

        num_patch = len(z_list)
        z_list = torch.stack(list(z_list), dim=0)

        loss = self.simclr_cmp_loss(z)


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
        return self.simclr_loss, True
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())