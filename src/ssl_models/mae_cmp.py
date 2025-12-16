import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from .abstract_ssl_model import AbstractSSLModel
from .emp import TotalCodingRate, cal_TCR

from ..backbones import ViT


class MAECMP(torch.nn.Module, AbstractSSLModel):
    def __init__(self,
                 vit_encoder = ViT,
                 image_size: int = 32,
                 emb_dim: int = 192, 
                 decoder_layer: int = 4,
                 decoder_head: int = 3,
                 mask_ratio: float = 0.75,
                 num_views: int = 2,
                 use_projector: bool = True,
                 dim_proj = 512,
                 omega_contr: float = 1.0,
                 omega_recon: float = 1.0,
                 tcr_strength=0.005,
                 alpha_multipatch=200,
                 temperature=0.5,
                 save_pth: str = None,
                 ) -> None:
        super().__init__()

        self.emb_dim = emb_dim
        self.encoder = vit_encoder
        self.patch_size = self.encoder.patch_size
        self.encoder.init_mask_ratio(mask_ratio)
        self.decoder = MAE_Decoder(image_size, self.patch_size, emb_dim, decoder_layer, decoder_head)
        self.mask_ratio = mask_ratio
        self.num_views = num_views # number of views (augmentations) per sample
        self.use_projector = use_projector # whether to use projector head for additional loss
        self.dim_proj = dim_proj
        self.omega_contr = omega_contr # weight for contrastive loss
        self.omega_recon = omega_recon # weight for CMP-MAE loss
        self.tcr_strength = tcr_strength
        self.alpha_multipatch = alpha_multipatch
        self.temperature = temperature
        self.save_pth = save_pth

        # How many patches is the image divided into
        self.num_patches = (image_size // self.patch_size) ** 2

        self.criterion_tcr = TotalCodingRate(eps=0.2)

        # Add a projector head for clf token (2-layer MLP)
        self.projector = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim, bias=False),
                                        nn.BatchNorm1d(self.emb_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(self.emb_dim, self.dim_proj),
                                        nn.BatchNorm1d(self.dim_proj, affine=False)) # output layer
        self.projector[3].bias.requires_grad = False # hack: not use bias as it is followed by BN

        self.model_name = 'mae_cmp'

        def mae_loss(predicted_img, img, mask):
            return torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
        
        self.criterion = mae_loss

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'image_size: {image_size}\n')
                f.write(f'patch_size: {self.patch_size}\n')
                f.write(f'emb_dim: {emb_dim}\n')
                f.write(f'decoder_layer: {decoder_layer}\n')
                f.write(f'decoder_head: {decoder_head}\n')
                f.write(f'mask_ratio: {mask_ratio}\n')
                f.write(f'num_views: {num_views}\n')
                f.write(f'dim_proj: {dim_proj}\n')
                f.write(f'omega_contr: {omega_contr}\n')
                f.write(f'omega_recon: {omega_recon}\n')
                f.write(f'tcr_strength: {tcr_strength}\n')
                f.write(f'alpha_multipatch: {alpha_multipatch}\n')
                f.write(f'temperature: {temperature}\n')


    def forward(self, x_views_list):
        x_views = torch.cat(x_views_list, dim=0)
        img = x_views
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        loss_mae = self.criterion(predicted_img, img, mask)
        
        clf_features = features[0]

        if self.use_projector:
            # Get projector outputs for clf token features
            projected_features = self.projector(clf_features)
        else:
            projected_features = clf_features

        if self.omega_contr > 0:
            contrastive_loss = self.contrastive_loss(projected_features)
        else:
            contrastive_loss = 0

        if self.omega_recon > 0:
            loss_reconstruction_cmp = self.cmp_reconstruction_loss(predicted_img, mask)
        else:
            loss_reconstruction_cmp = 0

        z_list = projected_features.chunk(self.num_views, dim=0) # separate views
        loss_TCR = cal_TCR(z_list, self.criterion_tcr, self.num_views)

        loss = self.alpha_multipatch * (loss_mae + self.omega_contr * contrastive_loss + self.omega_recon * loss_reconstruction_cmp) + self.tcr_strength * loss_TCR
        return loss, [clf_features],  [clf_features]
    
    def cmp_reconstruction_loss(self, predicted_img, mask):
        """Aggregate reconstructed images from different views
        and compute MAE loss between each reconstructed view and their average"""

        # Subdivide reconstructed images into views
        predicted_img_list = predicted_img.chunk(self.num_views, dim=0)
        predicted_img_list = torch.stack(list(predicted_img_list), dim=0) # [n_views, B, C, H, W]
        avg_img = predicted_img_list.mean(dim=0).detach() # [B, C, H, W]

        # Subdivide masks into views
        mask_list = mask.chunk(self.num_views, dim=0)
        mask_list = torch.stack(list(mask_list), dim=0) # [n_views, B, C, H, W]

        loss = 0
        for v in range(self.num_views):
            loss += torch.mean((predicted_img_list[v] - avg_img) ** 2 * mask_list[v]) / self.mask_ratio
        loss = loss / self.num_views
        return loss

    def contrastive_loss(self, z):
        # Normalize projection feature vectors
        z_norm = F.normalize(z, dim=-1) # [B*n_views, dim_proj]

        # Subdivide normalized projections in views from same sample
        z_norm_list = z_norm.chunk(self.num_views, dim=0)
        z_norm_list = torch.stack(list(z_norm_list), dim=0)
        # Compute average normalized projection
        z_avg_norm = z_norm_list.mean(dim=0).detach() # [B, dim_proj]
        batch_size = z_avg_norm.size(0)

        # Compute similarity matrix between all normalized projections and average projections (no mask needed)
        sim_matrix_avg = torch.exp(torch.mm(z_norm, z_avg_norm.t().contiguous()) / self.temperature) # [B*n_views, B]
        # Compute similarity matrix between all normalized projections (will need mask)
        sim_matrix = torch.exp(torch.mm(z_norm, z_norm.t().contiguous()) / self.temperature) # [B*n_views, B*n_views]
    
        # Mask the diagonal of each [B, B] block
        mask = torch.ones_like(sim_matrix).bool()
        eye_mask = ~torch.eye(batch_size, device=sim_matrix.device).bool()
        for i in range(self.num_views):
            for j in range(self.num_views):
                mask[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = eye_mask
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size * self.num_views, -1) # [B*n_views, B*n_views - n_views]
        # Concat the sim_matrix_avg and the masked sim_matrix
        sim_matrix = torch.cat([sim_matrix, sim_matrix_avg], dim=1) # [B*n_views, B*n_views - n_views + B]

        # Positive similarity (between each projection and the average projection of the corresponding sample)
        pos_sim = torch.exp(torch.sum(z_norm * z_avg_norm.repeat(self.num_views, 1), dim=-1) / self.temperature) # [B*n_views]

        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss
    
    def get_encoder(self):
       return self.encoder
    
    def get_encoder_for_eval(self):           
        return self.encoder.return_features_wrapper()
    
    def get_projector(self):
        # No projector head
        return torch.nn.Identity()
    
    def get_embedding_dim(self):
        return self.emb_dim
    
    def get_projector_dim(self):
        # No projector head
        return self.get_embedding_dim()
    
    def get_criterion(self):
        return self.criterion, False
    
    def get_name(self):
        return self.model_name
    
    def get_params(self):
        return list(self.parameters())

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))