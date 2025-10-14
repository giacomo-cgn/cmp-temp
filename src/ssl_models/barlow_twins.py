from torch import nn
import torch

from .abstract_ssl_model import AbstractSSLModel

class BarlowTwins(nn.Module, AbstractSSLModel):

    def __init__(self, encoder, dim_backbone_features, dim_features=2048, lambd=5e-3, save_pth=None):
        super(BarlowTwins, self).__init__()
        self.encoder = encoder
        self.save_pth = save_pth
        self.model_name = 'barlow_twins'
        self.dim_features = dim_features

        self.lambd = lambd

        # Create 3-layer projector
        self.projector = nn.Sequential(nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(dim_backbone_features, dim_backbone_features, bias=False),
                                        nn.BatchNorm1d(dim_backbone_features),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(dim_backbone_features, dim_features),
                                        #nn.BatchNorm1d(dim_features, affine=False)
                                        ) # output layer
        self.projector[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # self.bn_loss = nn.BatchNorm1d(dim_features, affine=False)

        def barlow_twins_loss(z1, z2):
            bn = torch.nn.BatchNorm1d(self.dim_features, affine=False).to(z1.device)
            z1 = bn(z1)
            z2 = bn(z2)

            batch_size = z1.shape[0]
            # print("z1 shape:", z1.shape)
            # print("z1:", z1)
            # print("z2 shape:", z2.shape)
            # print("z2:", z2)

            # empirical cross-correlation matrix
            c = z1.T @ z2
            c.div_(batch_size)
            # print("c shape:", c.shape)
            # print("c:", c)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            return on_diag + self.lambd * off_diag
        self.criterion = barlow_twins_loss

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'Lambda: {self.lambd}\n')
                f.write(f'dim_features: {dim_features}\n')

    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        z1 = self.projector(e1)
        z2 = self.projector(e2)

        loss = self.criterion(z1, z2)
        return loss, [z1, z2], [e1, e2]
    
    def get_encoder(self):
        return self.encoder
    
    def get_encoder_for_eval(self):
        return self.encoder
    
    def get_projector(self):
        return self.projector
        
    def get_embedding_dim(self):
        return self.projector[0].weight.shape[1]
    
    def get_projector_dim(self):
        return self.dim_features
    
    def get_criterion(self):
        return self.criterion, True
    
    def get_name(self):
        return self.model_name

    def after_backward(self):
        return
    
    def get_params(self):
        return list(self.parameters())
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()