import torch
import torch.nn as nn
import copy
import random


from .abstract_ssl_model import AbstractSSLModel
from ..utils import update_ema_params


class MoCo(nn.Module, AbstractSSLModel):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim_backbone_features,
                  dim_proj=2048,
                  moco_momentum=0.999,
                  moco_queue_size=2000,
                  moco_temp=0.07,
                  return_momentum_encoder=False, 
                  save_pth=None,
                  device = 'cpu',
                  queue_type: str = "fifo"):

        super(MoCo, self).__init__()
        self.save_pth = save_pth
        self.model_name = 'moco'
        self.dim_projector = dim_proj
        self.moco_momentum = moco_momentum
        self.moco_queue_size = moco_queue_size # K
        self.moco_temp = moco_temp
        self.return_momentum_encoder = return_momentum_encoder
        self.device = device
        self.queue_type = queue_type


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

        # Momentum network
        self.momentum_encoder = copy.deepcopy(self.online_encoder)
        self.momentum_projector = copy.deepcopy(self.online_projector)
        
        # Stop gradient in momentum network
        self.momentum_encoder.requires_grad_(False)
        self.momentum_projector.requires_grad_(False)

        # create the queue
        if queue_type != "none":
            self.register_buffer("queue", torch.randn(dim_proj, moco_queue_size))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            if self.queue_type == 'fifo':
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            elif self.queue_type == 'reservoir':
                self.register_buffer("seen_samples", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- SSL MODEL CONFIG ----\n')
                f.write(f'MODEL: {self.model_name}\n')
                f.write(f'dim_projector: {dim_proj}\n')
                f.write(f'MoCo Momentum: {self.moco_momentum}\n')
                f.write(f'MoCo Queue Size: {self.moco_queue_size}\n')
                f.write(f'MoCo Temperature: {self.moco_temp}\n')
                f.write(f'MoCo queue type: {self.queue_type}\n')


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        if self.queue_type == 'fifo':
            ptr = int(self.queue_ptr)
            # replace the keys at ptr (dequeue and enqueue)
            if ptr + batch_size < self.moco_queue_size:
                self.queue[:, ptr : ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.moco_queue_size  # move pointer
            else:
                self.queue[:, ptr:] = keys.T[:, :self.moco_queue_size - ptr]
                remaining_batch_size = batch_size - (self.moco_queue_size - ptr)
                self.queue[:, :remaining_batch_size] = keys.T[:, self.moco_queue_size - ptr:]
                ptr = remaining_batch_size # move pointer           
            self.queue_ptr[0] = ptr

        elif self.queue_type == 'reservoir':
            seen_samples = int(self.seen_samples)
            if seen_samples < self.moco_queue_size:
                # Store samples until the buffer is full
                if seen_samples + batch_size <= self.moco_queue_size:
                    self.queue[:, seen_samples : seen_samples + batch_size] = keys.T
                    seen_samples += batch_size
                else:
                    remaining_batch_size = self.moco_queue_size - seen_samples
                    self.queue[:, seen_samples:] = keys.T[:, :remaining_batch_size]
                    seen_samples = self.moco_queue_size
            else:
                # Replace samples with probability buffer_size/seen_samples
                for i in range(batch_size):
                    replace_index = random.randint(0, seen_samples + i)
                    if replace_index < self.moco_queue_size:
                        self.queue[:, replace_index] = keys.T[:, i]
                seen_samples += batch_size
            self.seen_samples[0] = seen_samples

    @torch.no_grad()
    def _batch_shuffle(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).to(self.device)

        # shuffled data
        x_shuffled = x[idx_shuffle]

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x_shuffled, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        # restore the original order
        x_unshuffled = x[idx_unshuffle]

        return x_unshuffled
    

    def forward(self, x_views_list):

        x1 = x_views_list[0]
        x2 = x_views_list[1]

        # First augmentation passed in the online (query) encoder
        e1_onl = self.online_encoder(x1)
        z1_onl = self.online_projector(e1_onl)
        q = nn.functional.normalize(z1_onl, dim=1)

        # Second augmentation passed in the momentum (key) encoder
        with torch.no_grad():
            # Update momentum encoder
            update_ema_params(
                self.online_encoder.parameters(), self.momentum_encoder.parameters(), self.moco_momentum) 
            # Update momentum projector
            update_ema_params(
            self.online_projector.parameters(), self.momentum_projector.parameters(), self.moco_momentum)

            x2, idx_unshuffle = self._batch_shuffle(x2)
            z2_mom = self.momentum_projector(self.momentum_encoder(x2))
            z2_mom = self._batch_unshuffle(z2_mom, idx_unshuffle)
            k = self._batch_unshuffle(z2_mom, idx_unshuffle)


        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        if self.queue_type != 'none':
            # negative logits: NxK
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
        else:
            logits = l_pos

        # apply temperature
        logits /= self.moco_temp

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # dequeue and enqueue
        if self.queue_type != 'none':
            self._dequeue_and_enqueue(k)

        loss = self.criterion(logits, labels)

        return loss.mean(), [z1_onl], [e1_onl]
    

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
        return self.criterion, False
    
    def get_name(self):
        return self.model_name

    def get_params(self):
        return list(self.get_encoder().parameters()) + list(self.get_projector().parameters())

