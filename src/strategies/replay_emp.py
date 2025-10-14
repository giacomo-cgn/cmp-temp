import torch
from torch import nn
from torch.nn import functional as F

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy
from ..buffers import AugmentedRepresentationsBuffer

from ..ssl_models.emp import TotalCodingRate, cal_TCR

class ReplayEMP(AbstractStrategy):

    def __init__(self,
                 emp_loss,
                 ssl_model: AbstractSSLModel = None,
                 buffer: AugmentedRepresentationsBuffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                 emp_tcr_param: float = 1,
                 emp_tcr_eps: float = 0.2,
                 emp_patch_sim: float = 200,
                ):
            
        super().__init__()
        self.ssl_model = ssl_model
        self.emp_loss = emp_loss
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size
        self.patch_sim = emp_patch_sim
        self.emp_tcr_param = emp_tcr_param

        self.strategy_name = 'replay_emp'

        self.replay_contractive_criterion = PastAlignSimilarityLoss()
        self.tcr = TotalCodingRate(emp_tcr_eps)

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')
                f.write(f'emp_tcr_param: {emp_tcr_param}\n')
                f.write(f'emp_tcr_eps: {emp_tcr_eps}\n')
                f.write(f'emp_patch_sim: {emp_patch_sim}\n')


        

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch

        if len(self.buffer.buffer) >= self.replay_mb_size:
            self.use_replay = True
            # Sample from buffer and concat
            replay_batch, replay_features, replay_indices = self.buffer.sample(self.replay_mb_size)
            replay_batch = replay_batch.to(self.device)
            
            combined_batch = torch.cat((replay_batch, stream_mbatch), dim=0)
            # Save buffer indices of replayed samples
            self.replay_indices = replay_indices
            self.replay_features = replay_features
        else:
            self.use_replay = False
            # Do not sample buffer if not enough elements in it
            combined_batch = stream_mbatch

        return combined_batch
    

    
    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features for replayed samples""" 
        if self.use_replay:
            self.z_views_list_stream = [z[:self.replay_mb_size] for z in z_list]
            z_views_list_replay = [z[self.replay_mb_size:] for z in z_list]

            # Standard EMP loss for stream samples
            loss_emp_stream = self.emp_loss(self.z_views_list_stream)

            # EMP+past_alignment loss for replayed samples
            loss_emp_replay_contractive, _ = self.replay_contractive_criterion(z_views_list_replay, self.replay_features)
            loss_emp_replay_tcr = cal_TCR(z_views_list_replay, self.tcr, len(z_views_list_replay))

            loss_emp_replay = self.emp_tcr_param * loss_emp_replay_tcr + self.patch_sim * loss_emp_replay_contractive

            # Sum losses (normalized by the number of samples relative to each)
            num_replay_samples = z_views_list_replay[0].size(0)
            num_stream_samples = self.z_views_list_stream[0].size(0)
            loss = (loss_emp_stream / num_stream_samples + loss_emp_replay / num_replay_samples) * (num_replay_samples + num_stream_samples)
        else:
            self.z_views_list_stream = z_list

        return loss
    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

        # Update buffer with new stream samples features of all views
        detached_z_views_list = [z.detach() for z in self.z_views_list_stream]
        self.buffer.add(self.stream_mbatch.detach(), detached_z_views_list)




class PastAlignSimilarityLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list_new, z_list_past):
        z_sim = 0
        num_patch_new = len(z_list_new)
        z_list_new = torch.stack(list(z_list_new), dim=0)
        z_list_past = torch.stack(list(z_list_past), dim=0)
        z_avg = z_list_past.mean(dim=0)
        
        z_sim = 0
        for i in range(num_patch_new):
            z_sim += F.cosine_similarity(z_list_new[i], z_avg, dim=1).mean()
            
        z_sim = z_sim/num_patch_new
        z_sim_out = z_sim.clone().detach()
                
        return -z_sim, z_sim_out