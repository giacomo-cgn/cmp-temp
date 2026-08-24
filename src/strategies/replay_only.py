import torch

from ..ssl_models import AbstractSSLModel
from .abstract_strategy import AbstractStrategy

class ReplayOnly(AbstractStrategy):

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 buffer = None,
                 device = 'cpu',
                 save_pth: str  = None,
                 replay_mb_size: int = 32,
                ):
            
        super().__init__()
        self.ssl_model = ssl_model
        self.buffer = buffer
        self.device = device
        self.save_pth = save_pth
        self.replay_mb_size = replay_mb_size

        self.strategy_name = 'replay_only'

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- STRATEGY CONFIG ----\n')
                f.write(f'STRATEGY: {self.strategy_name}\n')

    def before_forward(self, stream_mbatch):
        """Sample from buffer and concat with stream batch."""

        self.stream_mbatch = stream_mbatch

        if len(self.buffer.buffer) == 0:
            self.use_replay = False
            # Do not sample buffer if empty
            batch = stream_mbatch
        else:
            self.actual_replay_mb_size = min(self.replay_mb_size, len(self.buffer.buffer))
            # Sample from buffer and concat
            replay_batch, _, replay_indices = self.buffer.sample(self.actual_replay_mb_size)
            replay_batch = replay_batch.to(self.device)

            self.replay_indices = replay_indices
            batch = replay_batch

        return batch
    
    def after_forward(self, x_views_list, loss, z_list, e_list):
        """ Only update buffer features for replayed samples"""
        self.z_list = z_list
        if self.use_replay:
            # Ensure that is the same length as the actual replay batch size
            assert len(z_list[0]) == self.actual_replay_mb_size
            # Update replayed samples with avg of last extracted features
            avg_replayed_z = sum(z_list)/len(z_list)
            self.buffer.update_features(avg_replayed_z.detach(), self.replay_indices)
        
        return loss
    

    def after_mb_passes(self):
        """Update buffer with new samples after all mb_passes with streaming mbatch."""

        # Get features filled with zero for the stream samples (same dim as the features in z_list)
        z_list_stream_placeholder = torch.zeros(self.stream_mbatch.shape[0], self.z_list[0].shape[1]).to(self.device)

        # Update buffer with new stream samples and avg features
        self.buffer.add(self.stream_mbatch.detach(), z_list_stream_placeholder.detach())
