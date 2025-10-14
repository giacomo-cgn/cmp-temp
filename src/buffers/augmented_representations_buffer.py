import random
from typing import List, Tuple
import torch


class AugmentedRepresentationsBuffer:
    """
    Custom reservoir buffer that stores the input sample together with all the last representations coming from
    different augmentations of the input sample
    """
    def __init__(self, buffer_size, device='cpu'):
        self.buffer_size = buffer_size # Maximum size of the buffer
        self.buffer = torch.empty(0,1).to(device) # Buffer for input samples only (e.g. images)
        self.buffer_features = [] # List of tensors, each corresponding to the features
        # from all augmentations of one image [(num_views, dim_features), ...]
        # self.buffer_features = torch.empty(0,1).to(device) # Buffer for corresponding sample features
        self.device = device

        self.seen_samples = 0 # Samples seen so far

    # Add a batch of samples and features to the buffer
    def add(self, batch_x: torch.Tensor, batch_features: List[torch.Tensor]):
        """
        batch_x: batch of input samples
        batch_features: list of num_views tensors, each tensor is a batch of features extracted from one view
                        [(mb_size, dim_features), ...]
        """
        assert batch_x.size(0) == batch_features[0].size(0), 'Batch size mismatch'

        # Convert batch_features to be a list of tensors, but with each tensor corresponding to the features
        # from all augmentations of one image [(num_views, dim_features), ...]
        batch_features = [torch.stack([batch_features[j][i] for j in range(len(batch_features))]).to(self.device) \
                          for i in range(batch_features[0].size(0))]

        batch_x = batch_x.to(self.device)

        # Initialize empty buffers
        if self.buffer.size(0) == 0:
            # Extend buffer to have same dim of batch_x
            buffer_shape = list(batch_x.size())
            buffer_shape[0] = 0
            self.buffer = torch.empty(buffer_shape).to(self.device)

            # # Extend buffer_features to have same dim of batch_features
            # buffer_shape = list(batch_features.size())
            # buffer_shape[0] = 0
            # self.buffer_features = torch.empty(buffer_shape).to(self.device)

        batch_size = batch_x.size(0)

        if self.seen_samples < self.buffer_size:
            # Store samples until the buffer is full
            if self.seen_samples + batch_size <= self.buffer_size:
                # If there is enough space in the buffer, add all the samples
                self.buffer = torch.cat((self.buffer, batch_x), dim=0)
                self.buffer_features = self.buffer_features + batch_features
                # self.buffer_features = torch.cat((self.buffer_features, batch_features), dim=0)
                self.seen_samples += batch_size
            else:
                # If there is not enough space, add only the remaining samples
                remaining_space = self.buffer_size - self.seen_samples
                self.buffer = torch.cat((self.buffer, batch_x[:remaining_space]), dim=0)
                self.buffer_features = self.buffer_features + batch_features[:remaining_space]
                # self.buffer_features = torch.cat((self.buffer_features, batch_features[:remaining_space]), dim=0)
                self.seen_samples += remaining_space
        else:
            # Replace samples with probability buffer_size/seen_samples
            for i in range(batch_size):
                replace_index = random.randint(0, self.seen_samples + i)

                if replace_index < self.buffer_size:
                    self.buffer[replace_index] = batch_x[i]
                    self.buffer_features[replace_index] = batch_features[i]
                    # self.buffer_features[replace_index] = batch_features[i]
            
            self.seen_samples += batch_size

        
    # Sample batch_size samples from the buffer, 
    # returns samples and indices of extracted samples (for feature update)
    def sample(self, batch_size) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
        """
        Returns a batch of samples and their corresponding indices from the buffer
        """
        assert batch_size <= len(self.buffer)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # Sample batch_size indices
        indices = random.sample(range(len(self.buffer)), batch_size)

        # Get sample batch from indices
        print("buffer size:", self.buffer.size())
        print("buffer features size:", len(self.buffer_features))
        batch_x = self.buffer[indices]
        # batch_features = [self.buffer_features[i] for i in indices]
        num_views = len(self.buffer_features[0])
        print("num_views:", num_views)
        batch_features = [torch.stack([self.buffer_features[i][j] for i in indices]) for j in range(num_views)]
        print("batch_features shape:", batch_features[0].shape)
        print('batch_features len:', len(batch_features))

        return batch_x, batch_features, indices