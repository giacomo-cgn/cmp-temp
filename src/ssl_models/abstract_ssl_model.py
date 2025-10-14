from typing import Tuple

class AbstractSSLModel():
    def __init__(self):
        self.model_name = "AbstractSSLModel"

    def forward(self, x1, x2):
        pass

    def update_target(self):
        pass

    def get_encoder(self):
       pass
    
    def get_encoder_for_eval(self):
        pass
    
    def get_projector(self):
        pass
        
    def get_embedding_dim(self):
        pass
    
    def get_projector_dim(self):
        pass
    
    def get_criterion(self) -> Tuple[object, bool]:
        """Return the model's loss criterion and if it is a binary loss (e.g. MSE or cosine similarity)"""
        pass
    
    def get_name(self):
        pass

    def after_backward(self):
        pass

    def get_params(self) ->  list:
        return []
    
    def get_name(self):
        return self.model_name