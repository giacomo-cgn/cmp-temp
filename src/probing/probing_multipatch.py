import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from .abstract_probe import AbstractProbe
from ..utils import SupervisedDataset
from ..transforms import MultiPatchTransforms

class ProbingMultipatch(AbstractProbe):
    def __init__(self,                 
                 device: str = 'cpu',
                 mb_size: int = 512,
                 seed: int = 42,
                 config_save_pth: str = None,
                 dim_encoder_features: int = 512,
                 lr: float = 5e-2,
                 lr_patience: int = 5,
                 lr_factor: int = 3,
                 lr_min: float = 1e-4,
                 probing_epochs: int = 100,
                 eval_patches: int = 2,
    ):
        
        self.device = device
        self.mb_size = mb_size
        self.seed = seed
        self.probe_type = "multipatch"
        self.dim_encoder_features = dim_encoder_features

        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.probing_epochs = probing_epochs

        self.eval_patches = eval_patches

        self.criterion = nn.CrossEntropyLoss()


        if config_save_pth is not None:
            # Save model configuration
            with open(config_save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- PROBE CONFIG ----\n')
                f.write(f'Probing type: {self.probe_type}\n')
                f.write(f'Eval MB size: {mb_size}\n')
                f.write(f'Probing LR: {self.lr}\n')
                f.write(f'Probing lr patience: {self.lr_patience}\n')
                f.write(f'Probing lr factor: {self.lr_factor}\n')
                f.write(f'Probing lr min: {self.lr_min}\n')
                f.write(f'Probing epochs: {self.probing_epochs}\n')
                f.write(f'Eval patches: {self.eval_patches}\n')

    def get_name(self) -> str:
        return self.probe_type
    
    def probe(self,
              encoder: nn,
              tr_dataset: Dataset,
              test_dataset: Dataset,
              val_dataset: Dataset = None,
              exp_idx: int = None, # Task index on which probing is executed, if None, we are in joint or upto probing
              tr_samples_ratio: float = 1.0,
              save_file: str = None,
              ):
        
        if val_dataset is None:
            raise ValueError("Validation dataset is required for PyTorch linear probing")
        
        self.encoder = encoder.to(self.device)
        self.exp_idx = exp_idx
        self.save_file = save_file
        self.tr_samples_ratio = tr_samples_ratio

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                # Write header for probing log file
                if not os.path.exists(self.save_file) or os.path.getsize(self.save_file) == 0:
                    if self.exp_idx is not None:
                        f.write('probing_exp_idx,val_acc,test_acc\n')
                    else:
                        f.write(f'val_acc,test_acc\n')
        
        # Prepare dataloaders
        # Select only a random ratio of the train data for probing
        used_ratio_samples = int(len(tr_dataset) * self.tr_samples_ratio)
        tr_dataset, _ = random_split(tr_dataset, [used_ratio_samples, len(tr_dataset) - used_ratio_samples],
                                     generator=torch.Generator().manual_seed(self.seed)) # Generator to ensure same splits
        tr_dataset = SupervisedDataset(tr_dataset)
        train_loader = DataLoader(dataset=tr_dataset, batch_size=self.mb_size, shuffle=True, num_workers=8)
        test_dataset = SupervisedDataset(test_dataset)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.mb_size, shuffle=False, num_workers=8)
        if val_dataset is not None:
            val_dataset = SupervisedDataset(val_dataset)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.mb_size, shuffle=False, num_workers=8)

        with torch.no_grad():

            # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
            self.encoder.eval()

            # Get encoder activations for tr dataloader
            tr_activations_list = []
            tr_labels_list = []
            print('Getting train activations for probing...')
            for input_list, labels in tqdm(train_loader):
                print(f'len input list: {len(input_list)}')
                inputs = torch.cat(input_list, dim = 0).to(self.device)
                labels = labels.to(self.device)
                activations = self.encoder(inputs)
                activations_avg = chunk_avg(activations, self.eval_patches)
                tr_activations_list.append(activations_avg.detach())
                tr_labels_list.append(labels.detach())
            tr_activations = nn.functional.normalize(torch.cat(tr_activations_list, dim=0))
            tr_labels = torch.cat(tr_labels_list, dim=0)

            # Get encoder activations for val dataloader
            val_activations_list = []
            val_labels_list = []
            print('Getting val activations for probing...')
            for input_list, labels in tqdm(val_loader):
                inputs = torch.cat(input_list, dim = 0).to(self.device)
                labels = labels.to(self.device)
                activations = self.encoder(inputs)
                activations_avg = chunk_avg(activations, self.eval_patches)
                val_activations_list.append(activations_avg.detach())
                val_labels_list.append(labels.detach())
            val_activations = nn.functional.normalize(torch.cat(val_activations_list, dim=0))
            val_labels = torch.cat(val_labels_list, dim=0)

            # Get encoder activations for test dataloader
            test_activations_list = []
            test_labels_list = []
            print('Getting test activations for probing...')
            for input_list, labels in tqdm(test_loader):
                inputs = torch.cat(input_list, dim = 0).to(self.device)
                labels = labels.to(self.device)
                activations = self.encoder(inputs)
                activations_avg = chunk_avg(activations, self.eval_patches)
                test_activations_list.append(activations_avg.detach())
                test_labels_list.append(labels.detach())
            test_activations = nn.functional.normalize(torch.cat(test_activations_list, dim=0))
            test_labels = torch.cat(test_labels_list, dim=0)

        num_classes = len(torch.unique(tr_labels))
        if max(torch.unique(tr_labels)) > num_classes - 1:
            # If only a subset of labels, rename them in [0, num_class] range
            labels_set = set(torch.unique(tr_labels).tolist())
            new_labels = list(range(len(labels_set)))
            label_map = {k: v for k, v in zip(labels_set, new_labels)}
            tr_labels = torch.tensor([label_map[l.item()] for l in tr_labels], dtype=torch.long).to(self.device)
            val_labels = torch.tensor([label_map[l.item()] for l in val_labels], dtype=torch.long).to(self.device)
            test_labels = torch.tensor([label_map[l.item()] for l in test_labels], dtype=torch.long).to(self.device)


        # Set up Linear Probe
        linear_probe_clf = SSLEvaluator(self.dim_encoder_features, num_classes, 0, 0.0)
        linear_probe_clf.to(self.device)
        _lr = self.lr 
        linear_probe_clf_optimizer = torch.optim.Adam(linear_probe_clf.parameters(), lr=_lr)

        classifier_train_step = 0
        val_step = 0
        best_val_loss = 1e10
        best_val_acc = 0.0
        patience = self.lr_patience
        linear_probe_clf.train()
        best_model = None
        
        # Training loop of the probe
        for e in range(self.probing_epochs):
            train_loss = 0.0
            train_samples = 0.0
            index = 0
            while index + self.mb_size <= len(tr_labels):
                _x = tr_activations[index:index + self.mb_size, :]
                y = tr_labels[index:index + self.mb_size]

                _x = _x.detach()
                # forward pass
                mlp_preds = linear_probe_clf(_x.to(self.device))
                mlp_loss = self.criterion(mlp_preds, y)
                # update finetune weights
                mlp_loss.backward()
                linear_probe_clf_optimizer.step()
                linear_probe_clf_optimizer.zero_grad()
                train_loss += mlp_loss.item()
                train_samples += len(y)


                classifier_train_step += 1
                index += self.mb_size

            # Eval on validation sets
            linear_probe_clf.eval()
            val_loss = 0.0
            acc_correct = 0
            acc_all = 0
            with torch.no_grad():
                singelite = False if len(val_activations) > self.mb_size else True
                index = 0
                while index + self.mb_size < len(val_activations) or singelite:
                    _x = val_activations[index:index + self.mb_size, :]
                    y = val_labels[index:index + self.mb_size]
                    _x = _x.detach();
                    # forward pass
                    mlp_preds = linear_probe_clf(_x.to(self.device))
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    acc_correct += n_corr
                    acc_all += n_all
                    val_step += 1
                    index += val_loader.batch_size
                    singelite = False
            
             # mean validation loss
            val_loss = val_loss / acc_all
            val_acc = acc_correct / acc_all

            print(
            f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |'
            )
            
            # Adapt lr
            if val_acc > best_val_acc or best_model is None:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model = copy.deepcopy(linear_probe_clf.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            else:
                patience -= 1
                if patience <= 0:
                    _lr /= self.lr_factor
                    print(' lr={:.1e}'.format(_lr),)
                    if _lr < self.lr_min:
                        print(' NO MORE PATIENCE')
                        break
                    patience = self.lr_patience
                    linear_probe_clf_optimizer.param_groups[0]['lr'] = _lr
                    linear_probe_clf.model.load_state_dict(best_model)

        linear_probe_clf.model.load_state_dict(best_model)
        linear_probe_clf.eval()

        # Eval on test set
        with torch.no_grad():
            test_loss = 0.0
            acc_correct = 0
            acc_all = 0
            singelite = False if len(test_activations) > self.mb_size else True
            index = 0
            while index + self.mb_size < len(test_activations) or singelite:
                _x = test_activations[index:index + self.mb_size, :]
                y = test_labels[index:index + self.mb_size]
                _x = _x.detach();
                # forward pass
                mlp_preds = linear_probe_clf(_x.to(self.device))
                mlp_loss = F.cross_entropy(mlp_preds, y)
                test_loss += mlp_loss.item()
                n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                n_all = y.size()[0]
                _test_acc = n_corr / n_all
                acc_correct += n_corr
                acc_all += n_all
                index += test_loader.batch_size
                singelite = False
        
            # mean test loss
            test_loss = val_loss / acc_all
            test_acc = acc_correct / acc_all

            print(f'Test loss: {test_loss}, test acc: {test_acc}')

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if val_dataset is None:
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},_,{test_acc:.4f}\n')
                    else:
                        f.write(f'_,{test_acc:.4f}\n')
                else:        
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},{best_val_acc:.4f},{test_acc:.4f}\n')
                    else:
                        f.write(f'{best_val_acc:.4f},{test_acc:.4f}\n')



class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits
    

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)