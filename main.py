import torch

import os
import datetime
import tqdm as tqdm
import numpy as np
import pandas as pd
import time

from src.get_datasets import get_benchmark, get_iid_dataset, get_downstream_benchmark
from src.probing import exec_probing, ProbingSklearn, ProbingPytorch, ProbingMultipatch
from src.backbones import get_encoder

from src.ssl_models import BarlowTwins, SimSiam, BYOL, MoCo, SimCLR, EMP, MAE, SimSiamCMP, BYOLCMP, SimCLRCMP, MAECMP, recover_ssl_model

from src.strategies import NoStrategy, Replay, LUMP, MinRed, CaSSLe, CaSSLeR, ReplayEMP
from src.standalone_strategies import SCALE, OsirisR

from src.trainer import Trainer

from src.buffers import get_buffer

from src.utils import write_final_scores, read_command_line_args, calculate_forgetting, save_avg_stream_acc

def exec_experiment(**kwargs):
    standalone_strategies = ['scale']
    buffer_free_strategies = ['no_strategy', 'cassle']

    # Checks for CLEAR
    if  kwargs["dataset"] == "clear100":
        if kwargs["num_exps"] != 11:
            print(f'WARNING: Selected number of experiences {kwargs["num_exps"]} is different from default CLEAR100 experiences, resetting to 11 experiences.')
            kwargs["num_exps"] = 11
        if kwargs["iid"]:
            print(f'WARNING: IID pretraining is not supported for CLEAR100, resetting to False.')
            kwargs["iid"] = False

    # Ratios of tr set used for training linear probe
    if kwargs["use_probing_tr_ratios"]:
        probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
    else:
        probing_tr_ratio_arr = [1]


    # Set up save folders
    str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
    if kwargs["strategy"] in standalone_strategies:
        folder_name = f'{kwargs["strategy"]}_{kwargs["dataset"]}_{str_now}'
    elif kwargs['no_train']:
        folder_name = f'notrain_{kwargs["dataset"]}_{str_now}'
    else:
        folder_name = f'{kwargs["strategy"]}_{kwargs["model"]}_{kwargs["dataset"]}_{str_now}'
    if kwargs["iid"]:
        folder_name = 'iid_' + folder_name
    save_pth = os.path.join(kwargs["save_folder"], f'{folder_name}_{kwargs["name"]}')
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    

    # Save general kwargs
    with open(save_pth + '/config.txt', 'a') as f:
        f.write('\n')
        f.write(f'---- EXPERIMENT CONFIGS ----\n')
        f.write(f'Seed: {kwargs["seed"]}\n')
        f.write(f'Dataset Seed: {kwargs["dataset_seed"]}\n')
        f.write(f'Experiment Date: {str_now}\n')
        f.write(f'Model: {kwargs["model"]}\n')
        f.write(f'Encoder: {kwargs["encoder"]}\n')
        f.write(f'Dataset: {kwargs["dataset"]}\n')
        if kwargs["downstream"]:
            f.write(f'Downstream: {kwargs["downstream"]}\n')
            f.write(f'Downstream Dataset: {kwargs["downstream_dataset"]}\n')
        f.write(f'Number of Experiences: {kwargs["num_exps"]}\n')
        f.write(f'Memory Size: {kwargs["mem_size"]}\n')
        f.write(f'MB Passes: {kwargs["mb_passes"]}\n')
        f.write(f'Num Epochs: {kwargs["epochs"]}\n')
        f.write(f'Train MB Size: {kwargs["tr_mb_size"]}\n')
        f.write(f'Replay MB Size: {kwargs["repl_mb_size"]}\n')
        f.write(f'Online transforms type: {kwargs["online_transforms_type"]}\n')
        f.write(f'Multipatch: {kwargs["multipatch"]}\n')
        f.write(f'Num Views: {kwargs["num_views"]}\n')
        f.write(f'IID pretraining: {kwargs["iid"]}\n')
        f.write(f'Save final model: {kwargs["save_model_final"]}\n')
        f.write(f'-- Pretrained weights initialization configs --\n')
        f.write(f'Pretrain init: {kwargs["pretrain_init_type"]}\n')
        if kwargs["pretrain_init_type"] == 'encoder' or kwargs["pretrain_init_type"] == 'ssl':
            f.write(f'Pretrain init source: {kwargs["pretrain_init_source"]}\n')
            f.write(f'Pretrain init path: {kwargs["pretrain_init_pth"]}\n')

        f.write(f'-- Probing configs --\n')
        f.write(f'Probing after all experiences: {kwargs["probing_all_exp"]}\n')
        f.write(f'Probing on Separated exps: {kwargs["probing_separate"]}\n')
        f.write(f'Probing on Up To current exps: {kwargs["probing_upto"]}\n')
        f.write(f'Probing on all Joint exps: {kwargs["probing_joint"]}\n')
        f.write(f'Probing Validation Ratio: {kwargs["probing_val_ratio"]}\n')
        f.write(f'Probing Train Ratios: {probing_tr_ratio_arr}\n')

    # Dataset
    benchmark, image_size = get_benchmark(
        dataset_name=kwargs["dataset"],
        dataset_root=kwargs["dataset_root"],
        num_exps=kwargs["num_exps"],
        seed=kwargs["dataset_seed"],
        val_ratio=kwargs["probing_val_ratio"],
        evaluation_protocol_clear=kwargs["evaluation_protocol_clear"],
    )
    if kwargs["iid"]:
        iid_tr_dataset = get_iid_dataset(benchmark)

    if kwargs["multipatch"]:
        # Check that online transforms are absent
        assert kwargs["online_transforms_type"] == "none", f'Online transforms are not supported for multipatch at the moment. \
                                                            Use "--online-transforms-type none" instead of {kwargs["online_transforms_type"]} .' 
        benchmark, image_size = get_benchmark(
            dataset_name=kwargs["dataset"],
            dataset_root=kwargs["dataset_root"],
            num_exps=kwargs["num_exps"],
            seed=kwargs["dataset_seed"],
            val_ratio=kwargs["probing_val_ratio"],
            evaluation_protocol_clear=kwargs["evaluation_protocol_clear"],
            transforms="multipatch",
            num_views=kwargs["num_views"],
            strategy_name=kwargs["strategy"],
        )
        if kwargs["iid"]:
            iid_tr_dataset = get_iid_dataset(benchmark)

        benchmark_eval, _ = get_benchmark(
            dataset_name=kwargs["dataset"],
            dataset_root=kwargs["dataset_root"],
            num_exps=kwargs["num_exps"],
            seed=kwargs["dataset_seed"],
            val_ratio=kwargs["probing_val_ratio"],
            evaluation_protocol_clear=kwargs["evaluation_protocol_clear"],
            transforms="multipatch",
            num_views=kwargs["num_views_eval"],
            strategy_name=kwargs["strategy"],
        )

    # Downstream
    if kwargs["downstream"]:
        downstream_benchmark = get_downstream_benchmark(
            downstream_name=kwargs["downstream_dataset"],
            dataset_root=kwargs["downstream_dataset_root"],
            seed=kwargs["dataset_seed"],
            val_ratio=kwargs["probing_val_ratio"],
        )

    # Set seed (After get_benchmark!)
    torch.manual_seed(kwargs["seed"])
    np.random.default_rng(kwargs["seed"])

    # Device
    if torch.cuda.is_available():       
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        if kwargs["gpu_idx"] < torch.cuda.device_count():
            device = torch.device(f"cuda:{kwargs['gpu_idx']}")
        else:
            device = torch.device("cuda")
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Encoder
    encoder, dim_encoder_features = get_encoder(encoder_name=kwargs["encoder"],
                                                image_size=image_size,
                                                ssl_model_name=kwargs["model"],
                                                vit_avg_pooling=kwargs["vit_avg_pooling"],
                                                pretrain_init_type=kwargs["pretrain_init_type"],
                                                pretrain_init_source=kwargs["pretrain_init_source"],
                                                pretrain_init_pth=kwargs["pretrain_init_pth"],
                                                save_pth=save_pth
                                                )
    


    # Buffer
    if not kwargs["strategy"] in buffer_free_strategies:
        if kwargs["buffer_type"] == "default":
            # Set default buffer for each strategy
            if kwargs["strategy"] in ['replay', 'lump', 'osiris_r', 'cassle_r']:
                kwargs["buffer_type"] = "reservoir"
            elif kwargs["strategy"] == "minred":
                kwargs["buffer_type"] = "minred"
            elif kwargs["strategy"] == "scale":
                kwargs["buffer_type"] = "scale"
            elif kwargs["strategy"] == "replay_emp":
                kwargs["buffer_type"] = "aug_rep"
            else:
                raise Exception(f'Strategy {kwargs["strategy"]} not supported for default buffer')
        # Enforce buffer constraints for certain strategies  
        elif kwargs["buffer_type"] == "scale" and not kwargs["strategy"] == "scale":
            raise Exception(f"Buffer type {kwargs['buffer_type']} is only compatible with strategy 'scale'")
        elif kwargs["buffer_type"] == "aug_rep" and not kwargs["strategy"] == "replay_emp":
            raise Exception(f"Buffer type {kwargs['buffer_type']} is only compatible with strategy 'replay_emp'")
        
        buffer = get_buffer(buffer_type=kwargs["buffer_type"], mem_size=kwargs["mem_size"],
                            alpha_ema=kwargs["features_buffer_ema"],
                            device=device)

        # Save buffer configs
        with open(save_pth + '/config.txt', 'a') as f:
            f.write('\n')
            f.write(f'---- BUFFER CONFIGS ----\n')
            f.write(f'Buffer Type: {kwargs["buffer_type"]}\n')
            f.write(f'Buffer Size: {kwargs["mem_size"]}\n')
            if kwargs["buffer_type"] in ["minred", "reservoir", "fifo"]:
                f.write(f'Features update EMA param (MinRed): {kwargs["features_buffer_ema"]}\n')


    if kwargs["aligner_dim"] <= 0:
        aligner_dim = kwargs["dim_pred"]
    else:
        aligner_dim = kwargs["aligner_dim"]
    
    # ---- SSL model ----
    if not kwargs["strategy"] in standalone_strategies:
        if kwargs["model"] == 'simsiam':
            ssl_model = SimSiam(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                save_pth=save_pth)
            num_views = 2
        elif kwargs["model"] == 'simsiam_cmp':
            ssl_model = SimSiamCMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                         dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                         n_patches=kwargs["num_views"], tcr_strength=kwargs["tcr_strength"],
                                         alpha_multipatch=kwargs["alpha_multipatch"],
                                         save_pth=save_pth)
            num_views = kwargs["num_views"]

        elif kwargs["model"] == 'byol':
            ssl_model = BYOL(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                             byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                             save_pth=save_pth)
            num_views = 2
        elif kwargs["model"] == 'byol_cmp':
            ssl_model = BYOLCMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                      dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                      byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                                      n_patches=kwargs["num_views"], tcr_strength=kwargs["tcr_strength"], alpha_multipatch=kwargs["alpha_multipatch"],
                                      save_pth=save_pth)
            num_views = kwargs["num_views"]
            
        elif kwargs["model"] == 'barlow_twins':
            ssl_model = BarlowTwins(encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_features=kwargs["dim_proj"],
                                    lambd=kwargs["lambd"], save_pth=save_pth)
            num_views = 2

        elif kwargs["model"] == 'moco':
            ssl_model = MoCo(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"],
                             moco_momentum=kwargs["moco_momentum"], moco_queue_size=kwargs["moco_queue_size"],
                             moco_temp=kwargs["moco_temp"],return_momentum_encoder=kwargs["return_momentum_encoder"],
                             queue_type=kwargs["moco_queue_type"],
                             save_pth=save_pth, device=device)
            num_views = 2

        elif kwargs["model"] == 'simclr':
            ssl_model = SimCLR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                             save_pth=save_pth)
            num_views = 2

        elif kwargs["model"] == 'simclr_cmp':
            ssl_model = SimCLRCMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                 dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                                 n_patches=kwargs["num_views"], tcr_strength=kwargs["tcr_strength"],
                                 alpha_multipatch=kwargs["alpha_multipatch"],
                                 save_pth=save_pth)
            num_views = kwargs["num_views"]

        elif kwargs["model"] == 'emp':
            ssl_model = EMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"], n_patches=kwargs["num_views"],
                            emp_tcr_param=kwargs["emp_tcr_param"], emp_tcr_eps=kwargs["emp_tcr_eps"], 
                            emp_patch_sim=kwargs["emp_patch_sim"], save_pth=save_pth)
            num_views = kwargs["num_views"]

        elif kwargs["model"] == 'osiris_r':
            ssl_model = OsirisR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_proj=kwargs["dim_proj"], buffer=buffer, device=device,
                                    replay_mb_size=kwargs["repl_mb_size"],
                                    save_pth=save_pth)
            num_views = 2
            assert kwargs["strategy"] == kwargs["model"], 'Strategy and SSL model must be the same for Osiris-R'


        elif kwargs["model"] == 'mae':
            ssl_model = MAE(vit_encoder=encoder,
                            image_size=image_size, emb_dim=kwargs["mae_emb_dim"],
                            decoder_layer=kwargs["mae_decoder_layer"], decoder_head=kwargs["mae_decoder_head"],
                            mask_ratio=kwargs["mae_mask_ratio"], save_pth=save_pth)
            num_views = 1

        elif kwargs["model"] == 'mae_cmp':
            ssl_model = MAECMP(vit_encoder=encoder,
                            image_size=image_size, emb_dim=kwargs["mae_emb_dim"],
                            decoder_layer=kwargs["mae_decoder_layer"], decoder_head=kwargs["mae_decoder_head"],
                            mask_ratio=kwargs["mae_mask_ratio"], num_views=kwargs["num_views"], use_projector=kwargs["mae_cmp_use_proj"],
                            dim_proj=kwargs["dim_proj"], tcr_strength=kwargs["tcr_strength"], alpha_multipatch=kwargs["alpha_multipatch"],
                            omega_contr=kwargs["mae_cmp_omega_contr"], omega_recon=kwargs["mae_cmp_omega_recon"],
                            save_pth=save_pth)
            num_views = kwargs["num_views"]
            
        else:
            raise Exception(f'Invalid model {kwargs["model"]}') 
        
        # Initialization from pretrained weights of SSL model
        if kwargs["pretrain_init_type"] == 'ssl':
            if kwargs["pretrain_init_source"] == 'path':
                ssl_model = recover_ssl_model(ssl_model, kwargs["pretrain_init_pth"])
            else:
                raise Exception(f'Invalid pretrain_init_source for ssl type pretrain initialization: {kwargs["pretrain_init_source"]}')
            
        ssl_model = ssl_model.to(device)
            
    
    # ---- Strategy ----
    if not kwargs["strategy"] in standalone_strategies:
        if kwargs["strategy"] == 'no_strategy':
            strategy = NoStrategy(ssl_model=ssl_model, device=device, save_pth=save_pth)

        elif kwargs["strategy"] == 'replay':
            strategy = Replay(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
            
        elif kwargs["strategy"] == 'scale':
            pass
            
        elif kwargs["strategy"] == 'lump':
            strategy = LUMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer,
                            alpha_lump=kwargs["alpha_lump"])
            
        elif kwargs["strategy"] == 'minred':
            strategy = MinRed(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
        elif kwargs["strategy"] == 'cassle':
            strategy = CaSSLe(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim)
            
        elif kwargs["strategy"] == 'cassle_r':
            strategy = CaSSLeR(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim)

        elif kwargs["strategy"] == 'osiris_r':
            strategy = ssl_model # SSL model and strategy are combined
            
        elif kwargs["strategy"] == 'replay_emp':
            assert kwargs["buffer_type"] == "aug_rep", "Buffer type must be 'aug_rep_buffer' (AugmentedRepresentationsBuffer) for 'replay_emp' strategy"
            assert kwargs["model"] == 'emp', "SSL model has to be 'emp' for 'replay_emp' strategy"
            strategy = ReplayEMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                                buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                                emp_loss=ssl_model.get_criterion()[0], emp_tcr_param=kwargs["emp_tcr_param"],
                                emp_tcr_eps=kwargs["emp_tcr_eps"], emp_patch_sim=kwargs["emp_patch_sim"])

        else:
            raise Exception(f'Strategy {kwargs["strategy"]} not supported')

        # Set up the trainer wrapper
        trainer = Trainer(ssl_model=ssl_model, strategy=strategy, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          lars_eta= kwargs["lars_eta"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=kwargs["save_model_every_exp"], online_transforms_type=kwargs["online_transforms_type"], num_views=num_views)
        
    else:
        # Is a standalone strategy (already includes trainer and ssl model inside the strategy itself)
        trainer = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], dim_backbone_features=dim_encoder_features,
                        momentum=kwargs["optim_momentum"], weight_decay=kwargs["weight_decay"],
                        train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                        mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                        save_model=False, online_transforms_type=kwargs["online_transforms_type"],
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                        dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], buffer_type=kwargs["buffer_type"])
        

    # Init probing
    if kwargs["probing_upto"] and not kwargs["probing_all_exp"]:
        raise Exception("Without --probing-all-exp, probing upto is equal to probing joint, please set --probing-upto to false or --probing-all-exp to true")
    
    
    probes = []
    if kwargs["probing_rr"]:
         probes.append(ProbingSklearn(probe_type='rr', device=device, mb_size=kwargs["eval_mb_size"],
                               seed=kwargs["seed"], config_save_pth=save_pth))
    if kwargs["probing_knn"]:
         probes.append(ProbingSklearn(probe_type='knn', device=device, mb_size=kwargs["eval_mb_size"],
                               knn_k=kwargs["knn_k"], seed=kwargs["seed"], config_save_pth=save_pth))
         
    if kwargs["probing_torch"]:
        probes.append(ProbingPytorch(device=device, mb_size=kwargs["eval_mb_size"], config_save_pth=save_pth,
                                 dim_encoder_features=dim_encoder_features, lr=kwargs["probe_lr"],
                                 lr_patience=kwargs["probe_lr_patience"], lr_factor=kwargs["probe_lr_factor"],
                                 lr_min=kwargs["probe_lr_min"], probing_epochs=kwargs["probe_epochs"]))
        
    if kwargs["multipatch"]:
        probes.append(ProbingMultipatch(device=device, mb_size=kwargs["eval_mb_size"], config_save_pth=save_pth,
                                 dim_encoder_features=dim_encoder_features, lr=kwargs["probe_lr"],
                                 lr_patience=kwargs["probe_lr_patience"], lr_factor=kwargs["probe_lr_factor"],
                                 lr_min=kwargs["probe_lr_min"], probing_epochs=kwargs["probe_epochs"],
                                 eval_patches=kwargs["num_views_eval"]))

    if kwargs["downstream"]:
        # Using downstream as probing
        probing_benchmark = downstream_benchmark
    elif kwargs["multipatch"]:
        probing_benchmark = benchmark_eval # Temporary this way it does not work when mixing probe types
    else:
        probing_benchmark = benchmark


    if kwargs["iid"]:
        # IID training over the entire dataset
        print(f'==== Beginning self supervised training on iid dataset ====')
        if kwargs["probing_all_exp"]:
            # Evaluate iid trained model during training (not only at the end)
            iid_intermediate_eval_dict = {
                'status': True,
                'num_exps': kwargs["num_exps"],
                'kwargs': kwargs,
                'probes': probes,
                'benchmark': benchmark,
                'probing_tr_ratio_arr': probing_tr_ratio_arr,
            }
        else:
            iid_intermediate_eval_dict = {
                'status': False,
            }

        trained_ssl_model = trainer.train_experience(iid_tr_dataset, exp_idx=0, iid_intermediate_eval_dict=iid_intermediate_eval_dict)

        if not kwargs["probing_all_exp"]:
            exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(), 
                        pretr_exp_idx=0, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
        
    elif kwargs["no_train"]:
        # No SSL training is done, only using the randomly initialized encoder as feature extractor
        if kwargs["encoder"] in ['vit_tiny']:
            encoder_for_probing = encoder.return_features_wrapper()
        else:
            encoder_for_probing = encoder
        exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=encoder_for_probing, pretr_exp_idx=0,
                     probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)

    else:
        tr_time = 0
        # Self supervised training over the experiences
        for exp_idx, exp_dataset in enumerate(benchmark.train_stream):
            print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
            t_start = time.time()
            trained_ssl_model = trainer.train_experience(exp_dataset, exp_idx)
            tr_time += (time.time() - t_start)
            if kwargs["probing_all_exp"]:
                exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(), 
                     pretr_exp_idx=exp_idx, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
        with open(save_pth + '/times.txt', 'a') as f:
            f.write(f'Training time: {tr_time}')
            
        if not kwargs["probing_all_exp"]:
            # Probe only at the end of training
            exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(), 
                     pretr_exp_idx=exp_idx, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
                
        
    # Calculate and save final probing scores
    for probe in probes:
        probe_pth = os.path.join(save_pth, f'probe_{probe.get_name()}')
        if kwargs['probing_separate']:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_separate'),
                            output_file=os.path.join(save_pth, 'final_scores_separate.csv'))
        if kwargs['probing_joint']:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_joint'),
                            output_file=os.path.join(save_pth, 'final_scores_joint.csv'))
            if kwargs["probing_all_exp"]:
                save_avg_stream_acc(probe=probe.get_name(), save_pth=save_pth)

        if kwargs['probing_upto'] and not kwargs["probing_joint"]:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_upto'),
                            output_file=os.path.join(save_pth, 'final_scores_joint.csv'))
        #  Calculate forgetting
        if kwargs["probing_separate"] and kwargs["probing_all_exp"] and not (kwargs["iid"] or kwargs["no_train"]):
            calculate_forgetting(save_pth=probe_pth, num_exps=kwargs["num_exps"], probing_tr_ratio_arr=probing_tr_ratio_arr)

        
    # Save final pretrained model
    if kwargs["save_model_final"]:
        chkpt_pth = os.path.join(save_pth, 'checkpoints')
        if not os.path.exists(chkpt_pth):
            os.makedirs(chkpt_pth)
        if kwargs["no_train"]:
            torch.save(encoder.state_dict(),
                    os.path.join(chkpt_pth, f'final_model_state.pth'))
        else:
            if kwargs['strategy'] in standalone_strategies:
                torch.save(trained_ssl_model.get_encoder_for_eval().state_dict(),
                        os.path.join(chkpt_pth, f'final_model_state.pth'))
            else:
                # Default case:
                torch.save(trained_ssl_model.state_dict(),
                        os.path.join(chkpt_pth, f'final_model_state.pth'))


    return save_pth





if __name__ == '__main__':
    # Parse arguments
    args = read_command_line_args()

    exec_experiment(**args.__dict__)
