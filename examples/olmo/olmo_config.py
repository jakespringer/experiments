"""Helper functions for creating OLMo training configurations."""


def get_base_model_config():
    """Return the base model configuration shared by pretrain and CPT."""
    return {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 8,
'mlp_ratio': 8,
        'weight_tying': False,
        'alibi': False,
        'rope': True,
        'flash_attention': True,
        'attention_dropout': 0.0,
        'attention_layer_norm': False,
        'clip_qkv': None,
        'include_bias': False,
        'block_type': 'sequential',
        'layer_norm_type': 'rms',
        'layer_norm_with_affine': True,
        'layer_norm_eps': 1e-6,
        'bias_for_layer_norm': False,
        'attention_layer_norm_with_affine': False,
        'activation_type': 'swiglu',
        'residual_dropout': 0.0,
        'embedding_dropout': 0.0,
        'max_sequence_length': 1024,
        'vocab_size': 50280,
        'embedding_size': 50304,
        'eos_token_id': 0,
        'pad_token_id': 1,
        'init_device': 'cuda',
        'init_fn': 'normal',
        'init_std': 0.02,
        'init_cutoff_factor': 3,
    }


def get_train_config(
    run_name,
    save_folder,
    optimizer,
    max_duration,
    train_data_paths,
    eval_datasets,
    learning_rate=3e-4,
    seed=6198,
    scheduler_name='cosine_with_warmup',
    scheduler_t_warmup=6000,
    scheduler_alpha_f=0.1,
    weight_decay=0.1,
    optimizer_betas=(0.9, 0.95),
    optimizer_eps=1e-8,
    global_train_batch_size=256,
    device_train_microbatch_size=32,
    device_eval_batch_size=32,
    eval_interval=5000,
    save_interval_unsharded=10000,
    max_grad_norm=1.0,
    load_path=None,
    restore_dataloader=False,
    reset_optimizer_state=False,
    model_overrides=None,
    **overrides
):
    """
    Create a unified training configuration for both pretraining and CPT.
    
    Args:
        run_name: Name of the training run
        save_folder: Path to save checkpoints
        optimizer: Optimizer name (e.g., 'adamw', 'lionw')
        max_duration: Training duration (e.g., '4e9T' for 4B tokens)
        train_data_paths: List of paths to training data files
        eval_datasets: Dict of evaluation datasets {name: [paths]}
        learning_rate: Learning rate for optimizer
        seed: Random seed
        scheduler_name: Scheduler type ('cosine_with_warmup', 'linear_with_warmup', etc.)
        scheduler_t_warmup: Warmup steps
        scheduler_alpha_f: Final learning rate multiplier
        weight_decay: Weight decay coefficient
        optimizer_betas: Optimizer beta parameters
        optimizer_eps: Optimizer epsilon
        global_train_batch_size: Global batch size
        device_train_microbatch_size: Microbatch size for training
        device_eval_batch_size: Batch size for evaluation
        eval_interval: Steps between evaluations
        save_interval_unsharded: Steps between unsharded checkpoints
        max_grad_norm: Maximum gradient norm for clipping
        load_path: Path to checkpoint to load (for CPT)
        restore_dataloader: Whether to restore dataloader state
        reset_optimizer_state: Whether to reset optimizer state (for CPT)
        model_overrides: Dict of model config overrides
        **overrides: Additional top-level config overrides
    """
    model_config = get_base_model_config()
    if model_overrides:
        model_config.update(model_overrides)
    
    config = {
        'run_name': run_name,
        'seed': seed,
        'dry_run': False,
        
        'model': model_config,
        
        'ddp': {
            'grad_sync_mode': 'batch',
            'find_unused_params': False,
        },
        
        'compile': None,
        
        'optimizer': {
            'name': optimizer,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'eps': optimizer_eps,
            'decay_norm_and_bias': True,
            'decay_embeddings': True,
            'betas': list(optimizer_betas),
            'metrics_log_interval': 10,
        },
        
        'scheduler': {
            'name': scheduler_name,
            't_warmup': scheduler_t_warmup,
            'alpha_f': scheduler_alpha_f,
            'warmup_min_lr': 0,
        },
        
        'tokenizer': {
            'identifier': 'tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json',
            'truncate_direction': 'right',
        },
        
        'save_folder': save_folder,
        'remote_save_folder': None,
        'save_overwrite': False,
        'save_interval_unsharded': save_interval_unsharded,
        'save_num_unsharded_checkpoints_to_keep': -1,
        'load_path': load_path,
        
        'max_duration': max_duration,
        'stop_at': None,
        'global_train_batch_size': global_train_batch_size,
        'device_train_microbatch_size': device_train_microbatch_size,
        
        'precision': 'amp_bf16',
        'distributed_strategy': 'ddp',
        'gen1_gc_interval': 1,
        'max_grad_norm': max_grad_norm,
        'max_grad_norm_ratio': None,
        
        'speed_monitor': {'window_size': 20},
        
        'eval_interval': eval_interval,
        'eval_subset_num_batches': -1,
        'device_eval_batch_size': device_eval_batch_size,
        'evaluators': [{
            'label': 'all-small-ppl-validation',
            'data': {
                'num_workers': 0,
                'drop_last': True,
                'datasets': eval_datasets,
            },
        }],
        
        'data': {
            'pad_direction': 'right',
            'num_workers': 4,
            'drop_last': True,
            'pin_memory': True,
            'prefetch_factor': 8,
            'persistent_workers': True,
            'timeout': 0,
            'instance_filter': {
                'repetition_max_period': 13,
                'repetition_min_period': 1,
                'repetition_max_count': 32,
            },
            'paths': train_data_paths,
        },
    }
    
    # Handle CPT-specific settings
    if load_path is not None:
        config['restore_dataloader'] = restore_dataloader
        config['no_pre_train_checkpoint'] = True
        config['reset_optimizer_state'] = reset_optimizer_state
    
    # Apply any additional overrides
    config.update(overrides)
    return config
