"""OLMo pretraining and CPT grid search."""
import os
from dataclasses import dataclass

from experiments import Artifact, ArtifactSet, SlurmExecutor, Task
from .olmo_config import get_train_config


# Dataset configurations
PRETRAIN_DATA_PATHS = [
    f'/data/user_data/iwatts/data/c4_v1_7-dd_ngram_dp_030-qc_cc_en_bin_001-fix_gpt-neox-olmo-dolma-v1_5_part-{i:03d}-00000.npy'
    for i in range(171)
]

CPT_DATA_PATHS = [
    '/data/user_data/iwatts/data/cpt/starcoder/train/starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00000.npy'
]

EVAL_DATASETS = {
    'c4_en-validation': [
        '/data/user_data/iwatts/data/eval_data/v3_small_gptneox20b_c4_en_val_part-0-00000.npy'
    ],
    'starcoder-val': [
        '/data/user_data/iwatts/data/cpt/starcoder/test/starcoder_v0_decontaminated_doc_only_gpt-neox-olmo-dolma-v1_5_part-00-00001.npy'
    ],
}


@dataclass(frozen=True)
class PretrainedModel(Artifact):
    """Pretrained OLMo model."""
    optimizer: str  # adamw or lionw
    tokens_b: int   # 4, 8, 16, 32 (billions)
    
    @property
    def relpath(self) -> str:
        return f'OLMo-20M-{self.tokens_b}B-{self.optimizer}'
    
    def get_requirements(self):
        return {
            'partition': 'general',
            'gpus': '8',
            'cpus': 8,
            'mem': '32G',
            'time': '48:00:00',
        }
    
    def construct(self, builder: Task):
        run_name = f'OLMo-20M-{self.tokens_b}B-Pretrain-{self.optimizer}'
        save_folder = os.path.join(builder.artifact_path, self.relpath)
        
        # Create pretrain config using unified helper function
        config = get_train_config(
            run_name=run_name,
            save_folder=save_folder,
            optimizer=self.optimizer,
            max_duration=f'{self.tokens_b}e9T',
            train_data_paths=PRETRAIN_DATA_PATHS,
            eval_datasets=EVAL_DATASETS,
            learning_rate=3e-4,
            seed=6198,
            scheduler_name='cosine_with_warmup',
            scheduler_t_warmup=6000,
            scheduler_alpha_f=0.1,
            weight_decay=0.1,
            global_train_batch_size=256,
            device_train_microbatch_size=32,
            device_eval_batch_size=32,
            eval_interval=5000,
            save_interval_unsharded=10000,
        )
        
        # Save config
        config_path = os.path.join(builder.artifact_path, self.relpath, 'pretrain_config.yaml')
        builder.create_yaml_file(config_path, config)
        
        # Setup conda environment and run training
        train_script = os.path.join(builder.code_path, 'scripts/train.py')
        builder.run_command(
            'source ~/miniconda3/etc/profile.d/conda.sh && '
            'conda activate olmo && '
            f'torchrun --nproc_per_node=8 {train_script} {config_path} --save_overwrite'
        )


@dataclass(frozen=True)
class CPTModel(Artifact):
    """Continued pretraining model."""
    pretrained: PretrainedModel
    tokens_b: int          # 4, 8, 16, 32, 64 (billions)
    learning_rate: float   # 1e-3, 2e-4, 4e-5
    
    @property
    def relpath(self) -> str:
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        return f'OLMo-20M-{self.tokens_b}B-{self.pretrained.optimizer}-lr{lr_str}'
    
    def get_requirements(self):
        return {
            'partition': 'general',
            'gpus': '4',
            'cpus': 8,
            'mem': '32G',
            'time': '6:00:00',
        }
    
    def construct(self, builder: Task):
        lr_str = f'{self.learning_rate:.0e}'.replace('e-0', 'e-')
        run_name = f'OLMo-20M-{self.tokens_b}B-CPT-{self.pretrained.optimizer}-lr{lr_str}'
        save_folder = os.path.join(builder.artifact_path, self.relpath)
        
        # Get checkpoint path from pretrained model
        pretrain_checkpoint = os.path.join(
            builder.artifact_path, 
            self.pretrained.relpath,
            'step61046-unsharded'  # Adjust based on your checkpoint naming
        )
        
        # Create CPT config using unified helper function
        config = get_train_config(
            run_name=run_name,
            save_folder=save_folder,
            optimizer=self.pretrained.optimizer,
            max_duration=f'{self.tokens_b}e7T',
            train_data_paths=CPT_DATA_PATHS,
            eval_datasets=EVAL_DATASETS,
            learning_rate=self.learning_rate,
            seed=42,
            scheduler_name='linear_with_warmup',
            scheduler_t_warmup=0,
            scheduler_alpha_f=0,
            weight_decay=0,
            global_train_batch_size=64,
            device_train_microbatch_size=16,
            device_eval_batch_size=16,
            eval_interval=5000,
            save_interval_unsharded=100,
            load_path=pretrain_checkpoint,
            reset_optimizer_state=True,
        )
        
        # Save config
        config_path = os.path.join(builder.artifact_path, self.relpath, 'cpt_config.yaml')
        builder.create_yaml_file(config_path, config)
        
        # Setup conda environment and run CPT
        train_script = os.path.join(builder.code_path, 'scripts/train.py')
        builder.run_command(
            'source ~/miniconda3/etc/profile.d/conda.sh && '
            'conda activate olmo && '
            f'torchrun --master_port=29501 --nproc_per_node=4 {train_script} {config_path} --save_overwrite'
        )


# Create pretrained models: 2 optimizers x 4 token counts = 8 models
pretrained_models = ArtifactSet.from_product(
    cls=PretrainedModel,
    params={
        'optimizer': ['adamw', 'lionw'],
        'tokens_b': [4, 8, 16, 32],
    }
)

# Create CPT models: each pretrained model x 5 token counts x 3 learning rates
cpt_models = []
for pretrained in pretrained_models:
    for tokens_b in [4, 8, 16, 32, 64]:
        for lr in [1e-3, 2e-4, 4e-5]:
            cpt_models.append(CPTModel(
                pretrained=pretrained,
                tokens_b=tokens_b,
                learning_rate=lr,
            ))

cpt_models = ArtifactSet(cpt_models)

# Setup executor
executor = SlurmExecutor(
    project='olmo-gridsearch',
    artifact_path='/data/user_data/iwatts/catastrophic-forgetting',
    code_path='/home/iwatts/catastrophic-forgetting',
)

# Register stages
executor.stage('pretrain', pretrained_models)
executor.stage('cpt', cpt_models)

if __name__ == '__main__':
    executor.auto_cli()