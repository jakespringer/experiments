import os
from dataclasses import dataclass

from experiments import (
    Artifact, 
    ArtifactSet, 
    ArgumentProduct, 
    PrintExecutor, 
    SlurmExecutor,
    Task, 
)


@dataclass(frozen=True)
class PretrainedModel(Artifact):
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_layers: int
    num_heads: int
    num_hidden: int
    num_classes: int
    dropout: float
    optimizer: str

    def get_requirements(self):
        return {
            'gpus': 'A6000:4',
            'cpus': '8',
        }

    def construct(self, builder: Task):
        builder.create_yaml_file(
            os.path.join(builder.artifact_path, self.relpath, 'train.yaml'),
            {
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'num_hidden': self.num_hidden,
                'num_classes': self.num_classes,
                'dropout': self.dropout,
                'optimizer': self.optimizer,
                'output_path': os.path.join(builder.artifact_path, self.relpath),
            }
        )
        builder.run_command(
            f'echo python {os.path.join(builder.code_path, "pretrain.py")}', 
            kwargs={
                'config_path': 'pretrained/train.yaml',
            },
            vformat='{v}',
            kwformat='--{k} \'{v}\'',
        )
        builder.upload_to_gs(
            os.path.join(builder.artifact_path, self.relpath),
            os.path.join(builder.gs_path, self.relpath),
        )


@dataclass(frozen=True)
class FinetuningDataset(Artifact):
    dataset_name: str
    dataset_path: str

    @property
    def path(self) -> str:
        return f"FinetuningDataset/{self.dataset_name}"

    def construct(self, builder: Task):
        builder.run_command(
            f'echo python {os.path.join(builder.code_path, "prepare_finetuning_datasets.py")}',
            kwargs={
                'dataset_name': self.dataset_name,
                'dataset_path': self.dataset_path,
                'output_path': os.path.join(builder.artifact_path, self.path),
            },
            vformat='{v}',
            kwformat='--{k} \'{v}\'',
        )


@dataclass(frozen=True)
class FinetunedModel(Artifact):
    pretrained_model: PretrainedModel
    finetuning_dataset: FinetuningDataset
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_layers: int
    num_heads: int
    num_hidden: int
    num_classes: int
    dropout: float
    optimizer: str

    def get_requirements(self):
        return {
            'gpus': 'A6000:1',
            'cpus': '2',
        }

    def construct(self, builder: Task):
        builder.run_command(
            f'echo python {os.path.join(builder.code_path, "finetune.py")}',
            kwargs={
                'base_model': os.path.join(builder.artifact_path, self.pretrained_model.relpath),
                'dataset_path': os.path.join(builder.artifact_path, self.finetuning_dataset.relpath),
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'num_hidden': self.num_hidden,
                'num_classes': self.num_classes,
                'dropout': self.dropout,
                'optimizer': self.optimizer,
                'output_path': os.path.join(builder.artifact_path, self.relpath),
            },
            vformat='{v}',
            kwformat='--{k} \'{v}\'',
        )
        builder.upload_to_gs(
            os.path.join(builder.artifact_path, self.relpath),
            os.path.join(builder.gs_path, self.relpath),
        )

pretrained_models = ArtifactSet.from_product(
    cls=PretrainedModel,
    params={
        'learning_rate': [0.001, 0.0001],
        'num_epochs': 1,
        'batch_size': [32, 64],
        'num_layers': [6],
        'num_heads': [8],
        'num_hidden': [256],
        'num_classes': [2],
        'dropout': [0.2],
        'optimizer': ['adam', 'sgd']
    }
)

finetuning_datasets = ArtifactSet.from_product(
    cls=FinetuningDataset,
    params=ArgumentProduct(
        dataset_name=['cifar10', 'cifar100'],
    ).map(
        lambda variables: FinetuningDataset(
            dataset_name=variables['dataset_name'],
            dataset_path=f'finetuning/{variables["dataset_name"]}'
        )
    )
)

finetuned_models = ArtifactSet.join_product(pretrained_models, finetuning_datasets).map(
    lambda pretrained_model, finetuning_dataset: FinetunedModel(
        pretrained_model=pretrained_model,
        finetuning_dataset=finetuning_dataset,
        learning_rate=0.001,
        num_epochs=1,
        batch_size=32,
        num_layers=6,
        num_heads=8,
        num_hidden=256,
        num_classes=2,
        dropout=0.2,
        optimizer='adam'
    )
)

executor = SlurmExecutor(
    artifact_path='/user_data/jspringe/projects/dummy',
    code_path='/home/jspringe/projects/dummy/src',
    gs_path='gs://jspringe/projects/dummy',
)

executor.stage(
    'pretrain',
    pretrained_models,
)

executor.stage(
    'prepare_finetuning_datasets',
    finetuning_datasets
)

executor.stage(
    'finetune',
    finetuned_models
)

if __name__ == '__main__':
    executor.auto_cli()