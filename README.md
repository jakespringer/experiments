# Experiments Framework

A lightweight Python framework for defining and running large-scale experiments on Slurm clusters.

## What is it?

This framework lets you:
- **Define experiments as artifacts** with automatic dependency tracking
- **Run massive grid searches** with minimal boilerplate
- **Manage Slurm jobs** through a simple CLI
- **Track experiment history** and view logs easily

Instead of writing bash scripts and managing job dependencies manually, you define Python dataclasses that represent your experiments. The framework handles job submission, dependencies, and tracking automatically.

## Quick Start

### 1. Define Your Artifacts

```python
from dataclasses import dataclass
from experiments import Artifact, ArtifactSet, SlurmExecutor, Task

@dataclass(frozen=True)
class PretrainedModel(Artifact):
    learning_rate: float
    num_epochs: int
    
    def get_requirements(self):
        return {
            'partition': 'gpu',
            'gpus': 'A100:4',
            'cpus': 8,
            'time': '24:00:00',
        }
    
    def construct(self, builder: Task):
        # Create config file
        builder.create_yaml_file(
            f'{builder.artifact_path}/{self.relpath}/config.yaml',
            {'lr': self.learning_rate, 'epochs': self.num_epochs}
        )
        # Run training
        builder.run_command(
            f'python train.py --config {builder.artifact_path}/{self.relpath}/config.yaml'
        )
```

### 2. Create a Grid Search

```python
# Create 12 models (3 learning rates × 4 epoch counts)
models = ArtifactSet.from_product(
    cls=PretrainedModel,
    params={
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'num_epochs': [10, 20, 50, 100],
    }
)

# Setup executor
executor = SlurmExecutor(
    project='my-experiment',
    artifact_path='/data/outputs',
    code_path='/home/user/code',
)

executor.stage('pretrain', models)

if __name__ == '__main__':
    executor.auto_cli()
```

### 3. Launch and Manage

```bash
# Preview what will run
python my_experiment.py drylaunch

# Launch all jobs
python my_experiment.py launch

# Check status
python my_experiment.py history

# View logs for a specific job
python my_experiment.py cat 12345

# Cancel jobs
python my_experiment.py cancel
```

## Key Features

### Automatic Dependencies

Artifacts can depend on other artifacts. The framework automatically:
- Computes the correct execution order
- Submits jobs in dependency-ordered tiers
- Sets up Slurm dependencies between tiers

```python
@dataclass(frozen=True)
class FinetunedModel(Artifact):
    base_model: PretrainedModel  # Dependency!
    dataset: str
    
    def construct(self, builder: Task):
        # Access base model path
        base_path = f'{builder.artifact_path}/{self.base_model.relpath}'
        builder.run_command(f'python finetune.py --base {base_path}')
```

### Smart Job Grouping

Artifacts with different resource requirements are automatically grouped into separate Slurm array jobs:

```python
# These will become 2 separate jobs:
# - Job 1: Models with 4 GPUs
# - Job 2: Models with 8 GPUs
models = [
    Model(gpus=4, ...),  # Group 1
    Model(gpus=4, ...),  # Group 1
    Model(gpus=8, ...),  # Group 2
]
```

### Configuration Management

First run creates `~/.experiments/config.json`:

```json
{
  "log_directory": "~/.experiments/logs",
  "default_slurm_args": {
    "gpu": {
      "time": "24:00:00",
      "cpus": 8,
      "requeue": true
    },
    "cpu": {
      "time": "4:00:00",
      "cpus": 1
    }
  }
}
```

Customize defaults per partition. Artifact-specific requirements override these.

## CLI Commands

| Command | Description |
|---------|-------------|
| `launch [stages...]` | Submit jobs to Slurm |
| `drylaunch [stages...]` | Preview without submitting |
| `history` | View all submitted jobs |
| `cat <job_id>` | View logs for a job |
| `cancel [stages...]` | Cancel running jobs |

## Advanced Usage

### Multi-Stage Experiments

```python
# Stage 1: Pretrain
pretrained = ArtifactSet.from_product(...)

# Stage 2: Finetune (depends on pretrained)
finetuned = ArtifactSet.join_product(pretrained, datasets).map(
    lambda model, dataset: FinetunedModel(base_model=model, dataset=dataset)
)

executor.stage('pretrain', pretrained)
executor.stage('finetune', finetuned)

# Launch only finetuning (pretrain must have completed)
# python script.py launch finetune
```

### Custom File Creation

```python
def construct(self, builder: Task):
    # YAML file
    builder.create_yaml_file('config.yaml', {...})
    
    # Plain text file
    builder.create_file('script.sh', '#!/bin/bash\necho hello')
    
    # Binary file
    builder.create_file('data.bin', b'\x00\x01\x02')
```

### Commands with Arguments

```python
builder.run_command(
    'python train.py',
    kwargs={
        'config': 'config.yaml',
        'output': '/data/output',
        'lr': 0.001,
    },
    kwformat='--{k}={v}'  # Produces: --config=config.yaml --output=/data/output --lr=0.001
)
```

### Cloud Storage Integration

```python
# Upload results to Google Cloud Storage
builder.upload_to_gs(
    '/local/path/model',
    'gs://bucket/models/my-model'
)

# Download data from GCS
builder.download_from_gs(
    'gs://bucket/data/dataset.tar',
    '/local/path/dataset.tar'
)
```

## Installation

```bash
pip install -e .
```

## Project Structure

```
experiments/
├── artifact.py      # Artifact and ArtifactSet classes
├── executor.py      # Slurm executor and task builders
└── cli.py           # Command-line interface

~/.experiments/
├── config.json      # Global configuration
├── logs/            # Job log files
└── projects/        # Per-project job tracking
    └── my-project/
        └── pretrain/
            └── jobs.json
```

## Examples

See `examples/` for complete examples:
- `multistage_training.py` - Multi-stage ML pipeline
- `olmo/gridsearch.py` - Large-scale LLM training

## Requirements

- Python 3.10+
- Slurm cluster
- PyYAML

## Tips

- Use `@dataclass(frozen=True)` for immutable, hashable artifacts
- Artifact paths are automatically generated from their parameters
- Jobs are tracked in `~/.experiments/projects/{project}/` for easy history viewing
- Use `drylaunch` to verify your setup before launching expensive jobs

## License

MIT