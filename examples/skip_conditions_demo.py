#!/usr/bin/env python3
"""Advanced demo showing different skip conditions.

This example demonstrates:
1. Using 'exists' property for file-based skip checks
2. Overriding 'should_skip()' for custom skip logic
3. How dependencies work with skipped artifacts
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from experiments import Artifact, SlurmExecutor, Task


@dataclass
class DownloadData(Artifact):
    """Download artifact that checks if data file exists."""
    
    dataset_name: str
    data_dir: str = "./data"
    
    @property
    def exists(self) -> bool:
        """Skip if data file already downloaded."""
        data_file = Path(self.data_dir) / f"{self.dataset_name}.tar.gz"
        return data_file.exists()
    
    def construct(self, task):
        task.run_command(f"mkdir -p {self.data_dir}")
        task.run_command(f"wget https://example.com/{self.dataset_name}.tar.gz -O {self.data_dir}/{self.dataset_name}.tar.gz")


@dataclass
class PreprocessData(Artifact):
    """Preprocessing with time-based skip condition."""
    
    raw_data: DownloadData
    max_age_days: int = 7
    
    @property
    def exists(self) -> bool:
        """Skip if preprocessed data exists and is recent enough."""
        output_file = Path("./preprocessed") / f"{self.raw_data.dataset_name}_processed.pkl"
        
        if not output_file.exists():
            return False
        
        # Check if file is recent enough
        file_mtime = datetime.fromtimestamp(output_file.stat().st_mtime)
        age = datetime.now() - file_mtime
        return age < timedelta(days=self.max_age_days)
    
    def construct(self, task):
        task.run_command("mkdir -p ./preprocessed")
        input_path = f"{self.raw_data.data_dir}/{self.raw_data.dataset_name}.tar.gz"
        output_path = f"./preprocessed/{self.raw_data.dataset_name}_processed.pkl"
        task.run_command(f"python preprocess.py --input {input_path} --output {output_path}")


@dataclass
class TrainModel(Artifact):
    """Training with custom should_skip logic."""
    
    data: PreprocessData
    model_type: str
    use_cache: bool = True
    
    def should_skip(self) -> bool:
        """Custom skip logic that checks both exists and use_cache flag."""
        # Only skip if caching is enabled AND the model exists
        if not self.use_cache:
            return False
        
        return self.exists
    
    @property
    def exists(self) -> bool:
        """Check if model checkpoint exists."""
        checkpoint = Path("./models") / self.model_type / "best.ckpt"
        return checkpoint.exists()
    
    def construct(self, task):
        task.run_command(f"mkdir -p ./models/{self.model_type}")
        data_path = f"./preprocessed/{self.data.raw_data.dataset_name}_processed.pkl"
        model_path = f"./models/{self.model_type}"
        task.run_command(f"python train.py --data {data_path} --model {self.model_type} --output {model_path}")


@dataclass
class RunBenchmark(Artifact):
    """Benchmark that never skips (always runs for fresh results)."""
    
    model: TrainModel
    benchmark_suite: str
    
    def should_skip(self) -> bool:
        """Always run benchmarks to get fresh results."""
        return False  # Never skip
    
    def construct(self, task):
        model_path = f"./models/{self.model.model_type}"
        task.run_command(f"python benchmark.py --model {model_path} --suite {self.benchmark_suite}")


if __name__ == "__main__":
    executor = SlurmExecutor(
        artifact_path="./artifacts",
        code_path=".",
        project="skip_demo",
    )
    
    # Create pipeline
    data = DownloadData(dataset_name="imagenet")
    processed = PreprocessData(raw_data=data, max_age_days=7)
    
    # Model with caching enabled (will skip if exists)
    model_cached = TrainModel(data=processed, model_type="resnet", use_cache=True)
    
    # Model with caching disabled (always runs)
    model_no_cache = TrainModel(data=processed, model_type="vit", use_cache=False)
    
    # Benchmarks (always run)
    benchmark1 = RunBenchmark(model=model_cached, benchmark_suite="speed")
    benchmark2 = RunBenchmark(model=model_no_cache, benchmark_suite="accuracy")
    
    # Register stages
    executor.stage("data", [data, processed])
    executor.stage("train", [model_cached, model_no_cache])
    executor.stage("benchmark", [benchmark1, benchmark2])
    
    # Run CLI
    executor.auto_cli()

