#!/usr/bin/env python3
"""Demo of artifact exists functionality.

This example shows how to use the 'exists' property to skip artifacts
that have already been executed.
"""

from dataclasses import dataclass
from pathlib import Path
from experiments import Artifact, SlurmExecutor, Task, Project


@dataclass
class TrainModel(Artifact):
    """Training artifact that checks if model file exists."""
    
    model_name: str
    epochs: int
    output_dir: str = "./models"
    
    @property
    def exists(self) -> bool:
        """Check if the trained model already exists."""
        model_path = Path(self.output_dir) / self.model_name / "checkpoint.pt"
        return model_path.exists()
    
    def construct(self, task):
        # Create output directory
        output_path = f"{self.output_dir}/{self.model_name}"
        task.run_command(f"mkdir -p {output_path}")
        
        # Simulate training
        task.run_command(
            f"python train.py",
            kwargs={
                "model": self.model_name,
                "epochs": self.epochs,
                "output": output_path,
            }
        )
        
        # Create checkpoint file to mark completion
        task.run_command(f"touch {output_path}/checkpoint.pt")


@dataclass  
class EvaluateModel(Artifact):
    """Evaluation artifact that depends on trained model."""
    
    trained_model: TrainModel
    test_dataset: str
    
    def construct(self, task):
        model_path = f"{self.trained_model.output_dir}/{self.trained_model.model_name}"
        task.run_command(
            f"python evaluate.py",
            kwargs={
                "model": model_path,
                "dataset": self.test_dataset,
            }
        )


if __name__ == "__main__":
    # Initialize project and create executor (paths can be configured via project.json)
    Project.init('exists_demo')
    executor = SlurmExecutor()
    
    # Create artifacts
    # If checkpoint.pt exists for resnet50, it will be skipped
    model1 = TrainModel(model_name="resnet50", epochs=10)
    model2 = TrainModel(model_name="vit", epochs=20)
    
    eval1 = EvaluateModel(trained_model=model1, test_dataset="imagenet")
    eval2 = EvaluateModel(trained_model=model2, test_dataset="imagenet")
    
    # Register stages
    executor.stage("train", [model1, model2])
    executor.stage("eval", [eval1, eval2])
    
    # Run CLI
    # Try:
    #   python exists_demo.py print        # See all commands
    #   python exists_demo.py drylaunch    # Dry run (skips existing)
    #   python exists_demo.py launch       # Actually run (skips existing)
    executor.auto_cli()

