from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from experiments import Artifact, Task, Project, SlurmExecutor
Project.init('demo_random_files')


@dataclass(frozen=True)
class RandomFiles(Artifact):
    """Example artifact that generates a handful of small text files.

    The actual file creation is performed by the runtime script
    `examples/random_files/create_random_files.py`, which reads
    runtime config via runlib and then pushes contents to GCS after
    each file.
    """

    run_id: str = "demo"

    @property
    def relpath(self) -> str:
        return f"RandomFiles/{self.run_id}"

    def get_requirements(self) -> Dict[str, Any]:
        # Very small job; tune as needed for your cluster
        return {
            "partition": "general",
            "cpus": 1,
            "time": "00:10:00",
        }

    def construct(self, builder: Task) -> None:
        # Execute the runtime script; it will discover local/remote paths and
        # the relpath from exported environment variables.
        builder.run_command(
            f"python {Project.config.code_path}/examples/random_files/create_random_files.py"
        )


random_files = RandomFiles()
executor = SlurmExecutor()
executor.stage('random_files', random_files)

if __name__ == '__main__':
    executor.auto_cli()