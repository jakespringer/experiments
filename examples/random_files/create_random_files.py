from __future__ import annotations

import os

from experiments.runlib import (
    get_project_config,
    get_relpath,
    push_to_gs,
    check_exists_gs,
    download_from_gs,
)


def main() -> None:
    project = get_project_config()
    rel = get_relpath()

    # Expect project to define local_path and remote_path roots
    local_root = os.path.join(project.local_path, rel)
    remote_root = os.path.join(project.remote_path, rel)

    os.makedirs(local_root, exist_ok=True)

    stories = {
        "chapter1.txt": "Once upon a time, in a tiny cluster queue...\n",
        "chapter2.txt": "Jobs lined up, eager to compute.\n",
        "chapter3.txt": "Artifacts were crafted with care.\n",
        "chapter4.txt": "And results were pushed to the cloud.\n",
        "nested/chapter5.txt": "Nested files are also supported.\n",
        "nested/nested/chapter6.txt": "Nested files are also supported.\n",
    }

    for i, (name, text) in enumerate(stories.items()):
        os.makedirs(os.path.join(local_root, os.path.dirname(name)), exist_ok=True)
        with open(os.path.join(local_root, name), "w") as f:
            f.write(text)
        # Push only the contents each time; preserve the local directory itself

        if i % 2 == 0:
            continue

        # A bunch of printing
        print('Before push:')
        print(f"+ ls -l {local_root}")
        os.system(f"ls -l {local_root}")
        print('-'*40 + ' push_to_gs ' + '-'*40)

        # ACTUAL PUSH COMMAND HERE
        push_to_gs(os.path.join(local_root, "*"), os.path.join(remote_root, "."), directory=True, concurrent=True)
        
        # A bunch of printing
        print('After push:')
        print(f"+ ls -l {local_root}")
        os.system(f"ls -l {local_root}")
        print('='*120)

    remote_files = [
        os.path.join(remote_root, name) for name in stories.keys()
    ]
    print('Exists:', check_exists_gs(remote_files))

    print('Downloading:')
    download_from_gs(remote_root, local_root, directory=True, concurrent=True)
    print('After download:')
    print(f"+ ls -l {local_root}")
    os.system(f"ls -l {local_root}")
    print('='*120)


if __name__ == "__main__":
    main()


