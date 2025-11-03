"""Analysis utilities for loading and processing exported experiment data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_export(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the complete exported JSON file.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Dictionary containing project_config, global_config, and artifacts
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {file_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def get_project_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract project configuration from export file.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Project configuration dictionary
    """
    data = load_export(file_path)
    return data.get('project_config', {})


def get_global_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract global configuration from export file.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Global configuration dictionary
    """
    data = load_export(file_path)
    return data.get('global_config', {})


def get_artifacts(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Extract artifacts list from export file.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        List of artifact dictionaries
    """
    data = load_export(file_path)
    return data.get('artifacts', [])


def get_stages(file_path: Union[str, Path]) -> List[str]:
    """Get list of all unique stages in the export.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Sorted list of unique stage names
    """
    artifacts = get_artifacts(file_path)
    stages = set()
    for artifact in artifacts:
        stage_list = artifact.get('stage', [])
        if isinstance(stage_list, list):
            stages.update(stage_list)
        elif isinstance(stage_list, str):
            stages.add(stage_list)
    return sorted(stages)


def get_artifact_types(file_path: Union[str, Path]) -> List[str]:
    """Get list of all unique artifact types in the export.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Sorted list of unique artifact type names
    """
    artifacts = get_artifacts(file_path)
    types = set()
    for artifact in artifacts:
        artifact_type = artifact.get('artifact_type')
        if artifact_type:
            types.add(artifact_type)
    return sorted(types)


def filter_artifacts(
    artifacts: List[Dict[str, Any]],
    stage: Optional[Union[str, List[str]]] = None,
    artifact_types: Optional[Union[str, List[str]]] = None,
    exists: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """Filter artifacts list by various criteria.
    
    Args:
        artifacts: List of artifact dictionaries
        stage: Stage name(s) to filter by (None = no filter)
        artifact_types: Artifact type name(s) to filter by (None = no filter)
        exists: If True, only existing artifacts; if False, only non-existing (None = no filter)
        
    Returns:
        Filtered list of artifact dictionaries
    """
    filtered = artifacts
    
    # Filter by stage
    if stage is not None:
        stages_set = {stage} if isinstance(stage, str) else set(stage)
        filtered = [
            a for a in filtered
            if any(s in stages_set for s in (a.get('stage', []) if isinstance(a.get('stage'), list) else [a.get('stage', '')]))
        ]
    
    # Filter by artifact type
    if artifact_types is not None:
        types_set = {artifact_types} if isinstance(artifact_types, str) else set(artifact_types)
        filtered = [
            a for a in filtered
            if a.get('artifact_type') in types_set
        ]
    
    # Filter by exists
    if exists is not None:
        filtered = [
            a for a in filtered
            if a.get('exists') == exists
        ]
    
    return filtered


def load_artifacts_df(
    file_path: Union[str, Path],
    stage: Optional[Union[str, List[str]]] = None,
    artifact_types: Optional[Union[str, List[str]]] = None,
    exists: Optional[bool] = None,
    flatten: bool = True
) -> 'pd.DataFrame':
    """Load artifacts as a pandas DataFrame with optional filtering.
    
    Args:
        file_path: Path to the exported JSON file
        stage: Stage name(s) to filter by (None = no filter)
        artifact_types: Artifact type name(s) to filter by (None = no filter)
        exists: If True, only existing artifacts; if False, only non-existing (None = no filter)
        flatten: If True, flatten nested dictionaries with dot-separated keys
        
    Returns:
        pandas DataFrame with artifacts as rows
        
    Raises:
        ImportError: If pandas is not installed
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for load_artifacts_df. "
            "Install with: pip install pandas"
        )
    
    artifacts = get_artifacts(file_path)
    
    # Apply filters
    artifacts = filter_artifacts(artifacts, stage=stage, artifact_types=artifact_types, exists=exists)
    
    if not artifacts:
        # Return empty DataFrame with no columns if no artifacts
        return pd.DataFrame()
    
    # Flatten if requested
    if flatten:
        from .utils import flatten_dict
        flattened_artifacts = []
        for artifact in artifacts:
            flat = flatten_dict(artifact)
            flattened_artifacts.append(flat)
        return pd.DataFrame(flattened_artifacts)
    else:
        return pd.DataFrame(artifacts)


def summarize_export(file_path: Union[str, Path]) -> None:
    """Print a summary of the exported data.
    
    Args:
        file_path: Path to the exported JSON file
    """
    data = load_export(file_path)
    
    print("=" * 80)
    print(f"Export Summary: {file_path}")
    print("=" * 80)
    print()
    
    # Project config
    project_config = data.get('project_config', {})
    if project_config:
        print("Project Configuration:")
        for key, value in project_config.items():
            print(f"  {key}: {value}")
        print()
    
    # Artifacts
    artifacts = data.get('artifacts', [])
    print(f"Total Artifacts: {len(artifacts)}")
    print()
    
    # Stages
    stages = get_stages(file_path)
    print(f"Stages ({len(stages)}):")
    for stage in stages:
        count = len([a for a in artifacts if stage in a.get('stage', [])])
        print(f"  {stage}: {count} artifact(s)")
    print()
    
    # Artifact types
    types = get_artifact_types(file_path)
    print(f"Artifact Types ({len(types)}):")
    for artifact_type in types:
        count = len([a for a in artifacts if a.get('artifact_type') == artifact_type])
        print(f"  {artifact_type}: {count} artifact(s)")
    print()
    
    # Exists status
    exists_count = len([a for a in artifacts if a.get('exists') == True])
    not_exists_count = len([a for a in artifacts if a.get('exists') == False])
    unknown_count = len(artifacts) - exists_count - not_exists_count
    
    print("Existence Status:")
    print(f"  Exists: {exists_count}")
    print(f"  Does not exist: {not_exists_count}")
    if unknown_count > 0:
        print(f"  Unknown: {unknown_count}")
    print()
    
    print("=" * 80)


def count_by_stage(file_path: Union[str, Path]) -> Dict[str, int]:
    """Count artifacts by stage.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Dictionary mapping stage names to artifact counts
    """
    artifacts = get_artifacts(file_path)
    counts = {}
    
    for stage in get_stages(file_path):
        counts[stage] = len([a for a in artifacts if stage in a.get('stage', [])])
    
    return counts


def count_by_type(file_path: Union[str, Path]) -> Dict[str, int]:
    """Count artifacts by type.
    
    Args:
        file_path: Path to the exported JSON file
        
    Returns:
        Dictionary mapping artifact types to counts
    """
    artifacts = get_artifacts(file_path)
    counts = {}
    
    for artifact_type in get_artifact_types(file_path):
        counts[artifact_type] = len([a for a in artifacts if a.get('artifact_type') == artifact_type])
    
    return counts


def get_artifact_by_relpath(
    file_path: Union[str, Path],
    relpath: str
) -> Optional[Dict[str, Any]]:
    """Find an artifact by its relpath.
    
    Args:
        file_path: Path to the exported JSON file
        relpath: Relative path of the artifact to find
        
    Returns:
        Artifact dictionary if found, None otherwise
    """
    artifacts = get_artifacts(file_path)
    for artifact in artifacts:
        if artifact.get('relpath') == relpath:
            return artifact
    return None


def export_to_csv(
    file_path: Union[str, Path],
    output_file: Union[str, Path],
    stage: Optional[Union[str, List[str]]] = None,
    artifact_types: Optional[Union[str, List[str]]] = None,
    exists: Optional[bool] = None,
    flatten: bool = True
) -> None:
    """Export artifacts to CSV file.
    
    Args:
        file_path: Path to the exported JSON file
        output_file: Path to output CSV file
        stage: Stage name(s) to filter by (None = no filter)
        artifact_types: Artifact type name(s) to filter by (None = no filter)
        exists: If True, only existing artifacts; if False, only non-existing (None = no filter)
        flatten: If True, flatten nested dictionaries with dot-separated keys
        
    Raises:
        ImportError: If pandas is not installed
    """
    df = load_artifacts_df(file_path, stage=stage, artifact_types=artifact_types, exists=exists, flatten=flatten)
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} artifact(s) to {output_file}")

