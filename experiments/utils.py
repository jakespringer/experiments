def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary into a single-level dictionary with dot-separated keys.
    
    Args:
        d: Dictionary to flatten (potentially nested)
        parent_key: Key prefix for recursive calls (default: '')
        sep: Separator to use between nested keys (default: '.')
    
    Returns:
        A flattened dictionary with dot-separated keys
    
    Example:
        >>> nested = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
        >>> flatten_dict(nested)
        {'a.b': 1, 'a.c.d': 2, 'e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

