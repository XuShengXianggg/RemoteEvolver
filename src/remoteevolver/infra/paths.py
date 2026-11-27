"""Runtime locations; never write generated tools to installed source."""
import os
from pathlib import Path
def runtime_path(name):
    root = os.environ.get('REMOTEEVOLVER_WORKSPACE')
    if not root:
        raise RuntimeError('REMOTEEVOLVER_WORKSPACE must be set by the launcher')
    return Path(root).resolve() / name
