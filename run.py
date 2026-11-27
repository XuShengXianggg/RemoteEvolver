"""Relocatable entry point; install dependencies before running."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
if '--config' not in sys.argv and not any(a.startswith('--config=') for a in sys.argv):
    sys.argv[1:1] = ['--config', str(ROOT/'configs/default.yaml')]
if not any(a in ('run', 'check-config', '-h', '--help') for a in sys.argv[1:]):
    sys.argv.append('run')
from remoteevolver.cli import main

if __name__ == '__main__':
    main()
