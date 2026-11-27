"""Run-scoped tool package. No historical generated tools are shipped."""
import os
from pathlib import Path
root = Path(os.environ['REMOTEEVOLVER_TOOLS_DIR']).resolve()
__path__.append(str(root))
ledger = root / '__init__.py'
if ledger.exists():
    exec(compile(ledger.read_text(encoding='utf-8'), str(ledger), 'exec'), globals())
