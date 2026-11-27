"""Only the configured run's tool directory may be changed by registration."""
from pathlib import Path
from ...infra.paths import runtime_path


def assert_registration_target_is_evolution_only(tools_dir: Path | str) -> None:
    target = Path(tools_dir).resolve()
    if target != runtime_path('tools').resolve():
        raise ValueError(f'registration target is outside the configured run: {target}')
