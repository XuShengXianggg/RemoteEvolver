"""Explicit configuration. Paths are relative to the configuration file."""
from pathlib import Path
import os
import yaml


def load(path, *, require_keys=True):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf8'))
    if not isinstance(cfg, dict):
        raise ValueError('configuration must be a mapping')
    base = path.parent
    if cfg.get('env_file'):
        from dotenv import load_dotenv
        load_dotenv(base/cfg['env_file'], override=False)
    for role in ('main', 'judge', 'embedding'):
        spec = cfg.get(role)
        if not isinstance(spec, dict):
            raise ValueError(f'missing {role} configuration')
        for name in ('model', 'base_url'):
            value = str(spec.get(name, '')).strip()
            if value.startswith('${') and value.endswith('}'):
                value = os.environ.get(value[2:-1], '').strip()
            if not value and (require_keys or not str(spec.get(name, '')).startswith('${')):
                raise ValueError(f'{role}.{name} is required; no implicit fallback')
            spec[name] = value
        env = spec.get('api_key_env')
        if not env or (require_keys and not os.environ.get(env)):
            raise ValueError(f'{role}.api_key_env must reference a populated environment variable')
    for name in ('workspace', 'tool_config', 'tasks'):
        if cfg.get(name):
            cfg[name] = str((base/cfg[name]).resolve())
    if not cfg.get('workspace'):
        raise ValueError('workspace must be explicitly configured')
    for name in ('initial_tools_dir', 'snapshot'):
        if cfg.get(name):
            cfg[name] = str((base/cfg[name]).resolve())
    if cfg['judge'].get('prompt_file'):
        cfg['judge']['prompt_file'] = str((base/cfg['judge']['prompt_file']).resolve())
    for name, value in cfg.get('prompts', {}).items():
        if name not in ('system', 'reflection', 'tool_use'):
            raise ValueError(f'unknown prompt: {name}')
        cfg['prompts'][name] = str((base / value).resolve())
    for name in ('tasks', 'tool_config', 'initial_tools_dir', 'snapshot'):
        if cfg.get(name) and not Path(cfg[name]).exists():
            raise ValueError(f'{name} does not exist: {cfg[name]}')
    if cfg.get('mode', 'train') not in ('train', 'test'):
        raise ValueError('mode must be train or test')
    return cfg


def prepare_environment(cfg):
    root = Path(cfg['workspace'])
    os.environ['REMOTEEVOLVER_WORKSPACE'] = str(root)
    os.environ['REMOTEEVOLVER_TOOLS_DIR'] = str(root/'tools')
    os.environ['REMOTEEVOLVER_TOOL_CONFIG'] = cfg.get('tool_config', '')
    os.environ['RSEVO_EMBEDDING_MODEL'] = cfg['embedding']['model']
    os.environ['RSEVO_EMBEDDING_API_KEY'] = os.environ[cfg['embedding']['api_key_env']]
    os.environ['RSEVO_EMBEDDING_API_BASE_URL'] = cfg['embedding']['base_url']
    os.environ['EVOAGENT_LLM_CHAT_TEMPLATE_KWARGS'] = '{}'
    os.environ['EVOAGENT_LLM_REQUIRED_TOOL_THINKING_POLICY'] = 'preserve'
    os.environ['EVOAGENT_LLM_MAX_COMPLETION_TOKENS'] = '0'
    os.environ['EVOAGENT_LLM_TRANSPORT_TIMEOUT_SECONDS'] = '120'
    os.environ['EVOAGENT_LLM_CALL_WALL_TIMEOUT_SECONDS'] = '0'
    os.environ['EVOAGENT_LLM_RETRY_TIMEOUTS'] = '1'
    os.environ['EVOAGENT_EVOLVE_CHECKPOINT_DIR'] = str(root/'tool-checkpoints')
    options = cfg.get('runtime', {})
    from .settings import RUNTIME_SETTINGS
    unknown = set(options) - set(RUNTIME_SETTINGS)
    if unknown:
        raise ValueError(f'unknown runtime options: {sorted(unknown)}')
    for name, (env, default, kind) in RUNTIME_SETTINGS.items():
        value = kind(options.get(name, default))
        if kind in (int, float) and (value < 0 or (value == 0 and name != 'embedding_max_retries')):
            raise ValueError(f'runtime.{name} must be positive')
        os.environ[env] = str(value)
    os.environ['RSEVO_INFRA_EVENTS_PATH'] = str(root/'logs/infrastructure.jsonl')
    # Only documented process options can override the inherited client defaults.
    allowed = {'EVOAGENT_LLM_CHAT_TEMPLATE_KWARGS', 'EVOAGENT_LLM_REQUIRED_TOOL_THINKING_POLICY',
               'EVOAGENT_LLM_MAX_COMPLETION_TOKENS', 'EVOAGENT_LLM_TRANSPORT_TIMEOUT_SECONDS',
               'EVOAGENT_LLM_CALL_WALL_TIMEOUT_SECONDS', 'EVOAGENT_LLM_RETRY_TIMEOUTS'}
    for k,v in cfg.get('main_process_options', {}).items():
        if k not in allowed:
            raise ValueError(f'unsupported main process option: {k}')
        os.environ[k] = str(v)
