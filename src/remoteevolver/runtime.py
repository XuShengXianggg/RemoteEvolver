"""Run-scoped training/evaluation; independent Judge and durable task records."""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def workspace_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    f = (root/'.run.lock').open('a+b')
    f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError('another process owns this workspace')
    try:
        yield
    finally:
        f.close()


def copy_state(root, target):
    target.mkdir(parents=True, exist_ok=False)
    for name in ('tools', 'memory'):
        if (root/name).exists():
            shutil.copytree(root/name, target/name)


def restore_state(snapshot, root):
    # Only managed mutable state is replaced; logs, results and artifacts remain.
    for name in ('tools', 'memory'):
        p = root/name
        if p.exists():
            assert p.resolve().parent == root.resolve()
            shutil.rmtree(p)
        if (snapshot/name).exists():
            shutil.copytree(snapshot/name, p)


class ReadOnlyWorkflow:
    def __init__(self, inner): self.inner = inner
    def __getattr__(self, name): return getattr(self.inner, name)
    def compile_and_admit(self, *args, **kwargs): return None
    def record_outcome(self, *args, **kwargs): return None


def make_agent(cfg, root, *, training):
    from remoteevolver.core import Agent
    from remoteevolver.infra.config import (Config, LLMConfig, MCPServerConfig, ToolsConfig,
        AgentConfig, MemoryConfig, LongTermConfig, ShortTermConfig, ReflectionConfig, PromptConfig)
    from remoteevolver.evolve.workflow.config import WorkflowConfig
    servers = [MCPServerConfig(name='local', command=sys.executable, args=['-m','remoteevolver.mcpx'])]
    for server in cfg.get('mcp_servers', []):
        servers.append(MCPServerConfig(**server))
    tool_options = dict(cfg.get('tools', {}))
    tool_options['certification_dir'] = str(root/'tools/.certifications')
    main = cfg['main']
    config = Config(llm=LLMConfig(provider='openai', model=main['model'],
        api_key=os.environ[main['api_key_env']], base_url=main['base_url'],
        temperature=float(main.get('temperature',0))), mcp_servers=servers,
        tools=ToolsConfig(**tool_options), agent=AgentConfig(**cfg.get('agent',{})),
        memory=MemoryConfig(short_term=ShortTermConfig(**cfg.get('short_term',{})),
            long_term=LongTermConfig(path=str(root/'memory/history.json'))),
        reflection=ReflectionConfig(**cfg.get('reflection',{})), prompt=PromptConfig(**cfg.get('prompts',{})))
    policy = dict(cfg.get('workflow',{}))
    agent = Agent(config=config, evolve_enabled=training and cfg.get('tool_evolution',True),
                  workflow_enabled=cfg.get('skill_evolution',True), workflow_config=WorkflowConfig(**policy),
                  verbose=cfg.get('verbose',False))
    if not training:
        from .admission import _CaptureMemory, _NoopReflector
        agent._impl.long_term = _CaptureMemory()
        agent._impl.reflector = _NoopReflector()
        if agent._workflow:
            agent._impl.workflow = ReadOnlyWorkflow(agent._workflow)
    return agent


def run(cfg, tasks_path, *, mode='train'):
    from .config import prepare_environment
    prepare_environment(cfg)  # Must precede imports with run-scoped constants.
    from .judge import AnswerJudge, JudgeOracle
    root = Path(cfg['workspace']).resolve()
    tasks_path = Path(tasks_path).resolve()
    tasks = [json.loads(s) for s in Path(tasks_path).read_text(encoding='utf8').splitlines() if s.strip()]
    for task in tasks:
        task['inputs'] = [str((tasks_path.parent / p).resolve()) for p in task.get('inputs', [])]
    ids = [str(t['id']) for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate task IDs')
    for t in tasks:
        if not isinstance(t.get('question'),str) or 'reference' not in t:
            raise ValueError('every task requires id, question and reference')
    fingerprint = hashlib.sha256(json.dumps({'config':cfg,'tasks':tasks,'mode':mode},sort_keys=True).encode()).hexdigest()
    with workspace_lock(root):
        for name in ('logs','results','artifacts','memory','snapshots'):
            (root/name).mkdir(exist_ok=True)
        identity = root/'run.json'
        if identity.exists():
            if json.loads(identity.read_text())['fingerprint'] != fingerprint:
                raise ValueError('config/tasks changed: use a new workspace')
        else:
            identity.write_text(json.dumps({'fingerprint':fingerprint,'mode':mode},indent=2),encoding='utf8')
        ledger = root/'results/tasks.jsonl'
        rows=[]
        if ledger.exists():
            # A torn last line is preserved separately, then removed for safe append.
            data=ledger.read_bytes(); good=0
            lines = data.splitlines(keepends=True)
            for line_index, line in enumerate(lines):
                try: row=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    if line_index != len(lines)-1 or line.endswith(b'\n'):
                        raise ValueError('corrupt committed result ledger; manual review required')
                    (root/'logs/torn-result-line.bin').write_bytes(data[good:]); break
                rows.append(row); good += len(line)
            if good != len(data): ledger.write_bytes(data[:good])
        done={r['id'] for r in rows}
        if rows:
            restore_state(root/rows[-1]['snapshot'],root)
        elif (root/'snapshots/initial').exists():
            restore_state(root/'snapshots/initial',root)
        else:
            if cfg.get('snapshot'):
                restore_state(Path(cfg['snapshot']), root)
            elif cfg.get('initial_tools_dir'):
                shutil.copytree(cfg['initial_tools_dir'],root/'tools',dirs_exist_ok=True)
            (root/'tools').mkdir(exist_ok=True)
            (root/'tools/__init__.py').touch(exist_ok=True)
            copy_state(root,root/'snapshots/initial')
        judge=AnswerJudge(cfg['judge'],os.environ[cfg['judge']['api_key_env']])
        for index,task in enumerate(tasks):
            task_id=str(task['id'])
            if task_id in done: continue
            tag=f'{index+1:04d}-'+hashlib.sha256(task_id.encode()).hexdigest()[:10]
            output=root/'artifacts'/tag; output.mkdir(exist_ok=True)
            os.environ['EVOAGENT_OUTPUT_DIR']=str(output)
            oracle=JudgeOracle(judge,task['question'],task['reference'])
            started=time.time(); agent=None
            try:
                agent=make_agent(cfg,root,training=mode=='train')
                if agent._workflow:
                    agent._workflow.set_retrieval_context(task['question'], [Path(p).name for p in task.get('inputs',[])])
                # Reference never enters the task prompt.
                prompt=task['question']
                if task.get('inputs'): prompt+='\nInput files:\n'+'\n'.join(task['inputs'])
                prompt+='\nWritable output directory:\n'+str(output)
                answer=agent.run(prompt,expected=oracle if mode=='train' else None)
                if oracle.errors:
                    raise oracle.errors[0]
                oracle.matches(answer)
                result=oracle.cache[('task',answer)]
                usage=agent._llm.usage_since(0) if hasattr(agent._llm,'usage_since') else {}
                admissions=getattr(agent._workflow,'last_admission_results',[]) if agent._workflow else []
                append(root/'logs/skill-admission.jsonl',{'id':task_id,'candidates':admissions})
                append(root/'logs/judge.jsonl',{'id':task_id,'evaluations':oracle.records})
                if mode!='train':
                    append(root/'logs/test-trajectories.jsonl',{'id':task_id,'steps':agent._impl.long_term.steps})
                agent.close(); agent=None
                snapshot=root/'snapshots'/tag
                if snapshot.exists():
                    assert snapshot.parent==root/'snapshots'
                    shutil.rmtree(snapshot)  # Uncommitted snapshot from interrupted task.
                copy_state(root,snapshot)
                append(ledger,{'id':task_id,'index':index,'answer':answer,**result,
                    'main_usage':usage,'elapsed_seconds':time.time()-started,
                    'snapshot':str(snapshot.relative_to(root)), 'status':'ok' if result['correct'] else 'agent_failure'})
                print(f'{index+1}/{len(tasks)} {task_id}: score={result["score"]:.2f}',flush=True)
            except Exception as exc:
                append(root/'logs/failures.jsonl',{'id':task_id,'type':type(exc).__name__,
                    'status':'interrupted','elapsed_seconds':time.time()-started})
                # Stop at the first incomplete task. Resume restores last committed state.
                raise
            finally:
                if agent is not None: agent.close()
