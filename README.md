# RemoteEvolver

**Expanding Earth Observation Capabilities through Agent Harness Evolution**

[Paper](https://arxiv.org/abs/2611.27110) | [PDF](https://arxiv.org/pdf/2611.27110)

RemoteEvolver is an agent framework for expanding Earth observation capabilities through **tool evolution** and **skill evolution**. When a task requires missing capabilities, the agent searches GitHub for suitable tools, adapts and registers them for execution. Successful trajectories are distilled into reusable skills, which are retrieved to guide planning and execution on subsequent tasks.

## Installation

Requirements: **Python 3.10+** and **Git**. The main model must support OpenAI-compatible tool calling. Tool-specific dependencies and hardware requirements depend on the acquired repositories.

Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, replace the activation command with `.venv\Scripts\Activate.ps1`.

## Quick Start

1. Copy [.env.example](.env.example) to `.env` and fill in the model names, API base URLs, and keys for the **main model**, **judge**, and **embedding service**. If `.env` already exists, edit it directly. `GITHUB_TOKEN` is optional.
2. Set your task file, workspace, and initial tools in [configs/default.yaml](configs/default.yaml). See [examples/tasks.jsonl](examples/tasks.jsonl) for the task format and [examples/initial-tools](examples/initial-tools) for MCP tool wrappers. Both tool and skill evolution are enabled by default.

Validate the configuration locally and start the run:

```bash
python run.py check-config
python run.py
```

To use your own configuration, run `python run.py --config configs/my-run.yaml run`. YAML file paths are relative to the configuration file's directory. Results are saved to `results/tasks.jsonl` and logs to `logs/` under the configured workspace (default: `workspace/default/`).

## Evaluation and Resume

In `train` mode, tasks run sequentially while tools and skills accumulate. The runner saves an initial snapshot and a snapshot after each completed task.

To evaluate a snapshot, copy your run configuration, set `mode: test`, and specify the test task file, `snapshot` path, and a **new workspace**:

```bash
python run.py --config configs/evaluate.yaml run
```

Test mode uses existing tools and skills without further evolution. Keep `skill_evolution` enabled to use existing skills during evaluation.

To resume an interrupted run, repeat the original command with the same configuration and workspace. Execution continues from the first incomplete task. Preserve the complete workspace for recovery, and use separate workspaces for different experiments.

## License

This project is licensed under the [MIT License](LICENSE).
