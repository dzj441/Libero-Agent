# Libero-Agent

A 30-task benchmark for coding agents controlling a simulated Panda robot
through an evaluator-owned LIBERO environment. The first release contains
10 perception, 10 manipulation, and 10 long-horizon tasks. The broader task
catalog is reserved for a later release.

The agent receives observations and executes normalized seven-dimensional
OSC actions over MCP. The simulator owns task state, success checking,
action logging, and video recording. Object poses and checker progress are
not included in public observations.

## Install

The validated runtime is Linux, Python 3.10, MuJoCo 3.2.3, and robosuite 1.4.0.
A working EGL graphics stack is needed for headless GPU rendering.

```bash
git clone https://github.com/dzj441/Libero-Agent.git
cd Libero-Agent
conda create -n libero-agent python=3.10 -y
conda activate libero-agent
python -m pip install -r requirements-agent.txt
python -m pip install --no-deps -e .
```

Download the pinned VLABench assets separately. They are used by selected
perception, tube-insertion, and seesaw tasks and are not redistributed here.

```bash
hf download lerobot/vlabench-assets \
  --revision 08d7a4479a55c8fbb89ac0fd6fcf06e53d65b4a4 \
  --repo-type dataset --local-dir /path/to/vlabench-assets
export LIBERO_VLABENCH_ASSET_ROOT=/path/to/vlabench-assets
```

The download directory must contain `obj/`. Install `huggingface_hub` if the
`hf` command is unavailable. Review the asset provider's terms before use.

## Tasks and protocol

```bash
python scripts/run_benchmark.py --list-tasks
```

The [manifest](libero/libero/agent_env/release/manifest.json) freezes task
numbers, source suite IDs, instructions, initial-state IDs, and evaluator
types. Packaged BDDL and trusted initial states for non-RoboMemArena tasks are
grouped under the corresponding `libero_agent/` resource directories;
RoboMemArena compatibility inputs remain isolated in its pinned integration.
[The protocol](docs/PROTOCOL.md) specifies the reported evaluation settings.

| Condition | Tasks | Observation | Repetitions | Rollouts |
| --- | ---: | --- | ---: | ---: |
| Main table | 30 | Level 3 | 3 | 90 |
| Manipulation observation ablation | 10 | Levels 1 and 2 | 3 each | 60 |

Both conditions use seed 100 for every repetition, a 1,800-second episode
budget, 1,000,000 accepted action submissions, zero resets, and no ICL.
Each episode is a fresh agent process and environment. Reasoning effort is
`high` where the selected harness supports it; the model ID is always explicit.

## Launch

Install and authenticate the agent CLI separately. Native Codex launches use
the CLI's own authentication. Other harness integrations accept evaluator-local
API URL/key files; never put credentials in an agent workspace or a remote URL.

Check a schedule without launching agents or calling any model API:

```bash
python scripts/run_benchmark.py --dry-run --experiment main_table \
  --harness codex --model YOUR_VISION_MODEL --effort high \
  --gpus 0,1 --concurrency 4
```

Run the schedule by removing `--dry-run`. Use `--task-number 15 --waves 1` for
a single task. Use `--experiment manip_observation` for the 60-rollout ablation.

An external-harness example is:

```bash
python scripts/run_benchmark.py --experiment main_table \
  --harness claude_code --model YOUR_VISION_MODEL --effort high \
  --api-url-file /private/provider-url.txt \
  --api-key-file /private/provider-key.txt \
  --gpus 0,1 --concurrency 4 --batch-root agent_runs/main_table
```

The supervisor allocates two slots per GPU in this example. It writes
`supervisor_state.json`, per-task logs, and `wave1/2/3` result directories.
Workspaces are placed on the system temporary disk. There are no automatic
episode retries. `--resume` skips terminal records and refuses different
experimental settings. Infrastructure-invalid samples must be investigated
and explicitly rescheduled in a new batch.

Codex uses its filesystem isolation profile by default. The external-harness
integrations do not currently enforce that profile; treat their source-access
boundary as an explicit experimental limitation. CLI permission prompts are
not an equivalent filesystem sandbox.

## Local checks

```bash
python scripts/run_benchmark.py --list-tasks
python scripts/run_benchmark.py --dry-run --experiment main_table \
  --harness codex --model YOUR_VISION_MODEL --effort high
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/smoke_install.py \
  --task-number 1 --resolution 64 --gpu 0
```

The rendering smoke is optional and does not call a model API. Agent success
results, private sessions, and maintainer test suites are not part of the
source release.

## Acknowledgements

This project builds on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO),
with selected adaptations from [VLABench](https://github.com/OpenMOSS/VLABench),
[RoboMemArena](https://github.com/OpenHelix-Team/RoboMemArena),
[LIBERO-Mem](https://github.com/libero-mem/libero-mem),
[LiLo-VLA](https://github.com/YY-GX/LiLo-VLA),
[MetaWorld](https://github.com/Farama-Foundation/Metaworld),
[CLIER](https://github.com/michaal94/CLIER),
[CompoSuite](https://github.com/Lifelong-ML/CompoSuite),
[robosuite](https://github.com/ARISE-Initiative/robosuite), and
[RoboCasa](https://github.com/robocasa/robocasa) task patterns.
We thank the authors and contributors of these projects, especially the
RoboMemArena team for the task definitions and ordered-stage logic used by the
long-horizon subset.
