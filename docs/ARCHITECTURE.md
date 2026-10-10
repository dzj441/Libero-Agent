# Architecture

The public package has one purpose: run the frozen 30-task catalog through an
evaluator-owned LIBERO simulator. Research results, private tests, model
sessions, and generated demonstrations are kept outside the source release.

The installed runtime is organized by responsibility:

- `agent_env/runtime/` owns observations, OSC control, episode serving,
  recording, and public/private data contracts.
- `agent_env/evaluators/` owns task-specific success logic. It is not copied
  into an Agent workspace.
- `agent_env/icl/` projects optional demonstrations and experience context
  without exposing evaluator-private simulator state.
- `agent_env/integrations/` contains adapters for external task families.
  RoboMemArena compatibility inputs are pinned beneath its integration.
- `agent_env/harness/` contains harness-specific defaults, separate from task
  semantics and simulator control.
- `agent_env/launchers/` contains the Codex and external-harness single-episode
  supervisors used by the public batch entry point.

`agent_env/factory.py` is the single factory for ordinary LIBERO-compatible
tasks. RoboMemArena uses its integration factory because its frozen upstream
core must be activated before importing the simulator package. Both factories
enforce the installed 30-task manifest.

The public command is `libero-agent`, implemented by
`scripts/run_benchmark.py`. It supervises independent episodes through the
launchers and the evaluator-owned server, MCP adapter, and workspace-local
control client under `agent_env/`. Replay and viewer utilities live under
`tools/`; the installation smoke test lives in `scripts/smoke_install.py`.
These maintainer utilities are not installed as runtime commands.

`agent_env/release/manifest.json` is the single frozen source for task
identities, instructions, initialization states, budgets, and evaluation
conditions. Non-RoboMemArena BDDL and trusted initial states are packaged in
the `bddl_files/libero_agent/` and `init_files/libero_agent/` namespaces.
RoboMemArena compatibility inputs remain isolated beneath its pinned
integration.
