# Evaluation protocol

The first public catalog contains exactly 30 tasks: 10 perception, 10
manipulation, and 10 long-horizon tasks. Source suite IDs are retained so the
task definitions and checker implementations can be compared with the
development runtime. The release factories reject tasks outside the catalog.
Backend registration tables are not an additional public evaluation set.

## Reported conditions

| Parameter | Main table | Manipulation observation ablation |
| --- | --- | --- |
| Task numbers | 1–30 | 11–20 |
| Observation | Level 3 | Levels 1 and 2 |
| Repetitions per task/condition | 3 | 3 |
| Seed | 100 in every repetition | 100 in every repetition |
| Initialization | Manifest `init_state_id` | Same |
| Wall-clock budget | 1800 s | 1800 s |
| Accepted action submissions | 1,000,000 | 1,000,000 |
| Maximum micro-actions per submission | 50 | 50 |
| Resets | 0 | 0 |
| Outer episode retries | 0 | 0 |
| ICL | None | None |
| Render resolution | 256 × 256 per camera | Same |
| Reasoning | High when supported by the harness | Same |

One submission is one call to `osc_sequence`, not one physics step. The
1,000,000-submission limit acts as a practically unrestricted action budget;
the wall-clock budget remains binding. The simulator's episode clock starts
at `start_episode`. Model/harness setup and simulator readiness have separate
bounded supervision.

Each task/repetition launches a separate agent process, workspace, simulator,
and session. GPU count and concurrency are scheduling parameters and must be
recorded in each batch; they do not multiply the task catalog.

## Observation levels

- Level 1: head/wrist RGB and basic robot state.
- Level 2: Level 1 plus dynamic proprioception.
- Level 3: Level 2 plus metric depth and camera intrinsics/extrinsics.

No level publishes object poses, instance IDs, masks, bounding boxes, private
mass labels, goal-checker state, or stage progress. Goal images are task
specifications rather than expert trajectories.

## Outcomes

`SUCCESS` means the authoritative checker accepted the task. `FAIL` means an
ordinary completed episode reported failure. `SIM_TIMEOUT` means the simulator
exhausted the full episode wall-clock budget. `WATCHDOG_TIMEOUT` is counted as
a model failure only when episode timing evidence confirms that the full
budget elapsed. API errors, simulator launch failures, and early harness exits
are `INVALID`. Infrastructure recovery must retain the rejected attempt and
explicitly document its replacement; it is not an agent reset opportunity.

The ordinary success rate uses successes divided by valid episodes, with
simulation and full-budget watchdog timeouts included as failures. The
all-three-repetitions point rule for simple tasks is separate from the ordinary
success-rate definition.

## Goal semantics

Cooking text/image tasks share the same five unordered terminal predicates.
The image variant receives the goal reference image, and the text variant
receives a state description. Neither imposes an uncommunicated action order.
The strict-order cooking and coffee-machine prototypes are outside this release.
The bowl-rotation, pouring, wipe, seesaw, and RoboMemArena evaluators retain
their source-specific event/ordering conditions. Packaging does not change them.

## Isolation and reproducibility

Native Codex launches select filesystem isolation. External harness launchers
currently provide credentials, MCP configuration, process supervision, and
usage accounting, but do not guarantee equivalent filesystem isolation. Public
network resources remain reachable even when local source access is restricted.
Record the harness version, model channel, effort, timing, action count, and
token usage from run artifacts; reasoning effort labels are not an equal-token
budget across different providers.
