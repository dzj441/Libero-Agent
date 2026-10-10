#!/usr/bin/env python3
"""Supervise the frozen 30-task benchmark without automatic episode retries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "libero/libero/agent_env/release/manifest.json"
HARNESSES = ("codex", "claude_code", "kimi_code", "deepseek_harness", "qwen_code")


def read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def make_jobs(manifest, *, experiment, waves, seed, selected=()):
    jobs = []
    for wave in range(1, waves + 1):
        for task in manifest["tasks"]:
            if selected and task["number"] not in selected:
                continue
            if experiment == "manip_observation" and task["category"] != "manipulation":
                continue
            profiles = ("level1", "level2") if experiment == "manip_observation" else ("level3",)
            for profile in profiles:
                jobs.append({"job_id":f"w{wave}_t{task['number']:02d}_{profile}",
                             "wave":wave,"task":task,"profile":profile,"seed":seed})
    return jobs


def build_command(job, args, *, gpu, batch):
    task = job["task"]
    native = args.harness == "codex"
    module = (
        "libero.libero.agent_env.launchers.single_episode"
        if native
        else "libero.libero.agent_env.launchers.external_harness"
    )
    command = [sys.executable, "-u", "-m", module,
               "--suite",task["suite"],"--task-id",str(task["task_id"]),
               "--init-state-id",str(task["init_state_id"]),"--profile",job["profile"],
               "--seed",str(job["seed"]),"--resolution",str(args.resolution),
               "--render-gpu-device-id",str(gpu),"--initial-settle-control-steps","10",
               "--max-agent-steps",str(args.max_agent_steps),
               "--max-wall-time-seconds",str(args.max_wall_time_seconds),"--max-resets","0",
               "--icl","none","--run-id",job["job_id"],
               "--run-root",str(batch / f"wave{job['wave']}"),
               "--workspace-root",str(args.workspace_root)]
    if native:
        command += ["--codex-bin",args.codex_bin,"--codex-model",args.model,
                    "--codex-effort",args.effort,"--agent-isolation",args.agent_isolation,
                    "--https-proxy",args.proxy or ""]
    else:
        command += ["--harness",args.harness,"--model",args.model,
                    "--api-url-file",str(args.api_url_file),
                    "--api-key-file",str(args.api_key_file)]
        if args.proxy:
            command += ["--proxy",args.proxy]
        if args.harness == "claude_code":
            command += ["--claude-bin",args.claude_bin,"--claude-effort",args.effort]
        if args.harness in {"deepseek_harness", "kimi_code"} and args.effort != "high":
            raise ValueError("These integrations currently support only high reasoning")
        if args.harness == "qwen_code":
            command += ["--qwen-effort",args.effort]
    return command


def classify(run, *, budget):
    result = read_json(run / "result.json")
    harness = read_json(run / "harness_manifest.json")
    infrastructure = harness.get("infrastructure_error") or result.get("infrastructure_error")
    if result.get("status") == "finished" and not infrastructure:
        if result.get("success"):
            return "SUCCESS"
        if result.get("termination_reason") == "episode_wall_time_budget_exhausted":
            return "SIM_TIMEOUT"
        return "FAIL"
    if infrastructure and "wall-clock supervision timeout" in infrastructure:
        # Only elapsed episode evidence can turn an outer abort into a valid failure.
        elapsed = result.get("wall_time_seconds")
        if isinstance(elapsed, (int, float)) and elapsed >= budget:
            return "WATCHDOG_TIMEOUT"
    return "INVALID"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-tasks",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--experiment",choices=("main_table","manip_observation"),default="main_table")
    parser.add_argument("--waves",type=int,default=3)
    parser.add_argument("--seed",type=int,default=100)
    parser.add_argument("--task-number",type=int,action="append",default=[])
    parser.add_argument("--harness",choices=HARNESSES,default="codex")
    parser.add_argument("--model")
    parser.add_argument("--effort",choices=("low","medium","high"),default="high")
    parser.add_argument("--gpus",default="0")
    parser.add_argument("--concurrency",type=int,default=4)
    parser.add_argument("--batch-root",type=Path)
    parser.add_argument("--workspace-root",type=Path,default=Path("/tmp/libero-agent-workspaces"))
    parser.add_argument("--max-agent-steps",type=int,default=1_000_000)
    parser.add_argument("--max-wall-time-seconds",type=float,default=1800)
    parser.add_argument("--resolution",type=int,default=256)
    parser.add_argument("--api-url-file",type=Path,default=ROOT/".secrets/APIURL")
    parser.add_argument("--api-key-file",type=Path,default=ROOT/".secrets/APIKEY")
    parser.add_argument("--proxy")
    parser.add_argument("--codex-bin",default="codex")
    parser.add_argument("--claude-bin",default="claude")
    parser.add_argument("--agent-isolation",choices=("isolated","debug_full_access"),default="isolated")
    parser.add_argument("--resume",action="store_true",help="Skip terminal samples; do not retry invalid samples")
    args=parser.parse_args(argv)
    if not args.list_tasks and not args.model:
        parser.error("--model is required; provider model IDs must be specified explicitly")
    if args.waves < 1 or args.concurrency < 1 or args.resolution < 1:
        parser.error("waves, concurrency and resolution must be positive")
    if not 1 <= args.max_agent_steps <= 1_000_000:
        parser.error("max-agent-steps must be between 1 and 1000000")
    if not math.isfinite(args.max_wall_time_seconds) or args.max_wall_time_seconds <= 0:
        parser.error("max-wall-time-seconds must be finite and positive")
    if any(not 1 <= number <= 30 for number in args.task_number):
        parser.error("task-number must be in 1..30")
    try:
        args.gpu_ids=[int(part) for part in args.gpus.split(",")]
        if any(gpu < 0 for gpu in args.gpu_ids) or len(set(args.gpu_ids)) != len(args.gpu_ids):
            raise ValueError
    except ValueError:
        parser.error("gpus must be a comma-separated list of distinct nonnegative indices")
    return args


def main(argv=None):
    args=parse_args(argv)
    manifest=read_json(MANIFEST)
    if len(manifest.get("tasks",[])) != 30:
        raise RuntimeError("Installed release is missing its frozen 30-task manifest")
    if args.list_tasks:
        for task in manifest["tasks"]:
            print(f"{task['number']:02d} {task['category']:12s} {task['name']}")
        return 0
    jobs=make_jobs(manifest,experiment=args.experiment,waves=args.waves,
                   seed=args.seed,selected=args.task_number)
    batch=(args.batch_root or ROOT/"agent_runs"/datetime.now(timezone.utc).strftime("benchmark_%Y%m%dT%H%M%SZ")).resolve()
    if args.dry_run:
        print(json.dumps({"experiment":args.experiment,"model":args.model,"effort":args.effort,
                          "sample_count":len(jobs),"max_resets":0,"seed":args.seed,
                          "max_wall_time_seconds":args.max_wall_time_seconds,
                          "max_agent_steps":args.max_agent_steps,"gpus":args.gpu_ids,
                          "concurrency":args.concurrency,
                          "first_command":build_command(jobs[0],args,gpu=args.gpu_ids[0],batch=batch)},indent=2))
        return 0
    old=read_json(batch/"supervisor_state.json")
    if args.resume and old:
        expected={"experiment":args.experiment,"harness":args.harness,"model":args.model,
                  "effort":args.effort,"seed":args.seed,"max_resets":0,
                  "max_agent_steps":args.max_agent_steps,"max_wall_time_seconds":args.max_wall_time_seconds}
        if any(old.get(key) != value for key,value in expected.items()):
            raise ValueError("Resume settings differ from the original batch")
    if batch.exists() and not args.resume:
        raise FileExistsError("Batch already exists; choose a new batch root or use --resume")
    batch.mkdir(parents=True,exist_ok=True)
    state={"schema_version":"libero.agent_30.supervisor.v1","created_at":utc_now(),
           "status":"running","experiment":args.experiment,"harness":args.harness,
           "model":args.model,"effort":args.effort,"seed":args.seed,"max_resets":0,
           "max_wall_time_seconds":args.max_wall_time_seconds,"max_agent_steps":args.max_agent_steps,
           "concurrency":args.concurrency,"gpus":args.gpu_ids,"max_outer_retries":0,
           "sample_count":len(jobs),"jobs":old.get("jobs",{})}
    lock=threading.Lock()
    stop=threading.Event()
    active={}
    slots=queue.Queue()
    for index in range(args.concurrency):
        slots.put(args.gpu_ids[index % len(args.gpu_ids)])

    def save():
        state["updated_at"]=utc_now()
        temp=batch/".supervisor_state.json.tmp"
        temp.write_text(json.dumps(state,indent=2)+"\n")
        temp.replace(batch/"supervisor_state.json")

    def stop_signal(_sig,_frame):
        stop.set()
        with lock:
            for process in list(active.values()):
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)

    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)

    def run(job):
        identity=job["job_id"]
        with lock:
            if identity in state["jobs"] and state["jobs"][identity].get("status") in {"complete","invalid","interrupted"}:
                return
        if stop.is_set():
            return
        gpu=slots.get()
        try:
            if stop.is_set():
                return
            run_root=batch/f"wave{job['wave']}"/identity
            if run_root.exists():
                raise FileExistsError(f"Refusing to overwrite sample {identity}")
            command=build_command(job,args,gpu=gpu,batch=batch)
            log=batch/"supervisor_logs"/f"{identity}.log"
            log.parent.mkdir(exist_ok=True)
            with lock:
                state["jobs"][identity]={"status":"launching","gpu":gpu,"task_number":job["task"]["number"],
                                          "wave":job["wave"],"profile":job["profile"],"started_at":utc_now(),
                                          "run_directory":str(run_root)}
                save()
            environment=os.environ.copy()
            environment["PYTHONPATH"]=os.pathsep.join(filter(None,(str(ROOT),environment.get("PYTHONPATH"))))
            with log.open("w") as output:
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=output,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                with lock:
                    active[identity]=process
                    state["jobs"][identity].update(status="running",pid=process.pid)
                    save()
                try:
                    code=process.wait(timeout=args.max_wall_time_seconds+360)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        code=process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        code=process.wait()
                finally:
                    with lock:
                        active.pop(identity,None)
            outcome="INTERRUPTED" if stop.is_set() else classify(run_root,budget=args.max_wall_time_seconds)
            with lock:
                state["jobs"][identity].update(status="interrupted" if stop.is_set() else "invalid" if outcome=="INVALID" else "complete",
                                              outcome=outcome,returncode=code,finished_at=utc_now())
                save()
            print(f"{identity}: {outcome}",flush=True)
        except Exception as exc:
            # Never store arbitrary exception bodies, which can contain reflected credentials.
            with lock:
                state["jobs"].setdefault(identity,{})
                state["jobs"][identity].update(status="invalid",outcome="INVALID",error_type=type(exc).__name__)
                save()
        finally:
            slots.put(gpu)

    with lock:
        save()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures=[executor.submit(run,job) for job in jobs]
        for future in as_completed(futures):
            future.result()
    with lock:
        state["status"]="stopped" if stop.is_set() else "finished"
        state["finished_at"]=utc_now()
        save()
    invalid=any(job.get("status")!="complete" for job in state["jobs"].values())
    return 2 if stop.is_set() or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
