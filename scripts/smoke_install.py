#!/usr/bin/env python3
"""Verify an installed checkout with real rendering and one native OSC action."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"libero/libero/agent_env/release/manifest.json"


def worker(task,profile,resolution,gpu):
    sys.path.insert(0,str(ROOT))
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    if task["suite"]=="robomemarena":
        bootstrap=ROOT/"libero/libero/agent_env/integrations/robomemarena/bootstrap.py"
        activate_robomemarena_core=runpy.run_path(str(bootstrap))["activate_robomemarena_core"]
        activate_robomemarena_core(source_root=ROOT)
        from libero.libero.agent_env.integrations.robomemarena.runtime import make_robomemarena_agent_env
        factory=make_robomemarena_agent_env
        extra={}
    else:
        from libero.libero.agent_env.factory import make_libero_agent_env
        factory=make_libero_agent_env
        extra={"suite":task["suite"]}
    environment=factory(**extra,task_id=task["task_id"],init_state_id=task["init_state_id"],
                        seed=100,profile=profile,camera_height=resolution,camera_width=resolution,
                        render_gpu_device_id=gpu,max_agent_steps=1_000_000)
    try:
        initial=environment.start_episode()["observation"]
        after=environment.step_osc_sequence([[0,0,0,0,0,0,0]])
        observation=after["observation"]
        cameras=observation["cameras"]
        assert all(camera["rgb"].shape==(resolution,resolution,3) for camera in cameras.values())
        serialized=json.dumps(observation,default=lambda _item:"<array>")
        names=environment.collector.task_entities.instance_names
        assert not any(name in serialized for name in names)
        assert "annotations" not in observation
        assert after["accepted_agent_step"]==1
        return {"number":task["number"],"profile":profile,"status":"passed",
                "rgb_shapes":{key:list(value["rgb"].shape) for key,value in cameras.items()},
                "native_osc_submissions":1,"private_name_leaks":0}
    finally:
        environment.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-number",type=int,action="append",default=[])
    parser.add_argument("--all-profiles",action="store_true")
    parser.add_argument("--profile",choices=("level1","level2","level3"),default="level3")
    parser.add_argument("--resolution",type=int,default=64)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/release_smoke")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    tasks=[task for task in manifest["tasks"] if not args.task_number or task["number"] in args.task_number]
    if args.worker:
        try:
            row=worker(tasks[0],args.profile,args.resolution,args.gpu)
        except Exception as exc:
            row={"number":tasks[0]["number"],"profile":args.profile,"status":"error","error_type":type(exc).__name__,"error":str(exc)}
        print("SMOKE_RESULT "+json.dumps(row),flush=True)
        return 0 if row["status"]=="passed" else 1
    args.output.mkdir(parents=True,exist_ok=True)
    jobs=[(task,profile) for task in tasks for profile in
          (manifest["evaluation_policy"]["observation_profiles_by_category"][task["category"]] if args.all_profiles else [args.profile])]
    def run(job):
        task,profile=job
        command=[sys.executable,str(Path(__file__).resolve()),"--worker","--task-number",str(task["number"]),
                 "--profile",profile,"--resolution",str(args.resolution),"--gpu",str(args.gpu)]
        completed=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
        log=args.output/f"t{task['number']:02d}_{profile}.log"
        log.write_text(completed.stdout)
        rows=[json.loads(line.removeprefix("SMOKE_RESULT ")) for line in completed.stdout.splitlines() if line.startswith("SMOKE_RESULT ")]
        row=rows[-1] if rows else {"number":task["number"],"profile":profile,"status":"error","error_type":"WorkerExited","returncode":completed.returncode}
        print(f"t{task['number']:02d} {profile}: {row['status']}",flush=True)
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows=list(pool.map(run,jobs))
    report={"schema_version":"libero.agent_30.release_smoke.v1","resolution":args.resolution,
            "sample_count":len(rows),"passed":sum(row["status"]=="passed" for row in rows),"results":rows}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if report["passed"]==len(rows) else 1


if __name__=="__main__":
    raise SystemExit(main())
