#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import signal
import sys
import shlex
import json

# ========= 配置 =========
GPUS = [0, 1, 2]
MODEL = "steer_hf"
PRETRAINEDS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

TASKS = "triviaqa_cot,gsm8k_cot_zeroshot"
NUM_FEWSHOT = "0"
APPLY_CHAT_TEMPLATE = True
BATCH_SIZE = "auto"
LIMIT = "0.05"
STEER_SPAN = 48

STEER_LAYERS = [8, 12, 16, 20, 24]
STEER_LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]

BASE_OUTDIR = Path("./eval_grid") / TASKS.replace(",", "_").replace(" ", "_")
RUNS_JSON = BASE_OUTDIR / "runs.json"

RUNS_STATE = {}  # job_id -> 记录


# ========= JSON 读写 =========
def ensure_dirs():
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)


def load_runs_state():
    if not RUNS_JSON.exists():
        return {}
    try:
        with RUNS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("jobs", {})
    except:
        return {}


def save_runs_state():
    ensure_dirs()
    payload = {
        "updated_at": datetime.now().isoformat(),
        "jobs": RUNS_STATE,
    }
    with RUNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ========= Job 定义 =========
class Job:
    def __init__(self, pretrained: str, layer: int, lam: float):
        self.pretrained = pretrained
        self.layer = layer
        self.lam = float(lam)
        self.gpu = None
        self.returncode = None
        self.start_ts = None
        self.end_ts = None
        self.proc = None
        self.status = "pending"

        model_tag = pretrained.replace("/", "_")
        self.job_id = f"{model_tag}__L{layer}__lam{self.lam}"

        tag = "BASELINE" if self.lam == 0.0 else f"lam{str(self.lam).replace('.', 'p')}"
        safe_name = f"{pretrained.split('/')[-1]}_L{layer}_{tag}"
        self.outdir = BASE_OUTDIR / safe_name
        self.stdout_log = self.outdir / "stdout.log"
        self.stderr_log = self.outdir / "stderr.log"
        self._last_cmd_list = None

    def build_cmd(self, gpu_id: int):
        json_path = f"/home/youyang7/projects/fact-enhancement/artifacts/{self.pretrained.replace('/', '_')}_factual_dirs.json"

        model_args = (
            f"pretrained={self.pretrained},"
            f"steer_layer={self.layer},"
            f"steer_lambda={self.lam},"
            f"steer_span={STEER_SPAN},"
            f"steer_json_path={json_path}"
        )

        cmd = [
            "lm_eval",
            "--model", MODEL,
            "--model_args", model_args,
            "--tasks", TASKS,
            "--num_fewshot", NUM_FEWSHOT,
            "--device", f"cuda:{gpu_id}",
            "--batch_size", BATCH_SIZE,
            "--limit", LIMIT,
            "--output_path", str(self.outdir),
            "--log_samples",
        ]
        if APPLY_CHAT_TEMPLATE:
            cmd.append("--apply_chat_template")

        self._last_cmd_list = cmd
        return cmd

    def cmd_as_str(self):
        return shlex.join(self._last_cmd_list or [])

    def to_record(self):
        duration = None
        if self.start_ts and self.end_ts:
            duration = (self.end_ts - self.start_ts).total_seconds()

        return {
            "job_id": self.job_id,
            "pretrained": self.pretrained,
            "layer": self.layer,
            "lambda": self.lam,
            "is_baseline": self.lam == 0.0,
            "gpu": self.gpu,
            "status": self.status,
            "returncode": self.returncode,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "duration_sec": duration,
            "outdir": str(self.outdir),
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "cmd": self.cmd_as_str(),
        }


# ========= 运行逻辑 =========
def launch_job(job: Job, gpu_id: int):
    global RUNS_STATE

    job.gpu = gpu_id
    job.outdir.mkdir(parents=True, exist_ok=True)

    cmd = job.build_cmd(gpu_id)
    cmd_str = job.cmd_as_str()
    print(f"[LAUNCH] GPU {gpu_id} -> {cmd_str}")

    job.start_ts = datetime.now()
    job.status = "running"
    RUNS_STATE[job.job_id] = job.to_record()
    save_runs_state()

    stdout_f = job.stdout_log.open("w")
    stderr_f = job.stderr_log.open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))

    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    job.proc = proc
    return proc


def main():
    global RUNS_STATE
    ensure_dirs()

    RUNS_STATE = load_runs_state()
    if RUNS_STATE:
        print(f"[INFO] 发现 runs.json，启用断点续跑")

    # === 生成队列 ===
    queue = []
    for pretrained in PRETRAINEDS:
        for (L, lam) in itertools.product(STEER_LAYERS, STEER_LAMBDAS):
            job = Job(pretrained, L, lam)
            state = RUNS_STATE.get(job.job_id)

            # 只依赖 runs.json 判断是否跳过
            if state and state.get("status") == "done" and state.get("returncode") == 0:
                print(f"[SKIP] {job.job_id} 已完成，跳过")
                continue

            queue.append(job)

    running = {g: None for g in GPUS}

    def available_gpu():
        for g, v in running.items():
            if v is None:
                return g
        return None

    def handle_sigint(sig, frame):
        print("\n[CTRL-C] 中断，关闭所有子进程...")
        for g, pair in running.items():
            if pair is not None:
                job, proc = pair
                proc.terminate()
                job.status = "failed"
                job.end_ts = datetime.now()
                RUNS_STATE[job.job_id] = job.to_record()
        save_runs_state()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    # === 主循环 ===
    while queue or any(running.values()):
        # 启动新 job
        while queue:
            g = available_gpu()
            if g is None:
                break
            job = queue.pop(0)
            proc = launch_job(job, g)
            running[g] = (job, proc)

        # 轮询
        time.sleep(5)
        for g, pair in running.items():
            if pair is None:
                continue
            job, proc = pair
            ret = proc.poll()
            if ret is not None:
                job.returncode = ret
                job.end_ts = datetime.now()
                job.status = "done" if ret == 0 else "failed"

                RUNS_STATE[job.job_id] = job.to_record()
                save_runs_state()

                print(f"[DONE] GPU {g}, L={job.layer}, λ={job.lam}, rc={ret}, status={job.status}")
                running[g] = None

    print("\n[ALL DONE] 全部任务完成，结果见 runs.json")


if __name__ == "__main__":
    main()
