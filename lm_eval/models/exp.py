#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import csv
import signal
import sys
import shlex

# ========= 配置区（按需修改） =========
GPUS = [0, 1, 2]                # 并行使用的 GPU
MODEL = "steer_hf"
PRETRAINEDS = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # 你可以继续加
    "Qwen/Qwen2.5-7B-Instruct",
]

TASKS = "triviaqa_cot,gsm8k_cot_zeroshot" #"triviaqa_cot,truthfulqa_gen_cot,gsm8k_cot_zeroshot"
NUM_FEWSHOT = "0"
APPLY_CHAT_TEMPLATE = True
BATCH_SIZE = "auto"
LIMIT = "0.05"
STEER_SPAN = 48

STEER_LAYERS = [8, 12, 16, 20, 24]
# baseline=0.0 也在列表里
STEER_LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]

BASE_OUTDIR = Path("./eval_grid") / TASKS.replace(",", "_").replace(" ", "_")
RUNS_CSV = BASE_OUTDIR / "runs.csv"
LOG_PATH = BASE_OUTDIR / "runs.log"     # << 新增：全局文本日志
SKIP_IF_OUTDIR_EXISTS = True            # 目录下已有 metrics.json 则跳过该组合（断点续跑）

# ========= 实现 =========
def log_event(msg: str):
    """向全局 log 追加一行，自动带时间戳。"""
    ts = datetime.now().isoformat()
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
        
class Job:
    def __init__(self, pretrained: str, layer: int, lam: float):
        self.pretrained = pretrained
        self.layer = layer
        self.lam = float(lam)
        self.gpu = None

        # 短模型名：只取最后一段，并把特殊字符替换掉
        # meta-llama/Llama-3.1-8B-Instruct -> Llama-3.1-8B-Instruct
        model_tag = pretrained.split("/")[-1].replace(":", "_")

        tag = "BASELINE" if self.lam == 0.0 else f"lam{str(self.lam).replace('.', 'p')}"
        # 输出目录名里带上模型名，防止不同模型覆盖同一个 outdir
        safe_name = f"{model_tag}_L{layer}_{tag}"

        self.outdir = BASE_OUTDIR / safe_name
        self.stdout_log = self.outdir / "stdout.log"
        self.stderr_log = self.outdir / "stderr.log"
        self.returncode = None
        self.start_ts = None
        self.end_ts = None
        self.proc = None
        self._last_cmd_list = None

    def build_cmd(self, gpu_id: int):
        # --model_args 字符串里用自己的 pretrained
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
        return shlex.join(self._last_cmd_list) if self._last_cmd_list else ""


def ensure_dirs():
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

def write_runs_header_if_needed():
    if not RUNS_CSV.exists():
        with RUNS_CSV.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_start","timestamp_end","duration_sec",
                "pretrained", "layer","lambda","is_baseline","gpu","returncode",
                "output_path","stdout_log","stderr_log"
            ])


def append_run_row(job: Job):
    duration = None
    if job.start_ts and job.end_ts:
        duration = (job.end_ts - job.start_ts)
    with RUNS_CSV.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            job.start_ts.isoformat() if job.start_ts else "",
            job.end_ts.isoformat() if job.end_ts else "",
            f"{duration.total_seconds():.3f}" if duration else "",
            job.pretrained,
            job.layer, job.lam, int(job.lam == 0.0),
            job.gpu, job.returncode,
            str(job.outdir), str(job.stdout_log), str(job.stderr_log)
        ])


def launch_job_on_gpu(job: Job, gpu_id: int):
    job.gpu = gpu_id
    job.outdir.mkdir(parents=True, exist_ok=True)

    # 跳过逻辑：有 metrics.json 视为完成
    if SKIP_IF_OUTDIR_EXISTS and (job.outdir / "metrics.json").exists():
        job.start_ts = datetime.now()
        job.end_ts = datetime.now()
        job.returncode = 0
        append_run_row(job)
        log_event(f"SKIP  | GPU={gpu_id} L={job.layer} λ={job.lam} outdir={job.outdir} reason=metrics.json_exists")
        print(f"[SKIP] {job.outdir} 已存在 metrics.json，跳过。")
        return None

    cmd = job.build_cmd(gpu_id)
    cmd_str = job.cmd_as_str()
    log_event(f"START | GPU={gpu_id} L={job.layer} λ={job.lam} outdir={job.outdir} cmd={cmd_str}")
    print(f"[LAUNCH] GPU {gpu_id} -> {cmd_str}")

    stdout_f = job.stdout_log.open("w")
    stderr_f = job.stderr_log.open("w")
    env = os.environ.copy()
    # 可选：限制可见 GPU（同时我们也用 --device 精确绑定了目标卡）
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))
    job.start_ts = datetime.now()
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    job.proc = proc
    return proc

def main():
    ensure_dirs()
    write_runs_header_if_needed()
    log_event("SCHED | 初始化队列与日志")

    # 所有作业（含 baseline=0.0）
    queue = []
    for pretrained in PRETRAINEDS:
        for (L, lam) in itertools.product(STEER_LAYERS, STEER_LAMBDAS):
            queue.append(Job(pretrained, L, lam))

    running = {g: None for g in GPUS}

    def available_gpu():
        for g, v in running.items():
            if v is None:
                return g
        return None

    def handle_sigint(sig, frame):
        log_event("CTRL | 捕获到 SIGINT，尝试终止子进程")
        print("\n[CTRL-C] 捕获到中断信号，尝试终止子进程 ...")
        for g, pair in running.items():
            if pair is not None:
                job, proc = pair
                try:
                    proc.terminate()
                    log_event(f"TERM  | GPU={g} L={job.layer} λ={job.lam} pid={proc.pid}")
                except Exception as e:
                    log_event(f"TERM_FAIL | GPU={g} L={job.layer} λ={job.lam} err={e}")
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    while queue or any(running.values()):
        # 尝试在空闲 GPU 上启动新作业
        while queue:
            g = available_gpu()
            if g is None:
                break
            job = queue.pop(0)
            proc = launch_job_on_gpu(job, g)
            if proc is None:  # 被跳过
                continue
            running[g] = (job, proc)

        # 轮询已在跑的作业
        time.sleep(5)
        for g in list(running.keys()):
            pair = running[g]
            if pair is None:
                continue
            job, proc = pair
            ret = proc.poll()
            if ret is not None:
                job.returncode = ret
                job.end_ts = datetime.now()
                append_run_row(job)

                # 在日志里写结束标记
                duration_s = ""
                if job.start_ts and job.end_ts:
                    duration_s = f"{(job.end_ts - job.start_ts).total_seconds():.1f}s"
                log_event(f"END   | GPU={g} L={job.layer} λ={job.lam} rc={ret} dur={duration_s} outdir={job.outdir}")

                # 在各自日志文件里写一个结束标记
                try:
                    with job.stdout_log.open("a") as f:
                        f.write(f"\n\n[JOB END] returncode={ret} @ {job.end_ts}\n")
                    with job.stderr_log.open("a") as f:
                        f.write(f"\n\n[JOB END] returncode={ret} @ {job.end_ts}\n")
                except Exception:
                    pass

                print(f"[DONE] GPU {g} 完成：L={job.layer}, λ={job.lam} ({'BASELINE' if job.lam==0.0 else 'STEER'}), rc={ret}")
                running[g] = None

    log_event("SCHED | 全部任务完成")
    print("\n[ALL DONE] 全部任务完成。汇总见:", RUNS_CSV, " & ", LOG_PATH)

if __name__ == "__main__":
    main()
