"""Read resource availability without importing the training runtime in the API process."""

import csv
import io
import os
import shutil
import subprocess
import time

import psutil

_gpu_cache = (0, [])


def gpu_devices():
    global _gpu_cache
    if time.monotonic() - _gpu_cache[0] < 3:
        return _gpu_cache[1]
    executable = shutil.which("nvidia-smi")
    devices = []
    if executable:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,name,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode == 0:
                devices = [
                    dict(
                        index=int(row[0]),
                        name=row[1].strip(),
                        free_gb=float(row[2]) / 1024,
                        total_gb=float(row[3]) / 1024,
                    )
                    for row in csv.reader(io.StringIO(result.stdout))
                ]
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            devices = []
    _gpu_cache = (time.monotonic(), devices)
    return devices


def snapshot():
    memory = psutil.virtual_memory()
    return dict(
        cpu_percent=psutil.cpu_percent(),
        memory_percent=memory.percent,
        available_gb=memory.available / 1024**3,
        gpus=gpu_devices(),
    )
