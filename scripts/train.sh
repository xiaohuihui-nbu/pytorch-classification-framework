#!/usr/bin/env bash
# PowerShell: & "D:\software\Git\bin\bash.exe" scripts/train.sh
# Git Bash: bash scripts/train.sh [start|status|stop]
# 8 个模型同时运行；只修改下面参数和文件末尾的模型命令即可。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/train.sh"
cd -- "$ROOT"
EPOCHS=${EPOCHS:-10}
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
LOCK=logs/.train.lock

help() {
    cat <<'HELP'
bash scripts/train.sh          后台并发运行 8 个模型
bash scripts/train.sh status   查看所有批次和逐模型状态
bash scripts/train.sh stop     立即停止本脚本启动的活动批次及其训练子进程
EPOCHS=20 bash scripts/train.sh 修改统一轮数（也可编辑各模型命令）
日志：logs/train_<时间>_<唯一编号>/train_<模型>.log
保留自动接续上次权重的配置；各模型命令加 --fresh 可重新开始。
HELP
}

# 原子更新状态，查询命令不会读到只写了一半的结果。
write_status() { printf '%s\n' "$2" > "$1.tmp"; mv -f -- "$1.tmp" "$1"; }
active_batches() {
    local path state
    for path in logs/train_*/status.txt; do
        [[ -f $path ]] || continue
        state=$(cat -- "$path")
        case "$state" in STARTING|RUNNING|STOPPING) printf '%s\n' "${path%/status.txt}" ;; esac
    done
}
show_status() {
    local path
    for path in logs/train_*/status.txt; do
        [[ -f $path ]] || continue
        printf '\n%s: %s\n' "${path%/status.txt}" "$(cat -- "$path")"
        [[ ! -f ${path%/status.txt}/summary.csv ]] || cat -- "${path%/status.txt}/summary.csv"
    done
}
stop_batches() {
    local dir pending attempt
    local -a targets=()
    while IFS= read -r dir; do targets+=("$dir"); done < <(active_batches)
    if (( ${#targets[@]} == 0 )); then echo '没有活动训练批次。'; return; fi
    for dir in "${targets[@]}"; do touch -- "$dir/STOP"; done
    # 由后台监督进程终止自己持有的子进程，不依据历史 PID 随意杀进程。
    for ((attempt=0; attempt<30; attempt++)); do
        pending=0
        for dir in "${targets[@]}"; do
            case "$(cat -- "$dir/status.txt")" in STARTING|RUNNING|STOPPING) pending=1 ;; esac
        done
        if (( ! pending )); then show_status; return; fi
        sleep 1
    done
    echo '停止请求已提交，但尚未确认全部退出；请检查 batch.log 和 status。' >&2
    return 1
}

mode=${1:-start}
case "$mode" in
    -h|--help|help) help; exit 0 ;;
    status) [[ $# == 1 ]] || exit 2; show_status; exit 0 ;;
    stop) [[ $# == 1 ]] || exit 2; stop_batches; exit 0 ;;
    start)
        [[ $# -le 1 ]] || { help; exit 2; }
        [[ $EPOCHS =~ ^[1-9][0-9]*$ ]] || { echo 'EPOCHS 必须是正整数' >&2; exit 2; }
        command -v uv >/dev/null || { echo '未找到 uv' >&2; exit 1; }
        mkdir -p logs
        if ! mkdir -- "$LOCK" 2>/dev/null; then
            echo '已有批次启动锁，请先运行 status / stop；异常退出后的锁见文档处理。' >&2
            exit 1
        fi
        trap 'rmdir -- "$LOCK" 2>/dev/null || true' EXIT
        batch=$(mktemp -d "logs/train_$(date +%Y%m%d_%H%M%S)_XXXXXX")
        printf '%s\n' "$batch" > "$LOCK/batch"
        printf 'STARTING\n' > "$batch/status.txt"
        nohup bash "$SCRIPT" --worker "$batch" > "$batch/batch.log" 2>&1 < /dev/null &
        printf '%s\n' "$!" > "$batch/batch.pid"
        trap - EXIT
        printf '后台任务已提交（8 模型并发）。日志：%s/%s\n' "$ROOT" "$batch"
        echo '查看：bash scripts/train.sh status；停止：bash scripts/train.sh stop'
        exit 0 ;;
    --worker)
        [[ $# == 2 && -f $2/status.txt ]] || exit 2
        batch=$2 ;;
    *) help; exit 2 ;;
esac

models=()
pids=()
state=FAILED
# 每个后台任务使用独立进程组，Unix 可整体停止；Git Bash 使用 Windows 进程树。
set -m
rebuild_summary() {
    local model status code
    printf 'model,status,exit_code,log\n' > "$batch/summary.csv.tmp"
    for model in "${models[@]}"; do
        status=$(cat -- "$batch/$model.status")
        code=''
        [[ ! -f $batch/$model.exit ]] || code=$(cat -- "$batch/$model.exit")
        printf '%s,%s,%s,%s\n' "$model" "$status" "$code" "$batch/train_$model.log" >> "$batch/summary.csv.tmp"
    done
    mv -f -- "$batch/summary.csv.tmp" "$batch/summary.csv"
}
terminate_children() {
    local child native
    case "$OSTYPE" in
        msys*|cygwin*)
            # MSYS exec 可能改变 Windows 父子关系。先用本批次唯一日志参数找出
            # 原生 uv/Python 根进程，再终止各自进程树，避免仅杀 Bash 而遗留训练。
            CLS_BATCH_MARKER="logging.directory=../../$batch/" powershell.exe -NoProfile -Command '
                $ErrorActionPreference = "Stop"
                $targets = @(Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -in @("uv.exe", "python.exe", "pythonw.exe") -and
                    $_.CommandLine -and $_.CommandLine.Contains($env:CLS_BATCH_MARKER)
                })
                foreach ($target in $targets) {
                    $current = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $target.ProcessId)
                    if ($current -and $current.CreationDate -eq $target.CreationDate) {
                        & taskkill.exe /PID $target.ProcessId /T /F
                    }
                }
                $remaining = @(Get-CimInstance Win32_Process | Where-Object {
                    $_.Name -in @("uv.exe", "python.exe", "pythonw.exe") -and
                    $_.CommandLine -and $_.CommandLine.Contains($env:CLS_BATCH_MARKER)
                })
                if ($remaining.Count) { exit 1 }
                exit 0
            ' || return 1 ;;
    esac
    # jobs -pr 仅返回本监督进程尚在运行的子任务。
    for child in $(jobs -pr); do
        case "$OSTYPE" in
            msys*|cygwin*)
                native=$(ps -p "$child" | awk 'NR==2 {print $4}')
                if [[ -n $native ]]; then
                    MSYS_NO_PATHCONV=1 taskkill.exe /PID "$native" /T /F || {
                        [[ -z $(ps -p "$child" | awk 'NR==2 {print $4}') ]] || return 1
                    }
                fi ;;
            *) kill -KILL -- "-$child" 2>/dev/null || true ;;
        esac
    done
}
finish() {
    local code=$? model
    trap - EXIT INT TERM
    if [[ -n $(jobs -pr) ]]; then
        if ! terminate_children; then
            echo '未能终止所有训练子进程，请检查本批次 PID 和 batch.log。' >&2
            write_status "$batch/status.txt" STOPPING
            return 1
        fi
    fi
    wait || true
    for model in "${models[@]}"; do
        if [[ $(cat -- "$batch/$model.status") == RUNNING ]]; then
            write_status "$batch/$model.status" "$state"
            printf '%s\n' "$code" > "$batch/$model.exit"
            printf '\n[%s] %s，退出码=%s\n' "$(date '+%F %T')" "$state" "$code" >> "$batch/train_$model.log"
        fi
    done
    rebuild_summary
    printf '%s\n' "$code" > "$batch/exit_code.txt"
    write_status "$batch/status.txt" "$state"
    printf '[%s] 批次结束：%s，退出码=%s\n' "$(date '+%F %T')" "$state" "$code"
    # 仅移除本批次自己创建的锁。
    if [[ -f $LOCK/batch && $(cat -- "$LOCK/batch") == "$batch" ]]; then
        rm -- "$LOCK/batch"
        rmdir -- "$LOCK"
    fi
}
trap finish EXIT
trap 'state=STOPPED; exit 130' INT TERM

# 所有子模型均使用 --no-sync，避免并发安装/卸载同一虚拟环境。
echo "[$(date '+%F %T')] 同步项目环境"
uv sync --frozen --extra export &
setup_pid=$!
while kill -0 "$setup_pid" 2>/dev/null; do
    if [[ -f $batch/STOP ]]; then state=STOPPED; exit 130; fi
    sleep 0.2
done
wait "$setup_pid"
write_status "$batch/status.txt" RUNNING

run_model() {
    local model=$1
    shift
    [[ ! -f $batch/STOP ]] || { state=STOPPED; exit 130; }
    [[ -f configs/flower/flower_$model.yaml ]] || { echo "配置不存在：$model" >&2; return 1; }
    models+=("$model")
    write_status "$batch/$model.status" RUNNING
    local -a command=("$@" --workers 0 --set trainer.enable_progress_bar=false
        --set "logging.directory=../../$batch/$model")
    if [[ $EPOCHS == 1 ]]; then command+=(--warmup-epochs 0); fi
    printf '[%s] 启动 %s\n' "$(date '+%F %T')" "$model"
    (
        trap - EXIT INT TERM
        printf '开始时间：%s\n命令：' "$(date '+%F %T')"
        printf '%q ' "${command[@]}"; printf '\n'
        if "${command[@]}"; then code=0; result=SUCCEEDED; else code=$?; result=FAILED; fi
        printf '%s\n' "$code" > "$batch/$model.exit"
        write_status "$batch/$model.status" "$result"
        printf '[%s] %s：%s，退出码=%s\n' "$(date '+%F %T')" "$model" "$result" "$code"
        exit "$code"
    ) > "$batch/train_$model.log" 2>&1 &
    pids+=("$!")
    printf '%s\n' "$!" > "$batch/$model.pid"
    rebuild_summary
}

# 所有模型同时启动：注释整行可跳过模型；单独修改该行的 epochs/batch/lr。
run_model resnet18 uv run --no-sync python examples/train.py --config configs/flower/flower_resnet18.yaml --epochs "$EPOCHS"
run_model resnet50 uv run --no-sync python examples/train.py --config configs/flower/flower_resnet50.yaml --epochs "$EPOCHS"
run_model mobilenetv3_small uv run --no-sync python examples/train.py --config configs/flower/flower_mobilenetv3_small.yaml --epochs "$EPOCHS"
run_model efficientnet_b0 uv run --no-sync python examples/train.py --config configs/flower/flower_efficientnet_b0.yaml --epochs "$EPOCHS"
run_model convnext_tiny uv run --no-sync python examples/train.py --config configs/flower/flower_convnext_tiny.yaml --epochs "$EPOCHS"
run_model vit_tiny uv run --no-sync python examples/train.py --config configs/flower/flower_vit_tiny.yaml --epochs "$EPOCHS"
run_model deit_tiny uv run --no-sync python examples/train.py --config configs/flower/flower_deit_tiny.yaml --epochs "$EPOCHS"
run_model swin_tiny uv run --no-sync python examples/train.py --config configs/flower/flower_swin_tiny.yaml --epochs "$EPOCHS"

while [[ -n $(jobs -pr) ]]; do
    if [[ -f $batch/STOP ]]; then
        write_status "$batch/status.txt" STOPPING
        state=STOPPED
        exit 130
    fi
    rebuild_summary
    sleep 0.5
done
state=SUCCEEDED
for child in "${pids[@]}"; do
    if ! wait "$child"; then state=FAILED; fi
done
if [[ $state == FAILED ]]; then exit 1; fi
