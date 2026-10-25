#!/bin/bash
# GPU の実験の待ち行列を順に実行する（持ち主の許可でタスクスケジューラに登録、2026-10-22）。
# タスクスケジューラが 30 分ごと（とログオン時）に起動する。次のときは何もしないで終わる:
#   ・この実行役がすでに動いている（results/queue.lock の pid が生きている）
#   ・自分の GPU（GPU 0、2026-10-25 に GPU が 2 枚になってからの割り当て）に計算の python がいる、
#     またはこのリポジトリの sweep の bash が動いている（手で起動した実験と重ならないように）
# 実験は GPU 0 だけを見せて回す（プログラムの中は cuda のまま）。開始・終了は TmpFiles/GPU_QUEUE.md の履歴にも書く。
# そうでなければ、results/queue.txt の先頭から、results/queue_done.txt にまだない行（sweep の .sh へのパス）を 1 つずつ実行する。
# 途中で PC やアプリが止まっても、次に起動したときに続きから回る（実行中だった行は done に書かれていないので、もう一度回る）。
cd "$(dirname "$0")/.."
LOCK=results/queue.lock
Q=results/queue.txt
DONE=results/queue_done.txt
LOG=results/queue_runner.log
touch "$Q" "$DONE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then exit 0; fi
GPU=0
GQ=/d/AI_git/TmpFiles/GPU_QUEUE.md
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
# 自分の GPU に計算の python がいるか、手で起動した sweep の bash が動いていれば待つ（ほかのプロジェクトの GPU 1 の実験では待たない）
if nvidia-smi -i $GPU --query-compute-apps=pid,process_name --format=csv,noheader | grep -qi python; then exit 0; fi
if powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -match 'sweeps[/\\]sweep' } | Select-Object -First 1" | grep -q .; then exit 0; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
while true; do
  next=$(grep -vxF -f "$DONE" "$Q" | grep -v '^\s*$' | grep -v '^#' | head -1)
  [ -z "$next" ] && break
  echo "$(date '+%F %T') start $next" >> "$LOG"
  echo "- $(date '+%F %H:%M') GPU $GPU 開始 $next — SN2-Parts-2D" >> "$GQ"
  bash "$next"
  rc=$?
  echo "$(date '+%F %T') end   $next (exit $rc)" >> "$LOG"
  echo "- $(date '+%F %H:%M') GPU $GPU 終了 $next (exit $rc) — SN2-Parts-2D" >> "$GQ"
  echo "$next" >> "$DONE"
done
