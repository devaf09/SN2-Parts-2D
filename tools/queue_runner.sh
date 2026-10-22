#!/bin/bash
# GPU の実験の待ち行列を順に実行する（持ち主の許可でタスクスケジューラに登録、2026-10-22）。
# タスクスケジューラが 30 分ごと（とログオン時）に起動する。次のときは何もしないで終わる:
#   ・この実行役がすでに動いている（results/queue.lock の pid が生きている）
#   ・実験の python（experiments/ のプログラム）がすでに動いている（手で起動した実験と重ならないように）
# そうでなければ、results/queue.txt の先頭から、results/queue_done.txt にまだない行（sweep の .sh へのパス）を 1 つずつ実行する。
# 途中で PC やアプリが止まっても、次に起動したときに続きから回る（実行中だった行は done に書かれていないので、もう一度回る）。
cd "$(dirname "$0")/.."
LOCK=results/queue.lock
Q=results/queue.txt
DONE=results/queue_done.txt
LOG=results/queue_runner.log
touch "$Q" "$DONE"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then exit 0; fi
if powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'experiments' } | Select-Object -First 1" | grep -q .; then exit 0; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
while true; do
  next=$(grep -vxF -f "$DONE" "$Q" | grep -v '^\s*$' | grep -v '^#' | head -1)
  [ -z "$next" ] && break
  echo "$(date '+%F %T') start $next" >> "$LOG"
  bash "$next"
  echo "$(date '+%F %T') end   $next (exit $?)" >> "$LOG"
  echo "$next" >> "$DONE"
done
