#!/bin/bash
# 再開できる実行: run_missing.sh 結果ファイル コマンドの一覧ファイル
# 一覧の i 行目（0 から）は、結果ファイルの行数が i 以下のときだけ実行する（1 回の実行で結果が 1 行増える前提）。
cd "$(dirname "$0")/.."
OUT=$1; LIST=$2
export PYTHONIOENCODING=utf-8
i=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  n=$( [ -f "$OUT" ] && wc -l < "$OUT" || echo 0 )
  if [ "$n" -le "$i" ]; then eval "$cmd"; fi
  i=$((i + 1))
done < "$LIST"
