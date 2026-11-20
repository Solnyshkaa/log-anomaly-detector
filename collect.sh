#!/usr/bin/env bash
# Run on the monitored Ubuntu VM with an account that can read its journal.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: bash collect.sh SINCE UNTIL OUTPUT.jsonl" >&2
  exit 2
fi
umask 077
mkdir -p -- "$(dirname -- "$3")"
temporary=$(mktemp -- "$3.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
journalctl --since="$1" --until="$2" --output=json --no-pager > "$temporary"
if [[ ! -s "$temporary" ]]; then
  echo "No journal entries exported. Check the dates and journal permissions." >&2
  exit 1
fi
mv -f -- "$temporary" "$3"
