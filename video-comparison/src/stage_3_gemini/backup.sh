#!/bin/bash

SRC="completed_stage_3.json"
DEST_DIR="outputs"
mkdir -p "$DEST_DIR"

while true; do
  if [[ -s "$SRC" ]]; then
    # file exists and size > 0
    TIMESTAMP=$(date +%s)
    cp "$SRC" "$DEST_DIR/completed_stage_3_backup_${TIMESTAMP}.json"
    echo "Backup done for $TIMESTAMP (file size > 0)"
    sleep 300  # wait 5 minutes
  else
    # file is zero size (or doesn't exist)
    echo "File empty; will retry in 5 seconds"
    sleep 5
    continue
  fi
done
