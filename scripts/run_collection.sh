#!/usr/bin/env bash
# Collect parking episodes with automatic CARLA server restarts.
# Restarts CARLA every BATCH episodes to avoid the streaming-ID exhaustion
# crash that occurs after ~250 episodes in a single server session.
#
# Usage:
#   bash scripts/run_collection.sh            # 1500 episodes, data/raw
#   bash scripts/run_collection.sh 500        # collect 500 total
#   bash scripts/run_collection.sh 1500 data/raw_A 42

set -euo pipefail

CARLA=~/projects/parking_data_gen/ParkingScenes/carla/CarlaUE4.sh
SCRIPT=scripts/generate_episodes.py
PORT=2000
MAP=Town04_Opt

TARGET=${1:-1500}      # total episodes to reach
SAVE=${2:-data/raw}    # save directory
SEED=${3:-42}          # starting seed (increments each batch)

BATCH=200              # restart CARLA after this many episodes
CARLA_WAIT=180         # max seconds to wait for CARLA to initialise

mkdir -p logs "$SAVE"

restart_carla() {
    echo "[WRAPPER] Killing CARLA..." | tee -a logs/collect.log
    pkill -f CarlaUE4 2>/dev/null || true
    sleep 8
    echo "[WRAPPER] Starting CARLA on port $PORT..." | tee -a logs/collect.log
    "$CARLA" -carla-rpc-port=$PORT -RenderOffScreen \
        >logs/carla_server.log 2>&1 &
    echo "[WRAPPER] Waiting for CARLA to accept connections (max ${CARLA_WAIT}s)..." | tee -a logs/collect.log
    elapsed=0
    while true; do
        if python3 -c "
import carla, sys
try:
    c = carla.Client('127.0.0.1', $PORT)
    c.set_timeout(3.0)
    c.get_server_version()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            echo "[WRAPPER] CARLA ready after ${elapsed}s." | tee -a logs/collect.log
            break
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
        echo "[WRAPPER]   ...still waiting (${elapsed}s)" | tee -a logs/collect.log
        if [ "$elapsed" -ge "$CARLA_WAIT" ]; then
            echo "[WRAPPER] ERROR: CARLA did not start after ${CARLA_WAIT}s — aborting." | tee -a logs/collect.log
            exit 1
        fi
    done
}

current_count() {
    ls -d "$SAVE"/episode_* 2>/dev/null | wc -l
}

restart_carla

while true; do
    have=$(current_count)
    remaining=$(( TARGET - have ))

    if [ "$remaining" -le 0 ]; then
        echo "[WRAPPER] Done. $have / $TARGET episodes collected." | tee -a logs/collect.log
        break
    fi

    batch=$(( remaining < BATCH ? remaining : BATCH ))
    echo "[WRAPPER] Have $have, need $TARGET — collecting next $batch (seed=$SEED)." \
        | tee -a logs/collect.log

    set +e
    python "$SCRIPT" \
        --host 127.0.0.1 --port "$PORT" \
        --map "$MAP" \
        --num_episodes "$batch" \
        --seed "$SEED" \
        --save_path "$SAVE" \
        2>&1 | tee -a logs/collect.log
    exit_code=${PIPESTATUS[0]}
    set -e

    SEED=$(( SEED + 7 ))   # shift seed so next batch has different NPC layout

    have_after=$(current_count)
    echo "[WRAPPER] Batch done (exit $exit_code). Episodes: $have → $have_after." \
        | tee -a logs/collect.log

    if [ "$have_after" -lt "$TARGET" ]; then
        restart_carla
    fi
done

# Kill CARLA when finished
pkill -f CarlaUE4 2>/dev/null || true
echo "[WRAPPER] CARLA stopped." | tee -a logs/collect.log
