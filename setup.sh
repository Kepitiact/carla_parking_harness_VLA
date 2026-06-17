#!/usr/bin/env bash
# One-shot environment + CARLA setup for parking_data_gen.
# Run from the parking_data_gen/ directory.
set -e

CARLA_VERSION="0.9.14"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Step 1: Python 3.8 check"
# CARLA's pip wheel only supports Python 3.7/3.8.
# Current Python is $(python3 --version 2>&1).
if ! command -v pyenv &>/dev/null; then
    echo "ERROR: pyenv not found. Install pyenv or activate a Python 3.8 environment manually."
    exit 1
fi

if ! pyenv versions | grep -q "3\.8\."; then
    echo "Installing Python 3.8.20 via pyenv (this may take a few minutes)..."
    pyenv install 3.8.20
fi

echo "Setting local Python to 3.8.20"
pyenv local 3.8.20

echo "==> Step 2: Python virtual environment"
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    python3 -m venv "$SCRIPT_DIR/venv"
    echo "Created venv at $SCRIPT_DIR/venv"
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/venv/bin/activate"

echo "==> Step 3: Install Python dependencies"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "==> Step 4: Download CARLA $CARLA_VERSION server"
CARLA_DIR="$SCRIPT_DIR/ParkingScenes/carla"
mkdir -p "$CARLA_DIR"

if [ ! -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    echo "Downloading CARLA_${CARLA_VERSION}.tar.gz (~10 GB) ..."
    wget -q --show-progress \
        "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_${CARLA_VERSION}.tar.gz" \
        -O "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz"
    echo "Extracting..."
    tar -xf "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz" -C "$CARLA_DIR"
    rm "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz"
    echo "CARLA server extracted."
else
    echo "CARLA server already present, skipping download."
fi

echo "==> Step 5: Download AdditionalMaps (Town04, Town10, etc.)"
IMPORT_DIR="$CARLA_DIR/Import"
mkdir -p "$IMPORT_DIR"
if [ ! -f "$IMPORT_DIR/AdditionalMaps_${CARLA_VERSION}.tar.gz" ]; then
    wget -q --show-progress \
        "https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_${CARLA_VERSION}.tar.gz" \
        -O "$IMPORT_DIR/AdditionalMaps_${CARLA_VERSION}.tar.gz"
    echo "Importing maps..."
    cd "$CARLA_DIR"
    chmod +x ImportAssets.sh
    ./ImportAssets.sh
    cd "$SCRIPT_DIR"
else
    echo "AdditionalMaps already present, skipping."
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To generate data:"
echo "    # Terminal 1 — start CARLA server:"
echo "    $CARLA_DIR/CarlaUE4.sh -RenderOffScreen"
echo ""
echo "    # Terminal 2 — activate venv and generate:"
echo "    source $SCRIPT_DIR/venv/bin/activate"
echo "    cd $SCRIPT_DIR"
echo "    python scripts/generate_episodes.py --map Town04_Opt --num_episodes 50"
echo "    python scripts/build_infos_pkl.py"
echo "    python scripts/build_cache_pkl.py"
echo "    python scripts/verify_dataset.py"
echo "═══════════════════════════════════════════════════════"
