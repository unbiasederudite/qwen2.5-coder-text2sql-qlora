#!/usr/bin/env bash
# Sets up a Colab runtime: clones the repo and installs it. Usage: REPO=owner/name [REF=main] colab_setup.sh
set -eu

REF=${REF:-main}  # branch or tag
REPO_DIR=${REPO##*/}  # directory git clone creates

# Clone the repo
cd /content
if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR already exists"
else
    git clone --branch "$REF" "https://github.com/$REPO.git"
fi
cd "$REPO_DIR"

# Install the project and the Colab dependencies
pip install uv
uv pip install --system . --group colab
