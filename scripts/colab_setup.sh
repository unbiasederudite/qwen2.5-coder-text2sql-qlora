#!/usr/bin/env bash
set -e
exec 2>&1

: "${REPO:?set REPO=owner/name}"  # GitHub repository to clone
REF=${REF:-main}  # branch or tag
REPO_DIR=${REPO##*/}

# clone the repo
cd /content
if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR already exists"
else
    git clone --branch "$REF" https://github.com/$REPO.git
fi
cd "$REPO_DIR"

# dataset settings
source scripts/config.env

# install the project and the train dependencies
pip install uv
uv pip install --system . --group train

# download the Spider dataset
if [ -d "data/$SPIDER_DIR" ]; then
    echo "data/$SPIDER_DIR already exists"
else
    uvx gdown "$SPIDER_DRIVE_ID" -O "$SPIDER_DIR.zip"
    unzip -q "$SPIDER_DIR.zip" "$SPIDER_DIR/*" -d data
    rm "$SPIDER_DIR.zip"
fi
