#!/usr/bin/env bash
# Downloads datasets into data/. Usage: download_data.sh spider|test-suite [spider|test-suite]
set -eu

usage() {
    echo "usage: $0 spider|test-suite [spider|test-suite]" >&2
    exit 2
}

# check the targets before downloading anything
[ $# -gt 0 ] || usage
for target in "$@"; do
    case $target in
        spider | test-suite) ;;
        *) usage ;;
    esac
done
command -v unzip > /dev/null || { echo "unzip not found" >&2; exit 1; }

ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/config.env"
cd "$ROOT"

trap 'rm -rf "${tmp:-}"' EXIT  # delete the temporary folder if the script stops early

# download <drive id> <name under data/>
download() {
    local dest=data/$2
    # skip what is already downloaded
    if [ -d "$dest" ]; then
        echo "$dest already exists"
        return
    fi
    # download and unzip into a temporary folder
    mkdir -p data
    tmp=$(mktemp -d data/.download.XXXXXX)
    uvx gdown "$1" -O "$tmp/archive.zip"
    unzip -q "$tmp/archive.zip" -x '__MACOSX/*' -d "$tmp"
    # move the folder into place only once it is complete
    mv "$tmp"/*/ "$dest"  # the archive holds a single folder
    rm -rf "$tmp"
}

# download the requested targets
for target in "$@"; do
    case $target in
        spider) download "$SPIDER_DRIVE_ID" "$SPIDER_DIR" ;;
        test-suite) download "$TEST_SUITE_DRIVE_ID" "$TEST_SUITE_DIR" ;;
    esac
done
