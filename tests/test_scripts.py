import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")


@pytest.mark.parametrize("name", ["colab_setup.sh", "config.env"])
def test_file_has_valid_bash_syntax(name: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_config_env_defines_the_dataset_settings() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "config.env"}"; echo "$SPIDER_DRIVE_ID|$SPIDER_DIR"'],
        capture_output=True,
        text=True,
        check=True,
    )

    drive_id, folder = result.stdout.strip().split("|")
    assert re.fullmatch(r"[\w-]+", drive_id)  # an ID, with no stray text
    assert re.fullmatch(r"[\w.-]+", folder)  # the inline comment does not leak into the value


def test_setup_script_requires_repo() -> None:
    env = {"PATH": os.environ["PATH"]}  # no REPO

    result = subprocess.run(
        ["bash", str(SCRIPTS / "colab_setup.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,  # failing is the expected outcome here
    )

    assert result.returncode != 0
    assert "REPO" in result.stdout + result.stderr
