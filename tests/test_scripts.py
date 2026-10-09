import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent / "scripts"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")
needs_unzip = pytest.mark.skipif(shutil.which("unzip") is None, reason="unzip is not available")

FAKE_UVX = """#!/usr/bin/env bash
# stands in for `uvx gdown <id> -O <path>`
echo "$2" >> "$FAKE_LOG"
[ -z "${FAKE_FAIL:-}" ] || exit 1
cp "$FAKE_ZIPS/$2.zip" "$4"
"""


@pytest.mark.parametrize("name", ["colab_setup.sh", "download_data.sh", "config.env"])
def test_file_has_valid_bash_syntax(name: str) -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)


def test_config_env_defines_the_dataset_settings() -> None:
    names = ["SPIDER_DRIVE_ID", "SPIDER_DIR", "TEST_SUITE_DRIVE_ID", "TEST_SUITE_DIR"]
    echo = "|".join(f"${name}" for name in names)
    result = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "config.env"}"; echo "{echo}"'],
        capture_output=True,
        text=True,
        check=True,
    )

    spider_id, spider_dir, suite_id, suite_dir = result.stdout.strip().split("|")
    assert re.fullmatch(r"[\w-]+", spider_id) and re.fullmatch(r"[\w-]+", suite_id)  # IDs only
    assert re.fullmatch(r"[\w.-]+", spider_dir) and re.fullmatch(
        r"[\w.-]+", suite_dir
    )  # no comments


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


class Project:
    """A temporary project with the real download script, a fake `uvx` and fake archives."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.log = root / "uvx_calls.txt"
        (root / "scripts").mkdir()
        shutil.copy(SCRIPTS / "download_data.sh", root / "scripts")
        (root / "scripts" / "config.env").write_text(
            "SPIDER_DRIVE_ID=spider-id\nSPIDER_DIR=spider_data\n"
            "TEST_SUITE_DRIVE_ID=suite-id\nTEST_SUITE_DIR=database\n"
        )
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "uvx").write_text(FAKE_UVX)
        (bin_dir / "uvx").chmod(0o755)
        zips = root / "zips"
        zips.mkdir()
        self.make_zip(zips / "spider-id.zip", ["release/dev.json", "release/database/a/a.sqlite"])
        self.make_zip(zips / "suite-id.zip", ["suites/a/a.sqlite", "suites/a/av1.sqlite"])
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_LOG": str(self.log),
            "FAKE_ZIPS": str(zips),
        }

    @staticmethod
    def make_zip(path: Path, names: list[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, "x")
            archive.writestr("__MACOSX/junk", "x")  # like the real archives

    def run(self, *targets: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / "scripts" / "download_data.sh"), *targets],
            cwd=self.root.parent,  # the script must not depend on the working directory
            env={**self.env, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    @property
    def downloads(self) -> list[str]:
        return self.log.read_text().split() if self.log.exists() else []


@pytest.fixture
def project(tmp_path: Path) -> Project:
    root = tmp_path / "project"
    root.mkdir()
    return Project(root)


@needs_unzip
def test_download_requires_a_target(project: Project) -> None:
    result = project.run()

    assert result.returncode != 0
    assert "usage" in result.stderr
    assert project.downloads == []


@needs_unzip
def test_download_rejects_an_unknown_target_before_downloading_anything(project: Project) -> None:
    result = project.run("spider", "nope")

    assert result.returncode != 0
    assert "usage" in result.stderr
    assert project.downloads == []
    assert not (project.root / "data").exists()


@needs_unzip
def test_download_spider(project: Project) -> None:
    result = project.run("spider")

    assert result.returncode == 0, result.stderr
    data = project.root / "data"
    assert (data / "spider_data" / "dev.json").exists()
    assert (data / "spider_data" / "database" / "a" / "a.sqlite").exists()
    assert [p.name for p in data.iterdir()] == [
        "spider_data"
    ]  # no zip, no temporary directory, no __MACOSX
    assert project.downloads == ["spider-id"]


@needs_unzip
def test_download_test_suite(project: Project) -> None:
    result = project.run("test-suite")

    assert result.returncode == 0, result.stderr
    database = project.root / "data" / "database" / "a"
    assert sorted(p.name for p in database.iterdir()) == ["a.sqlite", "av1.sqlite"]
    assert project.downloads == ["suite-id"]


@needs_unzip
def test_download_both_targets(project: Project) -> None:
    result = project.run("spider", "test-suite")

    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in (project.root / "data").iterdir()) == [
        "database",
        "spider_data",
    ]
    assert project.downloads == ["spider-id", "suite-id"]


@needs_unzip
def test_download_skips_what_already_exists(project: Project) -> None:
    project.run("spider")

    result = project.run("spider")

    assert result.returncode == 0
    assert "already exists" in result.stdout
    assert project.downloads == ["spider-id"]  # downloaded once


@needs_unzip
def test_failed_download_leaves_nothing_behind(project: Project) -> None:
    result = project.run("spider", FAKE_FAIL="1")

    assert result.returncode != 0
    assert (
        list((project.root / "data").iterdir()) == []
    )  # no half-finished directory to be mistaken for data


@needs_unzip
@pytest.mark.parametrize("names", [["one/a.json", "two/b.json"], ["loose.txt"]])
def test_download_rejects_an_archive_without_a_single_folder(
    project: Project, names: list[str]
) -> None:
    project.make_zip(project.root / "zips" / "spider-id.zip", names)

    result = project.run("spider")

    assert result.returncode != 0
    assert list((project.root / "data").iterdir()) == []  # nothing is left behind


@needs_unzip
@pytest.mark.parametrize(
    ("target", "key"),
    [
        ("spider", "SPIDER_DRIVE_ID"),
        ("spider", "SPIDER_DIR"),
        ("test-suite", "TEST_SUITE_DRIVE_ID"),
        ("test-suite", "TEST_SUITE_DIR"),
    ],
)
def test_download_stops_when_a_config_key_is_missing(
    project: Project, target: str, key: str
) -> None:
    config = project.root / "scripts" / "config.env"
    kept = [line for line in config.read_text().splitlines() if not line.startswith(f"{key}=")]
    config.write_text("\n".join(kept) + "\n")

    result = project.run(target)

    assert result.returncode != 0
    assert key in result.stderr
    assert project.downloads == []  # nothing was downloaded
