"""Release contracts for bilingual documentation and packaged resources."""

from pathlib import Path
from urllib.parse import unquote, urlsplit
import csv
import re
import statistics
import tomllib
import xml.etree.ElementTree as ET
from unified_va.config import PROMPT, REVISION

ROOT = Path(__file__).resolve().parents[1]


def test_documentation_links_and_machine_independent_paths():
    documents = list(ROOT.glob("*.md")) + list((ROOT / "docs").glob("*.md"))
    documents += [ROOT / name / "README.md" for name in ("configs", "results", "data")]
    for path in documents:
        text = path.read_text(encoding="utf-8")
        prose = re.sub(r"```.*?```", "", text, flags=re.S)
        links = re.findall(r"\]\(([^)]+)\)", prose)
        links += re.findall(r'(?:href|src)="([^"]+)"', prose)
        for link in links:
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve()
            assert destination.is_relative_to(ROOT), (path, link)
            assert destination.exists(), (path, link)
    # Match concrete user/host paths, without treating documented placeholders
    # or paths created dynamically by the application as machine dependencies.
    host_path = re.compile(
        r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/]|/(?:home|root|Users|mnt)/)"
    )
    suffixes = {".py", ".md", ".json", ".yaml", ".toml", ".txt", ".svg", ".bib"}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(
            part.startswith(".")
            or part in ("__pycache__", "build", "dist")
            or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_file() and path.suffix in suffixes:
            assert not host_path.search(path.read_text(encoding="utf-8")), relative


def test_bilingual_readmes_share_commands_prompt_and_result_tables():
    english, chinese = [
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("README.md", "README_zh.md")
    ]
    fenced = lambda text: re.findall(r"```(?:bash|json|bibtex)\n(.*?)```", text, re.S)
    assert fenced(english) == fenced(chinese)
    for text in (english, chinese):
        assert "```text\n" + PROMPT + "\n```" in text
        assert REVISION in text
        assert "/actions/workflows/cpu-tests.yml/badge.svg" in text
        assert len(re.findall(r"^### Step [1-4]", text, re.M)) == 4
    baseline = list(
        csv.DictReader((ROOT / "results/baseline_results.csv").open(encoding="utf-8"))
    )
    stages = list(
        csv.DictReader((ROOT / "results/stage_results.csv").open(encoding="utf-8"))
    )
    for split in ("original", "alt1"):
        per_seed = [
            statistics.mean(
                float(r["rmse_va"])
                for r in baseline
                if r["split"] == split and r["seed"] == seed
            )
            for seed in ("42", "13", "7")
        ]
        groups = [per_seed] + [
            [
                float(r["rmse_va"])
                for r in stages
                if r["split"] == split and r["stage"] == stage and r["group"] == "full"
            ]
            for stage in ("Stage1", "Stage2")
        ]
        for scores in groups:
            expected = f"{statistics.mean(scores):.4f} ± {statistics.stdev(scores):.4f}"
            assert expected in english and expected in chinese


def test_distribution_metadata_license_and_svg():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["license"]["file"] == "LICENSE"
    assert metadata["tool"]["setuptools"]["package-data"]["unified_va"] == [
        "configs/*.json",
        "resources/*.json",
    ]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Beichen Huang et al." in license_text
    assert "AS IS" in license_text
    diagram = ET.fromstring((ROOT / "assets/framework.svg").read_text(encoding="utf-8"))
    assert diagram.tag.endswith("svg")
    text = " ".join(diagram.itertext())
    for branch in ("Global", "Language", "Domain", "Compound"):
        assert branch in text
