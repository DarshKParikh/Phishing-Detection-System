from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_files_exist():
    expected = [
        "phishing_scraper.py",
        "train_model.py",
        "ui.py",
        "gpt_phishing_analyzer.py",
        "app.py",
    ]

    for filename in expected:
        assert (ROOT / filename).exists(), f"Missing {filename}"


def test_required_directories_exist():
    for dirname in [
        "data",
        "models",
        "scripts",
        "tests",
        "docs",
    ]:
        assert (ROOT / dirname).is_dir(), f"Missing {dirname}"
