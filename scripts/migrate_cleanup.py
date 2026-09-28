from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def move_all_files(source: Path, destination: Path):
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            print(f"SKIP (already exists): {target}")
            continue
        shutil.move(str(item), str(target))
        print(f"MOVED: {item} -> {target}")
    try:
        source.rmdir()
    except OSError:
        pass

def remove_if_exists(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"REMOVED: {path}")

def main():
    for dirname in [
        "data/raw",
        "data/processed",
        "models",
        "scripts",
        "tests",
        "docs",
        "app",
    ]:
        (ROOT / dirname).mkdir(parents=True, exist_ok=True)

    move_all_files(ROOT / "csv files of data", ROOT / "data/raw")

    model = ROOT / "phishing_model.joblib"
    if model.exists():
        target = ROOT / "models" / model.name
        if not target.exists():
            shutil.move(str(model), str(target))
            print(f"MOVED: {model} -> {target}")

    for name in ["__pycache__", "DEBUG files", "debug"]:
        remove_if_exists(ROOT / name)

    remove_if_exists(ROOT / ".DS_Store")

    print("\nMigration complete.")
    print("IMPORTANT: existing Python files were not moved.")
    print("Run the test suite before making the next code refactor.")

if __name__ == "__main__":
    main()
