"""Build a submission zip from an explicit allowlist; never include task data."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = {"app.py", "cli.py", "schemas.py", "reporting.py", "Dockerfile.test", ".dockerignore",
         ".env.example", ".gitignore", "requirements.txt", "requirements.lock", "pyproject.toml",
         "README.md", "Design.md", "VERIFICATION.md"}
DIRECTORIES = {"agents", "workflow", "validation", "review_agent", "examples", "tests", "scripts"}
FIGURES = {f"docs/figures/{name}.{ext}" for name in
           ("architecture", "workflow", "collaboration", "snapshots") for ext in ("png", "svg")}


def package(output):
    output = Path(output)
    if output.exists():
        raise ValueError("提交包输出文件必须不存在")
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        paths = [ROOT / name for name in FILES | FIGURES]
        for name in DIRECTORIES:
            paths.extend((ROOT / name).rglob("*"))
        for path in sorted(paths):
            relative = path.relative_to(ROOT)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts[:-1]):
                continue
            allowed = relative.as_posix() in FILES | FIGURES or (relative.parts[0] in DIRECTORIES and path.suffix in (".py", ".md", ".lock"))
            if allowed and path.is_file() and not path.is_symlink():
                archive.write(path, "codeflow/" + relative.as_posix())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="codeflow-submission.zip")
    print(package(parser.parse_args().output))
