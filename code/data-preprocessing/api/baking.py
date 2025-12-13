import re
import sys
import glob
import subprocess
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback if installed


PYPROJECT = Path("pyproject.toml")


def load_pyproject():
    if not PYPROJECT.exists():
        print("pyproject.toml not found.")
        sys.exit(1)
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def bump_patch_version(old_version: str) -> str:
    parts = old_version.split(".")
    if len(parts) < 3:
        # falls z.B. "0.1" -> einfach Patch anhängen
        parts += ["0"] * (3 - len(parts))
    major, minor, patch = map(int, parts[:3])
    patch += 1
    return f"{major}.{minor}.{patch}"


def update_version_in_file(old_version: str, new_version: str):
    text = PYPROJECT.read_text(encoding="utf-8")

    pattern = rf'version\s*=\s*"{re.escape(old_version)}"'
    replacement = f'version = "{new_version}"'

    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n == 0:
        print("Could not resolve/update version entry in pyproject.toml.")
        sys.exit(1)

    PYPROJECT.write_text(new_text, encoding="utf-8")


def main():
    data = load_pyproject()

    project_name = None
    version = None

    if "project" in data:
        proj = data["project"]
        project_name = proj.get("name")
        version = proj.get("version")
    if (project_name is None or version is None) and "tool" in data and "poetry" in data["tool"]:
        poetry = data["tool"]["poetry"]
        project_name = project_name or poetry.get("name")
        version = version or poetry.get("version")

    if project_name is None or version is None:
        print("Could not read 'name' or 'version' from pyproject.toml.")
        sys.exit(1)

    print(f"Current version: {version}")
    new_version = bump_patch_version(version)
    print(f"New version:    {new_version}")

    update_version_in_file(version, new_version)


    print(f"\n[1/3] Deinstall {project_name} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", project_name],
        check=False,
    )

    # Build
    print("[2/3] Build Package (python -m build) ...")
    res = subprocess.run([sys.executable, "-m", "build"], check=False)
    if res.returncode != 0:
        print("Build failed.")
        sys.exit(res.returncode)

    # Wheel suchen (zu neuer Version passend)
    print("[3/3] Install new Wheel ...")
    dist = Path("dist")
    wheels = glob.glob(str(dist / f"{project_name.replace('-', '_')}-{new_version}*.whl"))

    if not wheels:
        # Fallback: irgendein Wheel mit neuer Versionsnummer
        wheels = glob.glob(str(dist / f"*-{new_version}*.whl"))

    if not wheels:
        print(f"No Wheel for version {new_version} in dist/ found.")
        sys.exit(1)

    wheel_path = wheels[0]
    print(f"Install {wheel_path} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", wheel_path],
        check=True,
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
