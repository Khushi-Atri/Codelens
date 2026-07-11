import subprocess, shutil, pathlib, os, stat

IGNORE_DIRS = {'.git','node_modules','__pycache__','.venv',
               'dist','build','.idea','.vscode','vendor'}

CODE_EXTENSIONS = {'.py','.js','.ts','.java','.cpp',
                   '.c','.go','.rs','.md','.jsx','.tsx','.ipynb'}

def force_remove(func, path, excinfo):
    """Handle read-only files on Windows when deleting .git folders."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repo(github_url: str, target_dir: str = "repo") -> str:
    """Clone a GitHub repo and return the local path."""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, onerror=force_remove)
    
    print(f"Cloning {github_url}...")
    subprocess.run(
        ["git", "clone", "--depth=1", github_url, target_dir],
        check=True,
        capture_output=True
    )
    print("Clone complete!")
    return target_dir

def collect_files(repo_path: str) -> list:
    """Walk the repo and return all indexable file paths."""
    files = []
    for p in pathlib.Path(repo_path).rglob("*"):
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.suffix in CODE_EXTENSIONS and p.is_file():
            if p.stat().st_size < 500_000:
                files.append(p)
    
    print(f"Found {len(files)} files to index")
    return files