"""
model_setup.py Session setup (auth, dependencies) for Kaggle/local
"""
import os
import subprocess

def setup_git_identity(email: str, name: str):
    """Set git identity for this session."""
    subprocess.run(f'git config --global user.email "{email}"', shell=True)
    subprocess.run(f'git config --global user.name "{name}"', shell=True)
    print(f"[model_setup] Git identity: {name} <{email}>")

def setup_hf_auth():
    """Set HF token from Kaggle Secrets."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("[model_setup] HF auth token set")
    except Exception as e:
        print(f"[model_setup] Warning: HF auth failed: {e}")

def install_deps():
    """Install requirements."""
    subprocess.run("pip install -q -r requirements.txt", shell=True, cwd="/kaggle/working/Enliven")
    print("[model_setup] Dependencies installed")

def clear_import_caches():
    """Clear Python import caches."""
    import sys
    import importlib
    for mod in list(sys.modules):
        if mod.startswith("src") or mod == "pipeline":
            del sys.modules[mod]
    sys.path_importer_cache.clear()
    importlib.invalidate_caches()
    print("[model_setup] Import caches cleared")
