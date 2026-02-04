#!/usr/bin/env python3
"""
推送当前 `Value_Function` 文件夹到 GitHub 的小脚本。
功能：
 - 在目标目录创建/更新 `.gitignore`（忽略 data/、*.err、*.out 等）
 - 收集并合并现有 `requirements*.txt` 到根目录 `requirements.txt`
 - 如果缺少 `README.md`，会生成一个简要说明文件
 - 初始化 git（如无），提交代码（受 .gitignore 控制），并推送到指定 GitHub 仓库

用法示例：
  export GITHUB_TOKEN=ghp_xxx
  python push_value_function_to_github.py --target /project/peilab/junhao/Value_Function --repo youruser/yourrepo

注意：脚本会尝试创建远端仓库（如果不存在），并会把 token 用于创建/推送。请确保 token 有相应权限。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import glob
import requests


DEFAULT_GITIGNORE = '''# Ignore data and large files
data/
**/data/
*.err
*.out
*.log
*.ckpt
*.pth
__pycache__/
.ipynb_checkpoints/
env/
venv/
dist/
build/
'''


def ensure_gitignore(target: Path):
    gi = target / '.gitignore'
    if gi.exists():
        return
    gi.write_text(DEFAULT_GITIGNORE)
    print(f"Wrote .gitignore to {gi}")


def gather_requirements(target: Path):
    # Find all requirements*.txt under target and merge unique non-comment lines
    req_files = list(target.rglob('requirements*.txt'))
    lines = []
    for rf in req_files:
        try:
            with open(rf, 'r') as f:
                for l in f:
                    s = l.strip()
                    if s and not s.startswith('#'):
                        lines.append(s)
        except Exception:
            continue

    unique = sorted(set(lines))
    out = target / 'requirements.txt'
    if unique:
        out.write_text('\n'.join(unique) + '\n')
        print(f"Wrote merged requirements to {out} ({len(unique)} packages)")
    else:
        # If no requirements found, create an empty placeholder
        if not out.exists():
            out.write_text('# requirements collected by push script\n')
            print(f"No requirement files found; created placeholder {out}")


def ensure_readme(target: Path):
    rd = target / 'README.md'
    if rd.exists():
        return
    content = f"# Value_Function\n\nThis repository contains the `Value_Function` code exported from the local workspace.\n\nSee `requirements.txt` for dependencies.\n"
    rd.write_text(content)
    print(f"Created README.md at {rd}")


def run(cmd, cwd=None, env=None):
    print('> ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def github_repo_exists(owner: str, repo: str, token: str):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    r = requests.get(url, headers={'Authorization': f'token {token}'})
    return r.status_code == 200


def create_github_repo(owner: str, repo: str, token: str):
    # Determine authenticated user
    user_resp = requests.get('https://api.github.com/user', headers={'Authorization': f'token {token}'})
    if user_resp.status_code != 200:
        raise RuntimeError('Failed to authenticate to GitHub API with provided token')
    auth_user = user_resp.json().get('login')

    if owner == auth_user:
        # create under user
        url = 'https://api.github.com/user/repos'
        payload = {'name': repo, 'private': False}
    else:
        # try to create under org
        url = f'https://api.github.com/orgs/{owner}/repos'
        payload = {'name': repo, 'private': False}

    r = requests.post(url, json=payload, headers={'Authorization': f'token {token}'})
    if r.status_code in (201, 202):
        print(f"Created remote repository {owner}/{repo}")
        return True
    elif r.status_code == 422:
        # already exists or validation failed
        print(f"Could not create repo (422): {r.text}")
        return False
    else:
        print(f"Failed to create repo: {r.status_code} {r.text}")
        return False


def get_authenticated_user(token: str) -> str:
    """Return authenticated GitHub login for the token."""
    r = requests.get('https://api.github.com/user', headers={'Authorization': f'token {token}'})
    if r.status_code != 200:
        raise RuntimeError('Failed to get authenticated user from GitHub API')
    return r.json().get('login')


def repo_has_push_permission(owner: str, repo: str, token: str) -> bool:
    """Check repository permissions for the token (push permission)."""
    url = f'https://api.github.com/repos/{owner}/{repo}'
    r = requests.get(url, headers={'Authorization': f'token {token}'})
    if r.status_code != 200:
        return False
    perms = r.json().get('permissions', {})
    return perms.get('push', False)


def init_commit_push(target: Path, owner: str, repo: str, token: str, branch: str = 'main'):
    # init git if needed
    if not (target / '.git').exists():
        run(['git', 'init', '--initial-branch', branch], cwd=str(target))
    # set user if not set
    try:
        run(['git', 'config', 'user.name'], cwd=str(target))
    except subprocess.CalledProcessError:
        run(['git', 'config', 'user.name', 'auto-push-script'], cwd=str(target))
        run(['git', 'config', 'user.email', 'auto@example.com'], cwd=str(target))

    run(['git', 'add', '.'], cwd=str(target))
    # commit if there are changes
    try:
        run(['git', 'commit', '-m', 'chore: initial push of Value_Function code'], cwd=str(target))
    except subprocess.CalledProcessError:
        print('Nothing to commit or commit failed')

    # Use authenticated username in remote URL to avoid ambiguous credential mapping
    try:
        auth_user = get_authenticated_user(token)
    except Exception:
        auth_user = None

    if auth_user:
        remote_url = f'https://{auth_user}:{token}@github.com/{owner}/{repo}.git'
    else:
        remote_url = f'https://{token}@github.com/{owner}/{repo}.git'
    # add or set remote
    try:
        run(['git', 'remote', 'add', 'origin', remote_url], cwd=str(target))
    except subprocess.CalledProcessError:
        # remote may already exist: set-url
        run(['git', 'remote', 'set-url', 'origin', remote_url], cwd=str(target))

    # before pushing, check permissions
    if not repo_has_push_permission(owner, repo, token):
        print(f"Warning: token does not seem to have push permission for {owner}/{repo}. Aborting push.")
        print("Possible causes: token missing 'repo' scope, repo owned by another user/org without collaborator access, or branch protection.")
        return

    # push
    run(['git', 'push', '-u', 'origin', branch], cwd=str(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Path to Value_Function folder')
    parser.add_argument('--repo', type=str, required=True, help='GitHub repo in format owner/repo')
    parser.add_argument('--token', type=str, default=os.environ.get('GITHUB_TOKEN'), help='GitHub token (or set GITHUB_TOKEN env)')
    parser.add_argument('--branch', type=str, default='main')
    args = parser.parse_args()

    if not args.token:
        print('Provide a GitHub token via --token or set GITHUB_TOKEN in environment')
        sys.exit(1)

    target = Path(args.target).resolve()
    if not target.exists():
        print(f'Target path does not exist: {target}')
        sys.exit(1)

    owner_repo = args.repo.split('/')
    if len(owner_repo) != 2:
        print('Repo must be in format owner/repo')
        sys.exit(1)
    owner, repo = owner_repo

    ensure_gitignore(target)
    gather_requirements(target)
    ensure_readme(target)

    # create remote repo if missing
    if not github_repo_exists(owner, repo, args.token):
        created = create_github_repo(owner, repo, args.token)
        if not created:
            print('Failed to create remote repo; aborting')
            sys.exit(1)

    init_commit_push(target, owner, repo, args.token, branch=args.branch)


if __name__ == '__main__':
    main()
