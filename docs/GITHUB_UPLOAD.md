# Upload to GitHub

**Live repo:** https://github.com/wyc66-66/OVDeploy — already published on `main`.

If you only need to **update** this repository:

```powershell
git add .
git commit -m "Your message"
git push origin main
```

---

First-time upload from a fresh clone (reference):

## What you must provide

| Item | Required | Notes |
|------|----------|-------|
| GitHub account | Yes | [github.com](https://github.com) |
| `gh auth login` | Yes | Browser login once on this PC |
| Repository name | Yes | Default: **OVDeploy** (public) |
| Author name/email | Optional | Amend commit before push if you want your profile shown |

You do **not** need: weights, tokens (if using browser login), or the full development tree.

## Regenerate public package (A + B + nuScenes)

From `submission-a/`:

```powershell
python scripts/package_github.py --clean
```

`--clean` auto-preserves `ovdeploy-public/.git`.

## Step 0: Fix GitHub network (important on this machine)

Your `hosts` file may map `github.com` → `127.0.0.1` while no local proxy is running on port 443.

Check:

```powershell
Get-Content C:\Windows\System32\drivers\etc\hosts | Select-String github
Test-NetConnection github.com -Port 443
```

If GitHub is blocked:

1. **Option A** — Start the proxy/accelerator that owns those `hosts` entries, then retry.
2. **Option B** — Run **as Administrator** from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fix_github_hosts.ps1
```

This backs up `hosts` and comments out GitHub-related lines. Restore from backup if needed.

## Step 1: Login

```powershell
cd path\to\ovdeploy-public
gh auth login
```

Choose: GitHub.com → HTTPS → Login with a web browser.

## Step 2: Create public repo and push

```powershell
powershell -ExecutionPolicy Bypass -File scripts/push_to_github.ps1
```

Or manually:

```powershell
gh repo create OVDeploy --public --source=. --remote=origin --push
```

Result: https://github.com/wyc66-66/OVDeploy

## Step 3 (optional): Set commit author

```powershell
git -c user.name="Your Name" -c user.email="YOUR_ID@users.noreply.github.com" commit --amend --reset-author --no-edit
git push -f origin main
```

Find noreply email: GitHub → Settings → Emails.

## Troubleshooting

- `127.0.0.1:443 refused` → Step 0 (hosts / proxy).
- `repository already exists` → use another name or `gh repo create OVDeploy-bench ...`.
- `auth required` → rerun `gh auth login`.
