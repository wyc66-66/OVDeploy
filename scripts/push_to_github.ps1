# Create public GitHub repo and push from ovdeploy-public root.
param(
    [string]$RepoName = "OVDeploy",
    [switch]$Public = $true
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root
Write-Host "Repo root: $root"

# Connectivity check
$tcp = Test-NetConnection github.com -Port 443 -WarningAction SilentlyContinue
if (-not $tcp.TcpTestSucceeded) {
    Write-Error @"
Cannot reach github.com:443.
Your hosts file may point GitHub to 127.0.0.1 without a running proxy.
See docs/GITHUB_UPLOAD.md — run scripts/fix_github_hosts.ps1 as Administrator, or start your GitHub proxy.
"@
}

gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in. Run: gh auth login"
    gh auth login --hostname github.com --git-protocol https --web
}

$vis = if ($Public) { "--public" } else { "--private" }
$existing = gh repo view $RepoName 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote repo exists; adding origin and pushing..."
    $user = (gh api user -q .login)
    git remote remove origin 2>$null
    git remote add origin "https://github.com/$user/$RepoName.git"
    git push -u origin main
} else {
    gh repo create $RepoName $vis --source=. --remote=origin --push
}

$url = gh repo view --json url -q .url
Write-Host "Done: $url"
