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
$user = gh api user -q .login
$fullName = "$user/$RepoName"

$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
gh repo view $fullName 2>$null | Out-Null
$repoExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEap

$git = "git"
if (Test-Path "C:\Program Files (x86)\Git\cmd\git.exe") {
    $git = "C:\Program Files (x86)\Git\cmd\git.exe"
}

if ($repoExists) {
    Write-Host "Remote repo exists; adding origin and pushing..."
    & $git remote remove origin 2>$null
    & $git remote add origin "https://github.com/$fullName.git"
    & $git push -u origin main
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "Creating public repo $fullName ..."
    gh repo create $RepoName $vis --source=. --remote=origin --push
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$url = gh repo view $fullName --json url -q .url
Write-Host "Done: $url"
