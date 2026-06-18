# Run as Administrator: comments GitHub entries in hosts when local proxy is not running.
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"
$hostsPath = "$env:SystemRoot\System32\drivers\etc\hosts"
$backup = "$hostsPath.bak_ovdeploy_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $hostsPath $backup
Write-Host "Backup: $backup"
$lines = Get-Content $hostsPath
$pattern = "github|githubusercontent"
$new = foreach ($line in $lines) {
    if ($line -match "^\s*#" -or $line.Trim() -eq "") { $line; continue }
    if ($line -match $pattern) { "# OVDeploy-push: $line" } else { $line }
}
Set-Content -Path $hostsPath -Value $new -Encoding UTF8
Write-Host "Commented GitHub-related hosts lines. Test: Test-NetConnection github.com -Port 443"
