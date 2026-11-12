param(
    [string]$OutputPath = "artifacts/reports/machine-info.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Get-CommandVersion {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        return [ordered]@{ available = $false; output = $null }
    }

    try {
        $CommandOutput = & $Name @Arguments 2>&1 | Select-Object -First 3
        return [ordered]@{
            available = ($LASTEXITCODE -eq 0)
            output = ($CommandOutput -join "`n")
        }
    } catch {
        return [ordered]@{ available = $false; output = $_.Exception.Message }
    }
}

$Warnings = [System.Collections.Generic.List[string]]::new()

try {
    $OperatingSystem = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
} catch {
    $OperatingSystem = $null
    $Warnings.Add("CIM operating-system information is unavailable for this account.")
}

try {
    $Processor = Get-CimInstance Win32_Processor -ErrorAction Stop | Select-Object -First 1
} catch {
    $Processor = $null
    $Warnings.Add("CIM processor information is unavailable for this account.")
}

try {
    $VideoControllers = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
        Select-Object Name, DriverVersion, AdapterRAM)
} catch {
    $VideoControllers = @()
    $Warnings.Add("Video-controller details are unavailable; run as an authorized account if GPU diagnosis is needed.")
}

try {
    $Disks = @(Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" -ErrorAction Stop |
        Select-Object DeviceID,
            @{Name = "size_gb"; Expression = { [math]::Round($_.Size / 1GB, 2) }},
            @{Name = "free_gb"; Expression = { [math]::Round($_.FreeSpace / 1GB, 2) }})
} catch {
    $Disks = @(Get-PSDrive -PSProvider FileSystem |
        Select-Object @{Name = "DeviceID"; Expression = { $_.Name + ":" }},
            @{Name = "size_gb"; Expression = { [math]::Round(($_.Used + $_.Free) / 1GB, 2) }},
            @{Name = "free_gb"; Expression = { [math]::Round($_.Free / 1GB, 2) }})
    $Warnings.Add("Disk information used the PowerShell fallback because CIM is unavailable.")
}

$OsCaption = if ($OperatingSystem) {
    $OperatingSystem.Caption
} else {
    [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
}
$OsVersion = if ($OperatingSystem) {
    $OperatingSystem.Version
} else {
    [System.Environment]::OSVersion.VersionString
}
$OsArchitecture = if ($OperatingSystem) {
    $OperatingSystem.OSArchitecture
} else {
    [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
}
$ProcessorName = if ($Processor) { $Processor.Name } else { $env:PROCESSOR_IDENTIFIER }
$LogicalProcessors = if ($Processor) {
    $Processor.NumberOfLogicalProcessors
} else {
    [System.Environment]::ProcessorCount
}
$MemoryGb = if ($OperatingSystem) {
    [math]::Round($OperatingSystem.TotalVisibleMemorySize / 1MB, 2)
} else {
    $null
}

$Report = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    operating_system = [ordered]@{
        caption = $OsCaption
        version = $OsVersion
        architecture = $OsArchitecture
    }
    processor = [ordered]@{
        name = $ProcessorName
        logical_processors = $LogicalProcessors
    }
    memory_gb = $MemoryGb
    video_controllers = @($VideoControllers)
    disks = @($Disks)
    tools = [ordered]@{
        python = Get-CommandVersion -Name "python" -Arguments @("--version")
        conda = Get-CommandVersion -Name "conda" -Arguments @("--version")
        git = Get-CommandVersion -Name "git" -Arguments @("--version")
        nvidia_smi = Get-CommandVersion -Name "nvidia-smi" -Arguments @()
    }
    warnings = @($Warnings)
}

$ResolvedOutput = Join-Path $RepoRoot $OutputPath
$OutputDirectory = Split-Path -Parent $ResolvedOutput
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Report | ConvertTo-Json -Depth 6 | Set-Content -Path $ResolvedOutput -Encoding UTF8

Write-Host "Machine report written to: $ResolvedOutput"
