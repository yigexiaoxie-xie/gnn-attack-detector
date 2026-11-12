param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ReportPath = "artifacts/reports/universal-detector-evaluation.json",
    [string]$ArchiveRoot = "artifacts/reports/training-runs"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath $ProjectRoot).Path

function Resolve-ProjectPath([string]$PathValue) {
    if ([IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return Join-Path $root $PathValue
}

$reportFile = Resolve-ProjectPath $ReportPath
$archiveBase = Resolve-ProjectPath $ArchiveRoot
if (-not (Test-Path -LiteralPath $reportFile -PathType Leaf)) {
    throw "Evaluation report was not found: $reportFile"
}

$report = Get-Content -LiteralPath $reportFile -Raw | ConvertFrom-Json
if ($null -eq $report.metrics_by_profile) {
    throw "Evaluation report has no metrics_by_profile section: $reportFile"
}

New-Item -ItemType Directory -Path $archiveBase -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$archiveDir = Join-Path $archiveBase $stamp
$suffix = 1
while (Test-Path -LiteralPath $archiveDir) {
    $archiveDir = Join-Path $archiveBase ("{0}-{1:00}" -f $stamp, $suffix)
    $suffix++
}
New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null

Copy-Item -LiteralPath $reportFile -Destination (Join-Path $archiveDir "evaluation.json")

$datasetRows = @()
foreach ($entry in $report.metrics_by_profile.PSObject.Properties) {
    $metrics = $entry.Value
    $matrix = $metrics.confusion_matrix
    if ($null -eq $matrix) {
        throw "Dataset '$($entry.Name)' has no confusion_matrix in the evaluation report."
    }

    $tp = [int]$matrix.tp
    $tn = [int]$matrix.tn
    $fp = [int]$matrix.fp
    $fn = [int]$matrix.fn
    $count = [int]$metrics.count
    $correct = $tp + $tn
    $datasetRows += [PSCustomObject][ordered]@{
        dataset = [string]$entry.Name
        accuracy = [double]$metrics.accuracy
        correct = $correct
        total = $count
        balanced_accuracy = [double]$metrics.balanced_accuracy
        recall = [double]$metrics.recall
        specificity = [double]$metrics.specificity
        precision = [double]$metrics.precision
        f1 = [double]$metrics.f1
        tp = $tp
        tn = $tn
        fp = $fp
        fn = $fn
    }
}

if ($datasetRows.Count -eq 0) {
    throw "Evaluation report contains no dataset metrics: $reportFile"
}

$macroAverage = ($datasetRows | Measure-Object -Property accuracy -Average).Average
$totalCorrect = ($datasetRows | Measure-Object -Property correct -Sum).Sum
$totalCount = ($datasetRows | Measure-Object -Property total -Sum).Sum
$weightedAverage = if ($totalCount -gt 0) { [double]$totalCorrect / [double]$totalCount } else { 0.0 }
$rootPrefix = $root
if (-not $rootPrefix.EndsWith("\") -and -not $rootPrefix.EndsWith("/")) {
    $rootPrefix += "\"
}
$sourceReport = $reportFile
if ($reportFile.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    $sourceReport = $reportFile.Substring($rootPrefix.Length).Replace("\", "/")
}
$summary = [ordered]@{
    archived_at_local = (Get-Date).ToString("o")
    source_report = $sourceReport
    profiles = @($report.profiles)
    selected_attacks = @($report.selected_attacks)
    dataset_count = $datasetRows.Count
    average_accuracy = [double]$macroAverage
    weighted_accuracy = [double]$weightedAverage
    datasets = @($datasetRows)
}
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $archiveDir "dataset_summary.json") -Encoding utf8
$datasetRows | ForEach-Object { [PSCustomObject]$_ } | Export-Csv -LiteralPath (Join-Path $archiveDir "dataset_summary.csv") -NoTypeInformation -Encoding utf8

$lines = @(
    "Detector training result archive",
    "Archived at: $($summary.archived_at_local)",
    "Source report: $($summary.source_report)",
    "",
    ("Average accuracy: {0:P2}" -f $summary.average_accuracy),
    ("Weighted accuracy: {0:P2}" -f $summary.weighted_accuracy),
    "",
    "Dataset results:"
)
foreach ($row in $datasetRows) {
    $lines += ("{0}: accuracy {1:P2} ({2}/{3}), attack recall {4:P2}, F1 {5:P2}" -f $row.dataset, $row.accuracy, $row.correct, $row.total, $row.recall, $row.f1)
}
$lines | Set-Content -LiteralPath (Join-Path $archiveDir "summary.txt") -Encoding utf8

Write-Host "Result archived: $archiveDir"
Write-Host ("Average accuracy: {0:P2}" -f $summary.average_accuracy)
