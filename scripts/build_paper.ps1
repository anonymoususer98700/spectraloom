param([switch]$SkipResults)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Paper = Join-Path $Repo 'paper_iclr'
$Python = Join-Path $Repo 'EEG-To-text\venv\Scripts\python.exe'
function Find-TeX([string]$Name) {
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\$Name.exe"
    if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    throw "Missing $Name. Install a TeX distribution and add its binaries to PATH."
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Failed: $Executable $Arguments. Build logs were retained." }
}
$XeLaTeX = Find-TeX 'xelatex'
$PdfLaTeX = Find-TeX 'pdflatex'
$BibTeX = Find-TeX 'bibtex'
if (-not $SkipResults) {
    if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python environment: $Python" }
    Invoke-Checked $Python @((Join-Path $PSScriptRoot 'update_paper_results.py'))
}
Push-Location (Join-Path $Paper 'figures')
try {
    foreach ($Figure in @('spectraloom_architecture','spectraloom_motivation')) {
        Invoke-Checked $XeLaTeX @('-interaction=batchmode','-halt-on-error',"$Figure.tex")
    }
} finally { Pop-Location }
Push-Location $Paper
try {
    Invoke-Checked $PdfLaTeX @('-interaction=batchmode','-halt-on-error','main.tex')
    Invoke-Checked $BibTeX @('main')
    Invoke-Checked $PdfLaTeX @('-interaction=batchmode','-halt-on-error','main.tex')
    Invoke-Checked $PdfLaTeX @('-interaction=batchmode','-halt-on-error','main.tex')
} finally { Pop-Location }
# Remove only explicit successful-build scratch targets inside the paper folder.
foreach ($Stem in @('main','figures\spectraloom_architecture','figures\spectraloom_motivation')) {
    foreach ($Extension in @('aux','log','out','blg','bbl')) {
        $Target = [IO.Path]::GetFullPath((Join-Path $Paper "$Stem.$Extension"))
        if (-not $Target.StartsWith($Paper + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe scratch path: $Target"
        }
        if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target }
    }
}
Write-Host "Built: $Paper\main.pdf"
