param(
    [string]$Device = 'cuda:0',
    [int]$EvaluationBatchSize = 16,
    [string]$OutputRoot = '',
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Repo 'EEG-To-text\venv\Scripts\python.exe'
$Manifest = Join-Path $Repo 'experiments\manifests\zuco1_to_zuco2_sentence_seed2026.csv'
$Outputs = if ($OutputRoot) {
    if ([System.IO.Path]::IsPathRooted($OutputRoot)) { [System.IO.Path]::GetFullPath($OutputRoot) }
    else { [System.IO.Path]::GetFullPath((Join-Path $Repo $OutputRoot)) }
} else { Join-Path $Repo 'experiments\outputs' }
$Training = Join-Path $Outputs 'training'
$Logs = Join-Path $Outputs 'logs'
$Seeds = @(42, 100, 312)

function Assert-Exit([string]$What) {
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

function Test-DecodingPolicy([string]$MetricsPath) {
    if (-not (Test-Path -LiteralPath $MetricsPath)) { return $false }
    try {
        $decode = (Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json).decoding
        return ([int]$decode.num_beams -eq 5 -and [int]$decode.max_length -eq 32 -and
            [int]$decode.min_new_tokens -eq 0 -and [int]$decode.no_repeat_ngram_size -eq 2 -and
            [double]$decode.repetition_penalty -eq 1.5 -and [double]$decode.length_penalty -eq 1.4)
    }
    catch { return $false }
}

function Invoke-Run($Item) {
    $env:PYTHONHASHSEED = [string]$Item.Seed
    $name = '{0}_seed{1}' -f $Item.Variant, $Item.Seed
    $folder = Join-Path $Training $name
    $complete = Join-Path $folder 'complete.json'
    $checkpoint = Join-Path $folder 'best.pt'
    $configPath = Join-Path $folder 'config.json'
    if (Test-Path -LiteralPath $configPath) {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        if ($config.model -ne $Item.Model -or [int]$config.seed -ne $Item.Seed -or
            $config.manifest_sha256 -ne $ManifestHash -or $config.conv_kernels -ne '3,5,7' -or
            [int]$config.batch_size -ne 8 -or [int]$config.gradient_accumulation -ne 8 -or
            [int]$config.epochs -ne 30 -or [int]$config.patience -ne 5 -or
            $config.normalization -ne 'train_feature' -or [double]$config.label_smoothing -ne 0 -or
            [double]$config.custom_lr -ne 0.00002 -or [double]$config.pretrained_lr -ne 0.000002) {
            throw "Existing run configuration differs from the recorded protocol: $name"
        }
        if ($Item.Variant -eq 'uniform_sba' -and -not $config.uniform_sba) { throw "Wrong ablation: $name" }
        if ($Item.Variant -eq 'no_multiscale' -and -not $config.ablate_multiscale) { throw "Wrong ablation: $name" }
        if ($Item.Variant -eq 'no_cab' -and -not $config.ablate_cab) { throw "Wrong ablation: $name" }
        if ($Item.Variant -in @('length_only', 'language_prior', 'noise_only') -and
            $config.train_control -ne $Item.Control) { throw "Wrong training control: $name" }
        if ($Item.Variant -eq 'static_sba') {
            $source = Join-Path $Repo 'EEG-To-text\model_static_sba.py'
            $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
            if ($config.variant_source_sha256 -ne $hash) { throw "Static SBA source changed: $name" }
        }
    }
    if (-not (Test-Path -LiteralPath $complete)) {
        Write-Host "Train/resume $name"
        $args = @(
            (Join-Path $Repo 'experiments\train_experiment.py'),
            '--manifest', $Manifest, '--run-name', $name, '--output-dir', $Training,
            '--seed', $Item.Seed, '--device', $Device, '--batch-size', 8,
            '--gradient-accumulation', 8, '--num-workers', 2, '--epochs', 30,
            '--patience', 5, '--custom-lr', '2e-5', '--pretrained-lr', '2e-6',
            '--normalization', 'train_feature', '--label-smoothing', 0,
            '--amp-dtype', 'bfloat16'
        ) + $Item.Flags
        & $Python @args
        Assert-Exit "Training $name"
    }
    else { Write-Host "Skip completed training: $name" }
    foreach ($phase in @('test', 'zero_shot')) {
        $metrics = Join-Path $folder ("{0}_predictions.metrics.json" -f $phase)
        if (Test-DecodingPolicy $metrics) { Write-Host "Skip completed evaluation: $name / $phase"; continue }
        if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint needed for $name / $phase" }
        & $Python (Join-Path $Repo 'experiments\evaluate_experiment.py') `
            --run-dir $folder --phase $phase --device $Device --batch-size $EvaluationBatchSize `
            --num-beams 5 --max-length 32 --min-new-tokens 0 --repetition-penalty 1.5 `
            --no-repeat-ngram-size 2 --length-penalty 1.4
        Assert-Exit "Evaluation $name / $phase"
    }
    if ($Item.TeacherForced) {
        $metrics = Join-Path $folder 'teacher_forced_test_predictions.metrics.json'
        if (-not (Test-Path -LiteralPath $metrics)) {
            if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint for teacher forcing: $name" }
            & $Python (Join-Path $Repo 'experiments\evaluate_teacher_forced.py') `
                --run-dir $folder --phase test --device $Device --batch-size $EvaluationBatchSize
            Assert-Exit "Teacher-forced evaluation $name"
        }
    }
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python environment: $Python" }
if (-not (Test-Path -LiteralPath $Manifest)) { throw "Missing manifest: $Manifest" }
$ManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLowerInvariant()
$Runs = @()
foreach ($seed in $Seeds) {
    $Runs += [pscustomobject]@{Variant='spectraloom'; Seed=$seed; Model='EEGConformer'; Flags=@('--model','EEGConformer'); Control='real'; TeacherForced=$true}
}
foreach ($seed in $Seeds) {
    $Runs += [pscustomobject]@{Variant='uniform_sba'; Seed=$seed; Model='EEGConformer'; Flags=@('--model','EEGConformer','--uniform-sba'); Control='real'; TeacherForced=$false}
    $Runs += [pscustomobject]@{Variant='no_multiscale'; Seed=$seed; Model='EEGConformer'; Flags=@('--model','EEGConformer','--ablate-multiscale'); Control='real'; TeacherForced=$false}
    $Runs += [pscustomobject]@{Variant='no_cab'; Seed=$seed; Model='EEGConformer'; Flags=@('--model','EEGConformer','--ablate-cab'); Control='real'; TeacherForced=$false}
    $Runs += [pscustomobject]@{Variant='static_sba'; Seed=$seed; Model='EEGConformerStaticSBA'; Flags=@('--model','EEGConformerStaticSBA'); Control='real'; TeacherForced=$false}
}
# Historical trained information controls were predeclared for seed 312 only.
foreach ($item in @(
    @{Variant='length_only'; Control='zero_values_original_mask'},
    @{Variant='language_prior'; Control='zero_values_full_mask'},
    @{Variant='noise_only'; Control='gaussian_original_mask'}
)) {
    $Runs += [pscustomobject]@{Variant=$item.Variant; Seed=312; Model='EEGConformer'; Flags=@('--model','EEGConformer','--train-control',$item.Control); Control=$item.Control; TeacherForced=$false}
}

if ($PlanOnly) {
    foreach ($item in $Runs) {
        $folder = Join-Path $Training ('{0}_seed{1}' -f $item.Variant,$item.Seed)
        $train = if (Test-Path -LiteralPath (Join-Path $folder 'complete.json')) { 'skip' } else { 'run/resume' }
        $test = if (Test-DecodingPolicy (Join-Path $folder 'test_predictions.metrics.json')) { 'skip' } else { 'run' }
        $zero = if (Test-DecodingPolicy (Join-Path $folder 'zero_shot_predictions.metrics.json')) { 'skip' } else { 'run' }
        Write-Host ('{0} seed {1}: train={2}, test={3}, zero-shot={4}' -f $item.Variant,$item.Seed,$train,$test,$zero)
    }
    Write-Host 'Then teacher-forced SpectraLoom and six paired EEG-reliance conditions x two phases x three seeds.'
    return
}

New-Item -ItemType Directory -Force -Path $Training,$Logs | Out-Null
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$transcript = Join-Path $Logs ('spectraloom_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $transcript
try {
    $provenance = Join-Path $Outputs 'provenance.json'
    if (-not (Test-Path -LiteralPath $provenance)) {
        & $Python (Join-Path $Repo 'experiments\capture_provenance.py') $provenance
        Assert-Exit 'Provenance capture'
    }
    foreach ($item in $Runs) { Invoke-Run $item }
    foreach ($seed in $Seeds) {
        $env:PYTHONHASHSEED = [string]$seed
        $folder = Join-Path $Training "spectraloom_seed$seed"
        if (-not (Test-Path -LiteralPath (Join-Path $folder 'best.pt'))) { throw "Missing SpectraLoom checkpoint for reliance: $folder" }
        foreach ($phase in @('test','zero_shot')) {
            foreach ($condition in @('real','gaussian','language_prior','mismatched','shuffled','wrong_subject')) {
                & $Python (Join-Path $Repo 'experiments\evaluate_experiment.py') `
                    --run-dir $folder --phase $phase --device $Device --batch-size $EvaluationBatchSize `
                    --num-beams 5 --max-length 32 --min-new-tokens 0 --repetition-penalty 1.5 `
                    --no-repeat-ngram-size 2 --length-penalty 1.4 `
                    --reliance-condition $condition --perturbation-seed 2026
                Assert-Exit "Reliance $seed / $phase / $condition"
            }
        }
    }
    & $Python (Join-Path $Repo 'experiments\aggregate_reliance.py') `
        --training-dir $Training --seeds 42 100 312 --variant spectraloom `
        --output (Join-Path $Outputs 'reliance_summary.json')
    Assert-Exit 'Reliance aggregation'
    foreach ($phase in @('test','zero_shot')) {
        $summary = if ($phase -eq 'test') { 'in_domain_summary.json' } else { 'zero_shot_summary.json' }
        & $Python (Join-Path $Repo 'experiments\aggregate_multiseed.py') `
            --training-dir $Training --phase $phase --output (Join-Path $Outputs $summary)
        Assert-Exit "$phase aggregation"
    }
    & $Python (Join-Path $Repo 'experiments\aggregate_teacher_forced.py') `
        --training-dir $Training --output (Join-Path $Outputs 'teacher_forced_summary.json')
    Assert-Exit 'Teacher-forced aggregation'
    & $Python (Join-Path $Repo 'experiments\build_result_tables.py') --outputs $Outputs
    Assert-Exit 'Table generation'
}
finally { Stop-Transcript }
Write-Host "Completed SpectraLoom suite. Transcript: $transcript"
