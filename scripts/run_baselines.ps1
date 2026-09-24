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
$Models = @(
    [pscustomobject]@{Variant='braintranslator_bart'; Model='BrainTranslator'},
    [pscustomobject]@{Variant='t5_large'; Model='T5Translator'},
    [pscustomobject]@{Variant='pegasus_xsum'; Model='PegasusTranslator'},
    [pscustomobject]@{Variant='eeg2text_feature'; Model='EEG2TextFeatureAdapter'}
)

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

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing Python environment: $Python" }
if (-not (Test-Path -LiteralPath $Manifest)) { throw "Missing manifest: $Manifest" }
$ManifestHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Manifest).Hash.ToLowerInvariant()
if ($PlanOnly) {
    foreach ($item in $Models) {
        foreach ($seed in $Seeds) {
            $folder = Join-Path $Training ('{0}_seed{1}' -f $item.Variant,$seed)
            $train = if (Test-Path -LiteralPath (Join-Path $folder 'complete.json')) { 'skip' } else { 'run/resume' }
            $test = if (Test-DecodingPolicy (Join-Path $folder 'test_predictions.metrics.json')) { 'skip' } else { 'run' }
            $zero = if (Test-DecodingPolicy (Join-Path $folder 'zero_shot_predictions.metrics.json')) { 'skip' } else { 'run' }
            $teacher = if (Test-Path -LiteralPath (Join-Path $folder 'teacher_forced_test_predictions.metrics.json')) { 'skip' } else { 'run' }
            Write-Host ('{0} seed {1}: train={2}, test={3}, zero-shot={4}, teacher-forced={5}' -f $item.Variant,$seed,$train,$test,$zero,$teacher)
        }
    }
    return
}

New-Item -ItemType Directory -Force -Path $Training,$Logs | Out-Null
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$transcript = Join-Path $Logs ('baselines_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
Start-Transcript -Path $transcript
try {
    $provenance = Join-Path $Outputs 'provenance.json'
    if (-not (Test-Path -LiteralPath $provenance)) {
        & $Python (Join-Path $Repo 'experiments\capture_provenance.py') $provenance
        Assert-Exit 'Provenance capture'
    }
    foreach ($item in $Models) {
        foreach ($seed in $Seeds) {
            $env:PYTHONHASHSEED = [string]$seed
            $name = '{0}_seed{1}' -f $item.Variant,$seed
            $folder = Join-Path $Training $name
            $checkpoint = Join-Path $folder 'best.pt'
            $configPath = Join-Path $folder 'config.json'
            if (Test-Path -LiteralPath $configPath) {
                $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
                if ($config.model -ne $item.Model -or [int]$config.seed -ne $seed -or
                    $config.manifest_sha256 -ne $ManifestHash -or
                    [int]$config.batch_size -ne 8 -or [int]$config.gradient_accumulation -ne 8 -or
                    [int]$config.epochs -ne 30 -or [int]$config.patience -ne 5 -or
                    $config.normalization -ne 'train_feature' -or [double]$config.label_smoothing -ne 0 -or
                    [double]$config.custom_lr -ne 0.00002 -or [double]$config.pretrained_lr -ne 0.000002) {
                    throw "Existing run configuration differs from the recorded protocol: $name"
                }
                if ($item.Variant -eq 'eeg2text_feature') {
                    $source = Join-Path $Repo 'EEG-To-text\model_feature_eeg2text.py'
                    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant()
                    if ($config.variant_source_sha256 -ne $hash) { throw "EEG2Text-FA source changed: $name" }
                }
            }
            if (-not (Test-Path -LiteralPath (Join-Path $folder 'complete.json'))) {
                Write-Host "Train/resume $name"
                & $Python (Join-Path $Repo 'experiments\train_experiment.py') `
                    --manifest $Manifest --run-name $name --output-dir $Training `
                    --seed $seed --model $item.Model --device $Device --batch-size 8 `
                    --gradient-accumulation 8 --num-workers 2 --epochs 30 --patience 5 `
                    --custom-lr '2e-5' --pretrained-lr '2e-6' --normalization train_feature `
                    --label-smoothing 0 --amp-dtype bfloat16
                Assert-Exit "Training $name"
            }
            else { Write-Host "Skip completed training: $name" }
            foreach ($phase in @('test','zero_shot')) {
                $metrics = Join-Path $folder ("{0}_predictions.metrics.json" -f $phase)
                if (Test-DecodingPolicy $metrics) { Write-Host "Skip completed evaluation: $name / $phase"; continue }
                if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint needed for $name / $phase" }
                & $Python (Join-Path $Repo 'experiments\evaluate_experiment.py') `
                    --run-dir $folder --phase $phase --device $Device --batch-size $EvaluationBatchSize `
                    --num-beams 5 --max-length 32 --min-new-tokens 0 --repetition-penalty 1.5 `
                    --no-repeat-ngram-size 2 --length-penalty 1.4
                Assert-Exit "Evaluation $name / $phase"
            }
            $teacher = Join-Path $folder 'teacher_forced_test_predictions.metrics.json'
            if (-not (Test-Path -LiteralPath $teacher)) {
                if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint for teacher forcing: $name" }
                & $Python (Join-Path $Repo 'experiments\evaluate_teacher_forced.py') `
                    --run-dir $folder --phase test --device $Device --batch-size $EvaluationBatchSize
                Assert-Exit "Teacher-forced evaluation $name"
            }
        }
    }
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
Write-Host "Completed baselines. Transcript: $transcript"
