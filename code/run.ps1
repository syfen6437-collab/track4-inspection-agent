param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = "Stop"
$bundledPython = "C:\Users\MR\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$python = if ($env:TRACK4_PYTHON) { $env:TRACK4_PYTHON } elseif (Test-Path $bundledPython) { $bundledPython } else { "python" }
& $python "$PSScriptRoot\run_pipeline.py" @PipelineArgs
