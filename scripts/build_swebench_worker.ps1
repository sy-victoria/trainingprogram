param(
    [Parameter(Mandatory=$true)][string]$SourceCheckout,
    [Parameter(Mandatory=$true)][string]$BuildDirectory
)
$ErrorActionPreference = 'Stop'
$revision = '02e7a74ffd0b707aab73d203fe87bdc7c76afc8e'
$repoRoot = Split-Path $PSScriptRoot -Parent
$resolvedSource = (Resolve-Path -LiteralPath $SourceCheckout).Path
$head = git -C $resolvedSource rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $revision) { throw 'Official source revision mismatch' }
$buildRoot = [IO.Path]::GetFullPath($BuildDirectory)
New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
git -C $resolvedSource archive --format=tar "--output=$buildRoot/swebench-source.tar" $revision
if ($LASTEXITCODE -ne 0) { throw 'Source archive failed' }
Copy-Item -LiteralPath "$repoRoot/docker/SWEbench.Dockerfile" -Destination "$buildRoot/Dockerfile"
Copy-Item -LiteralPath "$repoRoot/requirements/swebench-worker-resolved.txt" -Destination "$buildRoot/swebench-worker-resolved.txt"
docker build --progress plain -t trainingprogram-swebench:02e7a74 $buildRoot
if ($LASTEXITCODE -ne 0) { throw 'Worker image build failed' }
