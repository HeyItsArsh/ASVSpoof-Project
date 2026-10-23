param(
    [string]$OutputDirectory = "data/synthetic",
    [string]$TextFile = "scripts/tts_prompts.txt"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$destination = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled })
if ($voices.Count -eq 0) { throw "No Windows SAPI voices installed. Install a Windows speech voice in Settings." }
$prompts = @(Get-Content -LiteralPath $TextFile | Where-Object { $_.Trim().Length -gt 0 })
if ($prompts.Count -eq 0) { throw "No prompts found" }
$manifest = @()
$index = 0
try {
    foreach ($voice in $voices) {
        $synth.SelectVoice($voice.VoiceInfo.Name)
        foreach ($rate in @(-1, 1)) {
            $synth.Rate = $rate
            foreach ($text in $prompts) {
                $name = "local_tts_{0:D4}.wav" -f $index
                $path = Join-Path $destination $name
                if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite $path" }
                $synth.SetOutputToWaveFile($path)
                $synth.Speak($text)
                $synth.SetOutputToNull()
                $manifest += [PSCustomObject]@{ file=$name; label="spoof"; engine="Windows SAPI System.Speech"; voice=$voice.VoiceInfo.Name; rate=$rate; text=[string]$text }
                $index++
            }
        }
    }
    ConvertTo-Json -InputObject @($manifest) -Depth 4 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $destination "manifest.json")
    Write-Output "Generated $index local synthetic clips in $destination"
} finally { $synth.Dispose() }
