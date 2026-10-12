from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def main():
    renderer_path = Path(sys.argv[1]).resolve()
    arguments = sys.argv[2:]
    spec = importlib.util.spec_from_file_location("document_renderer", renderer_path)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    dependencies = Path(sys.executable).parent.parent
    os.environ["PATH"] = str(dependencies / "native" / "poppler" / "Library" / "bin") + os.pathsep + os.environ["PATH"]

    def convert_with_word(doc_path, user_profile, convert_tmp_dir, stem, verbose):
        pdf = Path(convert_tmp_dir) / (stem + ".pdf")
        environment = os.environ.copy()
        environment["TRACK4_RENDER_INPUT"] = str(Path(doc_path).resolve())
        environment["TRACK4_RENDER_PDF"] = str(pdf.resolve())
        script = r'''
$ErrorActionPreference = 'Stop'
$word = $null
$document = $null
$privateInstance = $false
try {
    $word = New-Object -ComObject Word.Application
    $privateInstance = $word.Documents.Count -eq 0
    if (-not $privateInstance) { throw 'Word instance is not private; refusing to touch user documents' }
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($env:TRACK4_RENDER_INPUT, $false, $true)
    $document.ExportAsFixedFormat($env:TRACK4_RENDER_PDF, 17)
} finally {
    $saveOption = 0
    if ($null -ne $document) { $document.Close([ref]$saveOption); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) }
    if ($null -ne $word) {
        if ($privateInstance) { $word.Quit([ref]$saveOption) }
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
}
'''
        completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                                   env=environment, capture_output=True, text=True, timeout=60)
        if completed.returncode or not pdf.exists():
            raise RuntimeError(completed.stderr or "Word did not export a PDF")
        return str(pdf), "Windows Word read-only PDF export"

    renderer.convert_to_pdf = convert_with_word
    sys.argv = [str(renderer_path), *arguments]
    renderer.main()


if __name__ == "__main__":
    main()
