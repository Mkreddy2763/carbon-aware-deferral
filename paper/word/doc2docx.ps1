param([string]$In, [string]$Out)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $d = $w.Documents.Open($In, $false, $true)
    $d.SaveAs2($Out, 16)   # wdFormatXMLDocument
    $d.Close($false)
} finally {
    $w.Quit()
}
