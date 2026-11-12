param([string]$In, [string]$Pdf)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $d = $w.Documents.Open($In)
    $d.Fields.Update() | Out-Null
    $d.Repaginate()
    $pages = $d.ComputeStatistics(2)   # wdStatisticPages
    $d.Save()
    $d.ExportAsFixedFormat($Pdf, 17)   # wdExportFormatPDF
    $d.Close($false)
    Write-Output "pages=$pages"
} finally {
    $w.Quit()
}
