# Автоматический скрипт для переименования NAM моделей
# Упрощает названия, убирая лишнюю информацию и технические детали

$modelsPath = "assets\nam_models"
$backupPath = "assets\nam_models_backup"

# Создаем backup
Write-Host "Creating backup..." -ForegroundColor Yellow
if (Test-Path $backupPath) {
    Remove-Item $backupPath -Recurse -Force
}
Copy-Item $modelsPath $backupPath -Recurse
Write-Host "Backup created at: $backupPath" -ForegroundColor Green

# Функция для автоматического упрощения имени
function Simplify-Name {
    param([string]$fileName)
    
    # Убираем расширение
    $name = $fileName -replace '\.nam$', ''
    
    # Убираем префиксы авторов
    $authorPrefixes = @(
        '^Helga B ',
        '^Tim R ',
        '^George B ',
        '^Keith B ',
        '^Jason Z ',
        '^Peter N ',
        '^Phillipe P ',
        '^Roman A ',
        '^Sascha S ',
        '^Luis R ',
        '^Mikhail K ',
        '^Tom C ',
        '^Tudor N '
    )
    
    foreach ($prefix in $authorPrefixes) {
        $name = $name -replace $prefix, ''
    }
    
    # Упрощаем технические детали
    $name = $name -replace '-ESR[0-9.,]+', ''  # Убираем ESR значения
    $name = $name -replace '_ESR[0-9.,]+', ''
    $name = $name -replace 'ESR[0-9.,]+', ''
    $name = $name -replace '_normalized-[0-9]+dB', ''
    $name = $name -replace 'normalized-[0-9]+dB', ''
    
    # Упрощаем gain levels (G4, G6, G8, G10, G12)
    $name = $name -replace '\s+G[0-9]+', ''
    $name = $name -replace '_G[0-9]+', ''
    $name = $name -replace '-G[0-9]+', ''
    
    # Упрощаем версии и технические детали
    $name = $name -replace '_V[0-9]+', ''
    $name = $name -replace '-V[0-9]+', ''
    $name = $name -replace '\(FIXED\)', ''
    $name = $name -replace '\(GROUND HUM\)', ''
    $name = $name -replace '_\(v1\)', ''
    $name = $name -replace '\(v1\)', ''
    
    # Упрощаем switch positions
    $name = $name -replace '_switch[0-9]+', ''
    $name = $name -replace 'switch[0-9]+', ''
    
    # Упрощаем параметры настроек (слишком детальные)
    $name = $name -replace 'Input@[^_]+', ''
    $name = $name -replace 'Dep@[0-9.]+', ''
    $name = $name -replace 'Gir@[0-9.]+', ''
    $name = $name -replace 'Pres@[0-9.]+', ''
    $name = $name -replace 'Bas@[0-9.]+', ''
    $name = $name -replace 'Mid@[0-9.]+', ''
    $name = $name -replace 'Tre@[0-9.]+', ''
    $name = $name -replace 'Mas@[0-9.]+', ''
    $name = $name -replace 'Gain@[0-9.]+', ''
    $name = $name -replace 'CHAR@[^_]+', ''
    $name = $name -replace 'ERA@[^_]+', ''
    $name = $name -replace 'EDG@[^_]+', ''
    $name = $name -replace 'FEL@[^_]+', ''
    $name = $name -replace 'Poweramp_', 'Poweramp '
    $name = $name -replace 'Knob@[0-9]+', ''
    $name = $name -replace 'Tone@[0-9.]+', ''
    $name = $name -replace 'Level@[0-9.]+', ''
    $name = $name -replace 'Overdrive@[0-9.]+', ''
    $name = $name -replace 'Output@[0-9.]+', ''
    $name = $name -replace 'Gain@[0-9.]+', ''
    $name = $name -replace '100HZ@[0-9]+', ''
    $name = $name -replace 'Bump@[^_]+', ''
    $name = $name -replace 'Vol@[0-9.]+', ''
    $name = $name -replace 'Drv@[0-9.]+', ''
    $name = $name -replace 'Brgt@[0-9.]+', ''
    $name = $name -replace 'Atk@[0-9.]+', ''
    $name = $name -replace 'SW[0-9]@[^_]+', ''
    
    # Стандартизируем названия педалей
    $name = $name -replace 'MXR Drive', 'M77'
    $name = $name -replace 'MXR M77', 'M77'
    $name = $name -replace 'OD808', '808'
    $name = $name -replace 'TS9', 'TS9'
    $name = $name -replace '805', '805'
    $name = $name -replace 'DS1', 'DS1'
    $name = $name -replace 'HM2', 'HM2'
    
    # Упрощаем "NoBoost" -> "Clean"
    $name = $name -replace 'No Boost', 'Clean'
    $name = $name -replace 'NoBoost', 'Clean'
    $name = $name -replace 'No Drive', 'Clean'
    $name = $name -replace 'NoDrive', 'Clean'
    
    # Упрощаем каналы
    $name = $name -replace 'chan2', 'Ch2'
    $name = $name -replace 'Channel 1', 'Ch1'
    $name = $name -replace 'Channel 2', 'Ch2'
    $name = $name -replace 'ch ', ' '
    $name = $name -replace ' ch ', ' '
    $name = $name -replace ' ch-', ' '
    $name = $name -replace 'Red ch', 'Red'
    $name = $name -replace 'Green ch', 'Green'
    $name = $name -replace 'Blue ch', 'Blue'
    
    # Упрощаем режимы
    $name = $name -replace 'L2', 'Lead'
    $name = $name -replace 'CL', 'Clean'
    $name = $name -replace 'CR', 'Crunch'
    $name = $name -replace 'OD1', 'OD1'
    $name = $name -replace 'OD2', 'OD2'
    $name = $name -replace 'GR', 'Green'
    $name = $name -replace 'OR', 'Orange'
    $name = $name -replace 'RD', 'Red'
    
    # Упрощаем длинные описания
    $name = $name -replace 'kinda bass heavy with medium distortion pure everything turned all the way up tone', 'Heavy'
    $name = $name -replace 'heavy distortion meant to be standalone less bass more honk', 'Standalone'
    $name = $name -replace 'light distortion with dimed EQ knobs use before an amp', 'EQ'
    $name = $name -replace 'all noon EQ', ''
    $name = $name -replace 'nice crunchy rhythm tone', ''
    $name = $name -replace 'all dimed no shift', ''
    $name = $name -replace 'droom metal at its best', ''
    $name = $name -replace 'preamp & 840 power amp with Boss DS1 Overdrive', 'DS1'
    
    # Упрощаем специфичные фразы
    $name = $name -replace 'NAM Capture ', ''
    $name = $name -replace 'Chase Tone Secret Pre Deco', ''
    $name = $name -replace 'Chase Tone Secret Pre Deco v2', ''
    $name = $name -replace ' - ', ' '
    $name = $name -replace '  ', ' '  # Двойные пробелы
    $name = $name -replace '  ', ' '
    
    # Упрощаем Feather/Lite/Std
    $name = $name -replace '_Feather', ' Feather'
    $name = $name -replace '-Feather', ' Feather'
    $name = $name -replace '_Lite', ' Lite'
    $name = $name -replace '-Lite', ' Lite'
    $name = $name -replace '_Std', ''
    $name = $name -replace '-Std', ''
    $name = $name -replace '_standard', ''
    $name = $name -replace '-standard', ''
    
    # Упрощаем скобки и специальные символы
    $name = $name -replace '\[', ''
    $name = $name -replace '\]', ''
    $name = $name -replace '@', ' '
    $name = $name -replace '_', ' '
    $name = $name -replace '  ', ' '  # Двойные пробелы
    $name = $name -replace '  ', ' '
    
    # Убираем лишние пробелы в начале и конце
    $name = $name.Trim()
    
    # Если имя слишком длинное, обрезаем
    if ($name.Length -gt 60) {
        $name = $name.Substring(0, 60).Trim()
    }
    
    return $name
}

Write-Host "`nStarting automatic renaming process..." -ForegroundColor Yellow
$files = Get-ChildItem -Path $modelsPath -Filter "*.nam"
$count = 0
$skipped = 0

foreach ($file in $files) {
    $oldName = $file.Name
    $newName = Simplify-Name -fileName $oldName
    
    # Если имя не изменилось или слишком короткое, пропускаем
    if ($newName -eq ($oldName -replace '\.nam$', '') -or $newName.Length -lt 3) {
        Write-Host "Skipped: $oldName" -ForegroundColor Gray
        $skipped++
        continue
    }
    
    $newNameWithExt = $newName + ".nam"
    $newPath = Join-Path $modelsPath $newNameWithExt
    
    # Проверяем, не существует ли уже файл с таким именем
    if (Test-Path $newPath) {
        Write-Host "Warning: $newNameWithExt already exists, skipping $oldName" -ForegroundColor Yellow
        $skipped++
        continue
    }
    
    try {
        Rename-Item -Path $file.FullName -NewName $newNameWithExt
        Write-Host "Renamed: $oldName -> $newNameWithExt" -ForegroundColor Cyan
        $count++
    }
    catch {
        Write-Host "Error renaming $oldName : $_" -ForegroundColor Red
        $skipped++
    }
}

Write-Host "`nRenaming complete!" -ForegroundColor Green
Write-Host "Processed: $count files" -ForegroundColor Cyan
Write-Host "Skipped: $skipped files" -ForegroundColor Yellow
Write-Host "Backup available at: $backupPath" -ForegroundColor Cyan

