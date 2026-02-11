# Финальный скрипт для переименования оставшихся файлов Tudor N
# Правильно обрабатывает файлы с квадратными скобками

$modelsPath = "assets\nam_models"

function Simplify-TudorName {
    param([string]$fileName)
    
    # Убираем расширение
    $name = $fileName -replace '\.nam$', ''
    
    # Убираем префикс "Tudor N "
    $name = $name -replace '^Tudor N ', ''
    
    # Убираем квадратные скобки, но сохраняем содержимое
    $name = $name -replace '\[', ''
    $name = $name -replace '\]', ' '
    
    # Извлекаем ESR значение для уникальности
    $esrValue = $null
    if ($name -match 'ESR[_\s-]([0-9.,]+)') {
        $esrValue = $matches[1]
    }
    
    # Определяем тип: Mesa, Suhr RL Input, или Suhr RL Poweramp
    $ampType = ""
    if ($name -match 'Mesa') {
        $ampType = "Mesa 2x12"
    } elseif ($name -match 'Suhr RL') {
        if ($name -match 'Poweramp') {
            $ampType = "Suhr RL Poweramp"
        } else {
            $ampType = "Suhr RL"
        }
    }
    
    # Убираем все параметры настроек
    $name = $name -replace 'Input@[^_\s]+', ''
    $name = $name -replace 'Dep@[0-9.]+', ''
    $name = $name -replace 'Gir@[0-9.]+', ''
    $name = $name -replace 'Pres@[0-9.]+', ''
    $name = $name -replace 'Bas@[0-9.]+', ''
    $name = $name -replace 'Mid@[0-9.]+', ''
    $name = $name -replace 'Tre@[0-9.]+', ''
    $name = $name -replace 'Mas@[0-9.]+', ''
    $name = $name -replace 'Gain@[0-9.]+', ''
    $name = $name -replace 'CHAR@[^_\s]+', ''
    $name = $name -replace 'ERA@[^_\s]+', ''
    $name = $name -replace 'EDG@[^_\s]+', ''
    $name = $name -replace 'FEL@[^_\s]+', ''
    $name = $name -replace 'Poweramp_', 'Poweramp '
    $name = $name -replace 'ESR[_\s-][0-9.,]+', ''
    $name = $name -replace '_normalized-[0-9]+dB', ''
    $name = $name -replace 'normalized-[0-9]+dB', ''
    
    # Очищаем от лишних символов
    $name = $name -replace '@', ' '
    $name = $name -replace '_', ' '
    $name = $name -replace '  ', ' '
    $name = $name -replace '  ', ' '
    $name = $name.Trim()
    
    # Формируем финальное имя
    if ($ampType) {
        $result = $ampType
        if ($esrValue) {
            $result = "$result ESR $esrValue"
        }
    } else {
        $result = $name
        if ($esrValue -and $result -notmatch $esrValue) {
            $result = "$result ESR $esrValue"
        }
    }
    
    return $result.Trim()
}

Write-Host "`nStarting final renaming process for Tudor N files..." -ForegroundColor Yellow
$files = Get-ChildItem -Path $modelsPath -Filter "*.nam" | Where-Object { 
    $_.Name -match "^Tudor N " 
}

$count = 0
$skipped = 0
$usedNames = @{}

foreach ($file in $files) {
    $oldName = $file.Name
    $baseNewName = Simplify-TudorName -fileName $oldName
    
    if ($baseNewName.Length -lt 3) {
        Write-Host "Skipped (name too short): $oldName" -ForegroundColor Gray
        $skipped++
        continue
    }
    
    # Проверяем уникальность и добавляем суффикс если нужно
    $newName = $baseNewName
    $counter = 1
    
    while ($usedNames.ContainsKey($newName) -or (Test-Path (Join-Path $modelsPath ($newName + ".nam")))) {
        # Извлекаем ESR для уникальности
        $esrMatch = $oldName -match 'ESR[_\s-]([0-9.,]+)'
        if ($esrMatch -and $counter -eq 1) {
            $esrValue = $matches[1]
            # Пытаемся добавить часть параметров для различия
            if ($oldName -match 'Dep@([0-9.]+)') {
                $depValue = $matches[1]
                $newName = "$baseNewName Dep$depValue"
            } elseif ($oldName -match 'Gain@([0-9.]+)') {
                $gainValue = $matches[1]
                $newName = "$baseNewName Gain$gainValue"
            } else {
                $newName = "$baseNewName $counter"
            }
        } else {
            $newName = "$baseNewName $counter"
        }
        $counter++
    }
    
    $newNameWithExt = $newName + ".nam"
    
    try {
        # Используем LiteralPath для файлов с квадратными скобками
        Rename-Item -LiteralPath $file.FullName -NewName $newNameWithExt -ErrorAction Stop
        Write-Host "Renamed: $oldName -> $newNameWithExt" -ForegroundColor Cyan
        $usedNames[$newName] = $true
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

