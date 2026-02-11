# Руководство по переименованию NAM моделей

## Анализ текущих названий

Проанализировано **260 NAM моделей**. Выявлены следующие проблемы:

### Проблемы:
1. **Префиксы авторов** - "Helga B ", "Tim R ", "George B " и т.д. (уже удаляются в коде)
2. **Технические детали** - ESR значения, gain levels (G4, G6, G8), switch positions
3. **Длинные описания** - "kinda bass heavy with medium distortion pure everything turned all the way up tone"
4. **Непонятные аббревиатуры** - CL, CR, OD1, OD2, GR, OR, RD
5. **Дублирование информации** - "NoBoost", "No Drive", "NoDrive" означают одно и то же

### Категории моделей:

#### Усилители Peavey (самая большая группа):
- **5150** - BlockLetter варианты
- **6505+** - Green/Red каналы
- **6534+** - разные бусты
- **JSX** - Ultra, Crunch
- **XXX** - 6L6/KT77 лампы

#### Marshall:
- **JCM2000** - Clean, Crunch, Lead
- **JCM900** - Dual Verb, разные каналы

#### Bugera:
- **333** - Clean, Crunch, Lead
- **6262** - Crunch, Lead
- **1990** - Lead

#### Педали:
- Boss: DS1, HM2, SD1, OS-2
- MXR: M77 (Custom Badass)
- Maxon: OD808
- Ibanez: TS9
- Klone, Plumes, и другие

## Решение

Созданы два скрипта для переименования:

### 1. `rename_models_auto.ps1` (РЕКОМЕНДУЕТСЯ)
**Автоматический скрипт**, который:
- Убирает префиксы авторов
- Удаляет технические детали (ESR, gain levels, switch positions)
- Стандартизирует названия педалей
- Упрощает длинные описания
- Заменяет "NoBoost" на "Clean"

**Использование:**
```powershell
.\rename_models_auto.ps1
```

**Преимущества:**
- Обрабатывает все 260 файлов автоматически
- Создает backup перед переименованием
- Универсальный подход

**Примеры преобразований:**
- `Helga B 5150 BlockLetter - NoBoost.nam` → `5150 Clean.nam`
- `Helga B 6505+ Red ch - MXR Drive.nam` → `6505+ Red M77.nam`
- `Tim R JCM2000 L2 G6.nam` → `JCM2000 Lead.nam`
- `Peter N HM-2_SWEDE_Feather_ESR-0.0097.nam` → `Boss HM2 Swede Feather.nam`

### 2. `rename_models.ps1` (Альтернатива)
**Скрипт с конкретными маппингами** для каждого файла:
- Более точный контроль
- Требует ручного добавления новых файлов
- Покрывает ~200 файлов

## После переименования

После выполнения скрипта можно **упростить функцию парсинга** в `PluginProcessor.cpp`:

### Текущий код (строки 1350-1375):
```cpp
// Удаляет известные префиксы авторов
juce::Array<juce::String> authorPrefixes;
authorPrefixes.add("Helga B ");
authorPrefixes.add("Tim R ");
// ... и т.д.

// Заменяет underscores на пробелы
fileName = fileName.replaceCharacter('_', ' ');
```

### Упрощенный код (после переименования):
```cpp
// Просто убираем расширение и заменяем underscores
juce::String fileName = file.getFileNameWithoutExtension();
fileName = fileName.replaceCharacter('_', ' ');
entry.displayName = fileName;
```

Или даже проще - если все файлы уже будут с правильными именами, можно вообще убрать парсинг:
```cpp
entry.displayName = file.getFileNameWithoutExtension();
```

## Рекомендации

1. **Запустите `rename_models_auto.ps1`** - он создаст backup и переименует все файлы
2. **Проверьте результат** - откройте папку `assets/nam_models` и убедитесь, что названия понятные
3. **Упростите код парсинга** - удалите логику удаления префиксов авторов из `scanModels()`
4. **Протестируйте** - убедитесь, что все модели отображаются корректно в LIBRARY вкладке

## Backup

Скрипт автоматически создает backup в `assets/nam_models_backup` перед переименованием.
Если что-то пойдет не так, можно восстановить:
```powershell
Remove-Item assets\nam_models -Recurse -Force
Rename-Item assets\nam_models_backup assets\nam_models
```

