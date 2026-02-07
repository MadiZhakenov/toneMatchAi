# Инструкции по Компиляции (Final Build)

## 1. Предварительные Требования

### Windows:
- Visual Studio 2019 или 2022 (с компонентами C++ Desktop Development)
- CMake 3.20+
- JUCE Framework (должен быть подключен через CMake или как подмодуль)
- Python 3.9+ (для тестирования Python-моста)

### macOS:
- Xcode 12+ (Command Line Tools)
- CMake 3.20+
- JUCE Framework
- Python 3.9+

### Linux:
- GCC 9+ или Clang 10+
- CMake 3.20+
- JUCE Framework
- Python 3.9+

---

## 2. Подготовка Проекта

### 2.1 Проверить структуру проекта

Убедитесь, что структура проекта соответствует следующей:

```
toneMatchAi/
├── plugin/
│   ├── CMakeLists.txt          # Главный CMake файл
│   ├── Source/                 # Исходный код плагина
│   ├── Scripts/
│   │   └── run_match.py        # Python-скрипт для AI matching
│   ├── Resources/
│   │   └── default_preset.json
│   └── ThirdParty/
│       └── NeuralAmpModelerCore/  # NAM библиотека
├── assets/
│   ├── nam_models/             # 259 NAM моделей
│   └── impulse_responses/      # IR файлы
└── src/                        # Python оптимизатор
```

### 2.2 Проверить зависимости

- Убедиться, что `NeuralAmpModelerCore` собран (или настроен как подмодуль)
- Проверить, что Eigen и nlohmann/json доступны
- Убедиться, что JUCE Framework правильно подключен в CMakeLists.txt

---

## 3. Компиляция через CMake (Windows)

### Шаг 1: Настройка CMake

Откройте PowerShell или Command Prompt и выполните:

```powershell
cd plugin
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Если JUCE не найден автоматически:**

```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release -DJUCE_DIR="C:/path/to/JUCE"
```

**Если нужно указать генератор Visual Studio:**

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
```

### Шаг 2: Генерация Visual Studio Solution

CMake автоматически создаст файл `Project.sln` в папке `build/`.

### Шаг 3: Открыть Solution и настроить Export Targets

1. Откройте `plugin/build/Project.sln` в Visual Studio
2. В Solution Explorer найдите проект плагина
3. Проверьте настройки проекта:
   - **Configuration:** Release
   - **Platform:** x64
   - **Output Directory:** `plugin/build/Release/`
4. Если используется JUCE CMake integration, экспорт VST3/AU настраивается автоматически

### Шаг 4: Финальная компиляция

**Вариант A: Через командную строку**

```powershell
cmake --build . --config Release --target ToneMatchAI_VST3
```

**Вариант B: Через Visual Studio**

- Выберите проект `ToneMatchAI_VST3` (или `ToneMatchAI_Standalone`)
- Build → Build Solution (F7)
- Или правый клик на проекте → Build

### Шаг 5: Проверка артефактов

После компиляции проверьте наличие:

- **VST3:** `plugin/build/Release/ToneMatchAI_artefacts/Release/VST3/ToneMatchAI.vst3`
- **Standalone:** `ToneMatchAI_artefacts/Release/Standalone/ToneMatchAI.exe`

---

## 4. Компиляция через CMake (macOS)

### Шаг 1: Настройка CMake

```bash
cd plugin
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Если JUCE не найден:**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DJUCE_DIR="/path/to/JUCE"
```

### Шаг 2: Компиляция

**Для VST3:**

```bash
cmake --build . --config Release --target ToneMatchAI_VST3
```

**Для AU:**

```bash
cmake --build . --config Release --target ToneMatchAI_AU
```

**Для Standalone:**

```bash
cmake --build . --config Release --target ToneMatchAI_Standalone
```

### Шаг 3: Установка плагина

Плагины автоматически копируются в системные папки:

- **VST3:** `~/Library/Audio/Plug-Ins/VST3/`
- **AU:** `~/Library/Audio/Plug-Ins/Components/`
- **Standalone:** `~/Library/Application Support/ToneMatchAI/`

---

## 5. Компиляция через CMake (Linux)

### Шаг 1: Настройка CMake

```bash
cd plugin
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Шаг 2: Компиляция

```bash
cmake --build . --config Release --target ToneMatchAI_VST3
```

### Шаг 3: Установка

```bash
sudo cmake --install . --config Release
```

Или скопируйте `.so` файл вручную:

```bash
cp plugin/build/Release/ToneMatchAI_artefacts/Release/VST3/ToneMatchAI.so ~/.vst3/
```

---

## 6. Альтернатива: Использование JUCE Projucer (если есть .jucer файл)

Если проект использует JUCE Project (`.jucer` файл):

### Шаг 1: Открыть Projucer

1. Запустите JUCE Projucer
2. File → Open → выберите `ToneMatchAI.jucer`

### Шаг 2: Настроить Export Targets

1. В левой панели выберите "Exporters"
2. Выберите нужные форматы:
   - **Visual Studio 2022** (Windows)
   - **Xcode** (macOS)
   - **Linux Makefile** (Linux)
3. Для каждого экспортера:
   - Включите **VST3**
   - Включите **AU** (macOS)
   - Включите **Standalone** (опционально)

### Шаг 3: Сохранить и сгенерировать

1. File → Save Project
2. File → Save and Open in IDE (или вручную откройте сгенерированный solution/project)

### Шаг 4: Компиляция

- В Visual Studio / Xcode выберите конфигурацию **Release**
- Build → Build Solution / Product → Build

---

## 7. Пост-компиляционные Шаги

### 7.1 Копирование Python-скрипта

Убедитесь, что `plugin/Scripts/run_match.py` доступен рядом с бинарником плагина.

**Windows:**

```powershell
# Скопировать скрипт в папку с плагином
Copy-Item plugin/Scripts/run_match.py plugin/build/Release/ToneMatchAI_artefacts/Release/VST3/
```

**macOS/Linux:**

```bash
# Скопировать скрипт в папку с плагином
cp plugin/Scripts/run_match.py ~/Library/Audio/Plug-Ins/VST3/ToneMatchAI.vst3/Contents/Resources/
```

**Альтернатива:** Убедитесь, что плагин может найти скрипт относительно исполняемого файла или через абсолютный путь.

### 7.2 Проверка путей к моделям

Плагин должен находить модели относительно исполняемого файла или через абсолютные пути.

**Проверьте:**
- `assets/nam_models/` - содержит 259 `.nam` файлов
- `assets/impulse_responses/` - содержит `.wav` IR файлы

**Если модели должны быть рядом с плагином:**

```powershell
# Windows
xcopy /E /I assets\nam_models plugin\build\Release\ToneMatchAI_artefacts\Release\VST3\nam_models
xcopy /E /I assets\impulse_responses plugin\build\Release\ToneMatchAI_artefacts\Release\VST3\impulse_responses
```

### 7.3 Тестирование скомпилированного плагина

1. Загрузите плагин в DAW (Reaper, Cubase, Logic, и т.д.)
2. Выполните тесты из `TESTING_PROTOCOL.md`
3. Проверьте, что:
   - Плагин загружается без ошибок
   - UI отображается корректно
   - Звук обрабатывается без артефактов
   - AI matching работает

---

## 8. Отладка Проблем Компиляции

### Проблема: CMake не находит JUCE

**Решение:**
```powershell
cmake .. -DJUCE_DIR="C:/path/to/JUCE" -DCMAKE_BUILD_TYPE=Release
```

Или установите переменную окружения:
```powershell
$env:JUCE_DIR = "C:/path/to/JUCE"
```

### Проблема: NeuralAmpModelerCore не компилируется

**Решение:**
- Убедитесь, что Eigen установлен и доступен
- Проверьте, что `ThirdParty/NeuralAmpModelerCore/CMakeLists.txt` правильно настроен
- Попробуйте собрать NeuralAmpModelerCore отдельно

### Проблема: Ошибки линковки

**Решение:**
- Проверьте, что все зависимости правильно указаны в CMakeLists.txt
- Убедитесь, что библиотеки скомпилированы для той же архитектуры (x64)
- Проверьте пути к библиотекам

### Проблема: Плагин не загружается в DAW

**Решение:**
- Убедитесь, что плагин скомпилирован в правильном формате (VST3 для VST3 хоста)
- Проверьте, что плагин скопирован в правильную системную папку
- Проверьте логи DAW на наличие ошибок загрузки
- Убедитесь, что все зависимости (DLL на Windows, dylib на macOS) доступны

---

## 9. Оптимизация Release Build

### Рекомендуемые настройки CMake:

```powershell
cmake .. -DCMAKE_BUILD_TYPE=Release ^
         -DCMAKE_CXX_FLAGS_RELEASE="/O2 /Ob2 /DNDEBUG" ^
         -DCMAKE_C_FLAGS_RELEASE="/O2 /Ob2 /DNDEBUG"
```

### macOS:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
        -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG"
```

### Linux:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native" \
        -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG -march=native"
```

---

## 10. Чеклист Компиляции

- [ ] Все зависимости установлены (JUCE, CMake, компилятор)
- [ ] Структура проекта корректна
- [ ] CMake конфигурация успешна (нет ошибок)
- [ ] Проект компилируется без ошибок
- [ ] Нет критических warnings (можно игнорировать предупреждения от зависимостей)
- [ ] Плагин создан в правильной папке
- [ ] Python-скрипт скопирован рядом с плагином
- [ ] Модели доступны (если требуются)
- [ ] Плагин загружается в DAW
- [ ] Базовое тестирование пройдено

---

**Удачи с компиляцией! 🎸**

