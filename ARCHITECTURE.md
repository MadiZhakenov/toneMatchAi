# Финальная Архитектура Проекта ToneMatch AI

## 1. Структура Проекта

```
toneMatchAi/
├── plugin/                          # JUCE VST3/AU Plugin
│   ├── CMakeLists.txt               # CMake конфигурация
│   ├── Source/
│   │   ├── PluginProcessor.cpp/h    # Главный AudioProcessor
│   │   ├── PluginEditor.cpp/h       # UI компонент
│   │   ├── DSP/
│   │   │   ├── DSPChain.cpp/h       # 6-stage DSP цепочка
│   │   │   └── NAMProcessor.cpp/h   # Обертка NeuralAmpModelerCore
│   │   ├── Bridge/
│   │   │   ├── PythonBridge.cpp/h   # ChildProcess для запуска Python
│   │   │   └── MatchResult.h         # Структура результата
│   │   ├── Preset/
│   │   │   ├── PresetManager.cpp/h  # Сохранение/загрузка пресетов
│   │   │   └── ...
│   │   └── UI/
│   │       ├── SlotComponent.cpp/h  # Слоты для FX/AMP/IR
│   │       └── KnobStrip.cpp/h      # Ряд слайдеров
│   ├── Scripts/
│   │   └── run_match.py             # Python CLI wrapper
│   ├── Resources/
│   │   └── default_preset.json      # Дефолтный пресет
│   └── ThirdParty/
│       └── NeuralAmpModelerCore/    # NAM библиотека (Eigen + NAM)
│
├── src/                             # Python AI Optimizer
│   ├── core/
│   │   ├── optimizer.py             # ToneOptimizer (Grid Search + Post-FX)
│   │   ├── nam_processor.py         # NAM обработка
│   │   ├── io.py                    # Audio I/O
│   │   ├── analysis.py              # Спектральный анализ
│   │   ├── loss.py                  # Функции потерь
│   │   ├── matching.py              # Match EQ фильтры
│   │   ├── processor.py             # Аудио обработка
│   │   └── ddsp_processor.py        # Дифференцируемая DSP
│   └── app.py                       # Streamlit UI (опционально)
│
├── assets/
│   ├── nam_models/                  # 259 NAM моделей (.nam файлы)
│   └── impulse_responses/           # IR кабинеты (.wav файлы)
│
├── run_universal_match.py           # Standalone Python скрипт
├── run_final_tune.py                # Standalone Python скрипт (фиксированный риг)
└── requirements.txt                 # Python зависимости
```

---

## 2. Поток Данных (Real-time Audio Processing)

### 2.1 Основной Audio Processing Flow

```
[DAW Audio Input]
    ↓
[PluginProcessor::processBlock]
    ↓
[DI Capture Buffer] (если нажата кнопка MATCH TONE!)
    ├─ capturedDI (AudioBuffer<float>)
    └─ capturedDIWritePos (atomic<int>)
    ↓
[DSPChain::process]
    ├─ Stage 1: Input Gain
    │   └─ juce::dsp::Gain<float>
    ├─ Stage 2: NAM Pedal
    │   └─ NAMProcessor (NeuralAmpModelerCore)
    ├─ Stage 3: NAM Amp
    │   └─ NAMProcessor (NeuralAmpModelerCore)
    ├─ Stage 4: IR Cabinet
    │   └─ juce::dsp::Convolution
    ├─ Stage 5: Pre-EQ
    │   └─ juce::dsp::IIR::Filter<float> (Peak Filter)
    ├─ Stage 6: Delay
    │   └─ juce::dsp::DelayLine<float> + wet/dry mix
    ├─ Stage 7: Reverb
    │   └─ juce::dsp::Reverb
    └─ Stage 8: Final EQ Gain
        └─ juce::dsp::Gain<float>
    ↓
[DAW Audio Output]
```

### 2.2 Параметры и APVTS

Все параметры управляются через `AudioProcessorValueTreeState` (APVTS):

- `inputGain` (-24.0 to +24.0 dB)
- `preEqGainDb` (-12.0 to +12.0 dB)
- `preEqFreqHz` (400.0 to 3000.0 Hz)
- `reverbWet` (0.0 to 0.7)
- `reverbRoomSize` (0.0 to 1.0)
- `delayTimeMs` (50.0 to 500.0 ms)
- `delayMix` (0.0 to 0.5)
- `finalEqGainDb` (-3.0 to +3.0 dB)

Параметры читаются в real-time через атомики (`std::atomic<float>*`), что гарантирует lock-free доступ.

---

## 3. AI Matching Flow

### 3.1 Полный Процесс AI Matching

```
[User clicks "MATCH TONE!" button]
    ↓
[PluginProcessor::triggerMatch]
    ├─ Start capturing DI → saved to temp/tonematch_di.wav
    └─ User selects reference file via FileChooser
    ↓
[PythonBridge::startMatch]
    ├─ Launch ChildProcess:
    │   python run_match.py --di <di_path> --ref <ref_path> --out <json_path>
    └─ Wait for completion (timeout: 10 minutes)
    ↓
[run_match.py - Python Script]
    ├─ Load DI and Reference audio files
    ├─ ToneOptimizer.optimize_universal()
    │   ├─ Stage 1: Smart Sommelier
    │   │   └─ Reference analysis (genre, gain, spectral characteristics)
    │   ├─ Stage 2: Fast Grid Search
    │   │   ├─ Evaluate 261 NAM models (FX + AMP combinations)
    │   │   ├─ Test with different IRs
    │   │   └─ Find TOP-3 rigs with lowest loss
    │   └─ Stage 3: Deep Post-FX Optimization
    │       ├─ Use "Sighted" Optimizer (Adam)
    │       ├─ Minimize 4-component error vector:
    │       │   ├─ Harmonic Loss
    │       │   ├─ Envelope Loss
    │       │   ├─ Spectral Shape Loss
    │       │   └─ Brightness Loss
    │       └─ Optimize 7 Post-FX parameters
    └─ Write JSON output:
        {
          "rig": {
            "fx_nam": "...",
            "amp_nam": "...",
            "ir": "...",
            "fx_nam_path": "...",
            "amp_nam_path": "...",
            "ir_path": "..."
          },
          "params": {
            "input_gain_db": 0.0,
            "pre_eq_gain_db": 0.0,
            "pre_eq_freq_hz": 800.0,
            "reverb_wet": 0.0,
            "reverb_room_size": 0.5,
            "delay_time_ms": 100.0,
            "delay_mix": 0.0,
            "final_eq_gain_db": 0.0
          },
          "loss": 0.0
        }
    ↓
[PythonBridge::parseResultJson]
    ├─ Read JSON file from temp directory
    ├─ Parse rig paths and parameters
    └─ Create MatchResult struct
    ↓
[PluginProcessor::onMatchComplete] (called on message thread)
    ├─ Build RigParameters from MatchResult
    └─ Call applyNewRig()
    ↓
[PluginProcessor::applyNewRig]
    ├─ Load NAM models:
    │   ├─ pedal.loadModel(fx_nam_path)
    │   └─ amp.loadModel(amp_nam_path)
    ├─ Load IR:
    │   └─ loadIR(ir_path)
    └─ Update APVTS parameters:
        └─ setParam() for each parameter
    ↓
[DSPChain loads new models]
    ├─ NAMProcessor::loadModel() (thread-safe with mutex)
    ├─ Convolution::loadImpulseResponse()
    └─ Models ready for real-time processing
    ↓
[UI Updates]
    ├─ SlotComponent displays model names
    ├─ Sliders reflect new parameter values
    └─ "LAST RIG FOUND" label updated
```

### 3.2 Thread Safety

- **PythonBridge:** Запускается в отдельном потоке (`juce::Thread`)
- **Model Loading:** Защищено `std::mutex` в `NAMProcessor`
- **Parameter Updates:** Через `juce::MessageManager::callAsync()` на message thread
- **Real-time Processing:** Lock-free доступ к параметрам через атомики

---

## 4. Preset System

### 4.1 Preset Format

Пресеты сохраняются в JSON формате:

```json
{
  "name": "My Preset",
  "rig": {
    "fx_nam": "DS1",
    "amp_nam": "5150 BlockLetter",
    "ir": "BlendOfAll",
    "fx_nam_path": "C:/path/to/DS1.nam",
    "amp_nam_path": "C:/path/to/5150.nam",
    "ir_path": "C:/path/to/BlendOfAll.wav"
  },
  "params": {
    "input_gain_db": 0.0,
    "pre_eq_gain_db": 2.5,
    "pre_eq_freq_hz": 1200.0,
    "reverb_wet": 0.3,
    "reverb_room_size": 0.7,
    "delay_time_ms": 250.0,
    "delay_mix": 0.2,
    "final_eq_gain_db": -1.0
  },
  "loss": 0.001234
}
```

### 4.2 Preset Flow

```
[Save Preset]
    ↓
[PresetManager::savePreset]
    ├─ Serialize APVTS state → JSON
    ├─ Add rig paths (fx_nam_path, amp_nam_path, ir_path)
    ├─ Add current parameter values
    └─ Write to .json file
    ↓
[Load Preset]
    ↓
[PresetManager::loadPreset]
    ├─ Read JSON file
    ├─ Parse rig paths and params
    └─ Call varToState()
    ↓
[PresetManager::varToState]
    ├─ Build RigParameters struct
    └─ Call processor.applyNewRig()
    ↓
[PluginProcessor::applyNewRig]
    ├─ Load NAM models
    ├─ Load IR
    └─ Update APVTS parameters
```

### 4.3 Preset Storage

- **Default Location:** `Documents/ToneMatchAI/Presets/`
- **Format:** `.json` файлы
- **Paths:** Абсолютные пути к моделям (для переносимости можно использовать относительные)

---

## 5. UI Architecture

### 5.1 UI Layout

```
[PluginEditor]
├── [Top Bar]
│   ├── Title: "ToneMatch AI"
│   └── Version info
│
├── [Zone 1: AI Matching]
│   ├── "MATCH TONE!" button
│   ├── "LAST RIG FOUND" label
│   └── Final EQ Gain slider
│
├── [Zone 2: Post-FX Parameters]
│   ├── [Delay Block]
│   │   ├── Delay Time slider
│   │   └── Delay Mix slider
│   └── [Reverb Block]
│       ├── Reverb Wet slider
│       └── Reverb Size slider
│
├── [Zone 3: Equipment Slots]
│   ├── FX Slot (SlotComponent)
│   ├── AMP Slot (SlotComponent)
│   └── IR Slot (SlotComponent)
│
└── [Zone 4: Additional Controls]
    ├── Input Gain knob (KnobStrip)
    ├── Pre-EQ knobs (KnobStrip)
    └── "SAVE AS PRESET" button
```

### 5.2 UI Components

- **SlotComponent:** Отображает имя загруженной модели, позволяет выбрать файл
- **KnobStrip:** Горизонтальный ряд слайдеров с лейблами
- **SliderAttachment:** Автоматическая синхронизация слайдеров с APVTS параметрами

---

## 6. Ключевые Зависимости

### 6.1 C++ Plugin

- **JUCE Framework:**
  - Audio processing (`juce_audio_processors`)
  - DSP utilities (`juce_dsp`)
  - UI components (`juce_gui_basics`)
  - Plugin format (VST3, AU)

- **NeuralAmpModelerCore:**
  - NAM моделирование (WaveNet, LSTM, ConvNet)
  - Зависит от Eigen для линейной алгебры

- **Eigen:**
  - Матричные операции для нейронных сетей

- **nlohmann/json:**
  - Парсинг JSON результатов от Python

### 6.2 Python Optimizer

- **PyTorch 2.0+:**
  - Дифференцируемая обработка аудио
  - Neural network training (опционально)

- **librosa:**
  - Спектральный анализ
  - Извлечение признаков

- **scipy:**
  - Оптимизация (Adam, L-BFGS-B)

- **pedalboard:**
  - Аудио эффекты (Reverb, Delay, EQ)

- **numpy:**
  - Массивы и численные операции

---

## 7. Модели и Ресурсы

### 7.1 NAM Models

- **Location:** `assets/nam_models/`
- **Format:** `.nam` файлы (Neural Amp Modeler format)
- **Count:** 259 моделей (педали и усилители)
- **Loading:** Через `NAMProcessor::loadModel()`

### 7.2 Impulse Responses

- **Location:** `assets/impulse_responses/`
- **Format:** `.wav` файлы
- **Usage:** Кабинетное моделирование через Convolution

---

## 8. Threading Model

### 8.1 Threads

1. **Audio Thread (Real-time):**
   - `PluginProcessor::processBlock()`
   - `DSPChain::process()`
   - Должен быть lock-free

2. **Message Thread (UI):**
   - `PluginEditor` обновления
   - Пользовательские взаимодействия
   - Загрузка моделей (через `callAsync`)

3. **Background Thread:**
   - `PythonBridge::run()` - выполнение Python-скрипта
   - Моделирование NAM (если не в real-time)

### 8.2 Synchronization

- **APVTS Parameters:** Атомики для lock-free доступа
- **Model Loading:** `std::mutex` в `NAMProcessor`
- **UI Updates:** `juce::MessageManager::callAsync()`

---

## 9. Memory Management

- **Smart Pointers:** `std::unique_ptr` для владения ресурсами
- **JUCE Memory:** Использование JUCE memory management где возможно
- **Buffer Management:** Предварительное выделение буферов в `prepareToPlay()`
- **Model Loading:** Загрузка моделей в фоновом потоке, swap через mutex

---

## 10. Performance Considerations

### 10.1 Real-time Constraints

- **Latency:** < 10ms при стандартных настройках буфера
- **CPU Usage:** Оптимизировано для real-time обработки
- **Memory:** Предварительное выделение буферов

### 10.2 Optimization Strategies

- Lock-free доступ к параметрам
- Предварительная загрузка моделей
- Эффективное использование SIMD где возможно
- Минимизация аллокаций в audio thread

---

## 11. Расширяемость

### 11.1 Добавление Новых Эффектов

1. Добавить новый параметр в `createParameterLayout()`
2. Добавить обработку в `DSPChain::process()`
3. Добавить UI элемент в `PluginEditor`
4. Обновить `PresetManager` для сохранения/загрузки

### 11.2 Добавление Новых Моделей

- Просто добавьте `.nam` файл в `assets/nam_models/`
- Python оптимизатор автоматически найдет его
- Плагин загрузит через `NAMProcessor::loadModel()`

---

**Архитектура спроектирована для масштабируемости, производительности и удобства поддержки.**

