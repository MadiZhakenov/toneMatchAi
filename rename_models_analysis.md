# Анализ и схема переименования NAM моделей

## Выявленные паттерны:

### 1. Усилители Peavey
- **5150** - разные варианты с бустами
- **6505+** - Green/Red каналы, с/без бустов
- **6534+** - разные бусты
- **JSX** - Ultra, Crunch режимы
- **XXX** - 6L6/KT77 лампы, разные бусты

### 2. Marshall
- **JCM2000** - Clean, Crunch, Lead (L2) каналы
- **JCM900** - Dual Verb, разные каналы A/B, gain levels

### 3. Bugera
- **333** - Clean, Crunch, Lead
- **6262** - Crunch, Lead
- **1990** - Lead

### 4. Педали
- **DS1** - Boss Distortion
- **TS9** - Tube Screamer
- **OD808** - Maxon Overdrive
- **M77** - MXR Custom Badass
- **HM2** - Boss Heavy Metal
- **Klone** - Klon Centaur clone

### 5. Другие усилители
- Fender TwinVerb
- Vox AC15
- Magnatone Super 59
- Splawn Pro Mod
- Jet City
- И другие...

## Схема упрощения:

### Формат: `[Усилитель/Педаль] [Канал] [Буст]`

**Примеры:**
- `Helga B 5150 BlockLetter - Boosted.nam` → `5150 Boosted.nam`
- `Helga B 5150 BlockLetter - NoBoost.nam` → `5150 Clean.nam`
- `Helga B 6505+ Red ch - MXR Drive.nam` → `6505+ Red M77.nam`
- `Helga B 6505+ Red ch - NoBoost.nam` → `6505+ Red.nam`
- `Tim R JCM2000 L2 G6.nam` → `JCM2000 Lead.nam`
- `Tim R JCM2000 Crunch 805'd.nam` → `JCM2000 Crunch 805.nam`

### Правила:
1. Убрать префиксы авторов
2. Упростить названия усилителей
3. Заменить "NoBoost" на "Clean" или просто убрать
4. Упростить названия педалей (MXR Drive → M77, OD808 → 808)
5. Убрать технические детали (ESR, G4/G6/G8, switch positions)
6. Использовать понятные сокращения каналов (Red, Green, Blue, Clean, Crunch, Lead)

