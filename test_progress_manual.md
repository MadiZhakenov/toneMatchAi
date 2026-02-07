# Тестирование асинхронного обновления статуса

## Способ 1: Через логи в плагине (рекомендуется)

### Шаги:

1. **Скомпилируйте плагин** с добавленным логированием
2. **Запустите DAW** (FL Studio, Reaper, и т.д.)
3. **Загрузите плагин** ToneMatch AI
4. **Откройте консоль/логи** вашей DAW или системы

### Что делать:

1. Нажмите кнопку **"MATCH TONE!"**
2. Выберите референсный аудиофайл
3. **Наблюдайте логи** в консоли:

```
[Progress] Stage: 1, Status: Grid Search..., Progress: 30.0%
[Editor] ValueTree changed: progressStage -> Stage: 1, Status: Grid Search..., Progress: 30.0%
[Editor] ValueTree changed: statusText -> Stage: 1, Status: Grid Search..., Progress: 30.0%
[Editor] ValueTree changed: progress -> Stage: 1, Status: Grid Search..., Progress: 30.0%
```

4. Дождитесь завершения процесса и проверьте, что все этапы отображаются:
   - Stage 1: Grid Search... (30%)
   - Stage 2: Optimizing... (70%)
   - Stage 3: Done (100%)

### Ожидаемый результат:

- Логи показывают, что `setProgressStage()` вызывается в Processor
- Логи показывают, что `valueTreePropertyChanged()` вызывается в Editor
- UI обновляется синхронно (кнопка скрывается, показывается прогресс бар)
- Статус текст обновляется в реальном времени

---

## Способ 2: Standalone тест (не требует DAW)

### Компиляция теста:

```bash
cd E:\Users\Desktop\toneMatchAi
# Убедитесь, что JUCE доступен
# Скомпилируйте test_progress_updates.cpp с JUCE
```

### Запуск:

```bash
./test_progress_updates
```

### Ожидаемый вывод:

```
========================================
ValueTree Progress Update Test
========================================

Simulating tone matching process...

1. Starting match...
[Processor] Updated: Stage=1, Status=Grid Search..., Progress=30%
[Listener] Property changed: progressStage -> Stage=1, Status=Grid Search..., Progress=30%
[Listener] Property changed: statusText -> Stage=1, Status=Grid Search..., Progress=30%
[Listener] Property changed: progress -> Stage=1, Status=Grid Search..., Progress=30%

2. Optimizing parameters...
[Processor] Updated: Stage=2, Status=Optimizing..., Progress=70%
[Listener] Property changed: progressStage -> Stage=2, Status=Optimizing..., Progress=70%
...

========================================
Test complete!
========================================
```

---

## Способ 3: Визуальная проверка в UI

### Шаги:

1. **Запустите плагин** в DAW
2. **Нажмите "MATCH TONE!"**
3. **Наблюдайте UI изменения:**

   - ✅ Кнопка "MATCH TONE" должна **исчезнуть**
   - ✅ Появится **Progress Bar** (синяя полоса)
   - ✅ Появится **Status Label** с текстом "Grid Search..."
   - ✅ Progress Bar должен заполняться до ~30%
   - ✅ Статус меняется на "Optimizing..." (70%)
   - ✅ Статус меняется на "Done" (100%)
   - ✅ Кнопка возвращается, прогресс бар скрывается

### Проверка асинхронности:

- UI обновляется **мгновенно** при изменении прогресса
- Нет задержек или "зависаний"
- Все обновления происходят на **message thread** (без блокировок)

---

## Способ 4: Добавить тестовую кнопку (для разработки)

Можно временно добавить кнопку в Editor для симуляции обновлений:

```cpp
// В setupSectionA() добавить:
testButton.onClick = [this]() {
    processorRef.setProgressStage(1, "Test: Grid Search...");
    juce::MessageManager::callAsync([this]() {
        processorRef.setProgressStage(2, "Test: Optimizing...");
        juce::MessageManager::callAsync([this]() {
            processorRef.setProgressStage(3, "Test: Done");
        });
    });
};
```

Это позволит тестировать UI без реального Python процесса.

---

## Что проверить:

- [ ] ValueTree обновления происходят синхронно
- [ ] Editor получает уведомления через `valueTreePropertyChanged()`
- [ ] UI обновляется без задержек
- [ ] Progress Bar корректно отображает прогресс (0-100%)
- [ ] Status Label показывает правильный текст
- [ ] Кнопка скрывается/показывается в нужные моменты
- [ ] Нет race conditions или блокировок

---

## Отладка проблем:

Если обновления не работают:

1. Проверьте, что `progressState.addListener(this)` вызван в конструкторе Editor
2. Проверьте, что `progressState.removeListener(this)` НЕ вызывается до завершения
3. Убедитесь, что обновления происходят на **message thread**
4. Проверьте логи на наличие ошибок

