#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для просмотра и форматирования логов debug.log
"""

import json
import sys
import io
from pathlib import Path
from datetime import datetime

# Исправляем кодировку для Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def format_timestamp(ts):
    """Форматирует timestamp"""
    try:
        dt = datetime.fromtimestamp(ts / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return str(ts)

def format_entry(entry):
    """Форматирует одну запись лога"""
    location = entry.get("location", "unknown")
    message = entry.get("message", "")
    timestamp = entry.get("timestamp", 0)
    data = entry.get("data", {})
    
    result = []
    result.append(f"\n{'='*80}")
    result.append(f"Время: {format_timestamp(timestamp)}")
    result.append(f"Место: {location}")
    result.append(f"Сообщение: {message}")
    
    if data:
        result.append(f"\nДанные:")
        for key, value in data.items():
            if isinstance(value, float):
                if abs(value) < 0.0001 and value != 0:
                    result.append(f"  {key}: {value:.2e}")
                else:
                    result.append(f"  {key}: {value:.6f}")
            elif isinstance(value, bool):
                warning = " [WARNING: ZERO!]" if value and 'zero' in key.lower() else ""
                result.append(f"  {key}: {value}{warning}")
            else:
                result.append(f"  {key}: {value}")
    
    return "\n".join(result)

def main():
    log_file = Path(".cursor/debug.log")
    
    if not log_file.exists():
        print("Файл debug.log не найден!")
        return
    
    # Читаем последние N строк
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return
    
    print(f"Всего строк в файле: {len(lines)}")
    print(f"\nПоказываю последние 20 записей из processor.py:\n")
    
    # Фильтруем записи из processor.py
    processor_entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            location = entry.get("location", "")
            if "processor.py" in location:
                processor_entries.append(entry)
        except:
            pass
    
    if not processor_entries:
        print("Не найдено записей из processor.py")
        print("\nПоказываю последние 10 записей любого типа:\n")
        # Показываем последние записи любого типа
        for line in lines[-10:]:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                print(format_entry(entry))
            except:
                print(f"Не удалось распарсить: {line[:100]}...")
    else:
        print(f"Найдено {len(processor_entries)} записей из processor.py")
        print(f"Показываю последние {min(20, len(processor_entries))} записей:\n")
        
        for entry in processor_entries[-20:]:
            print(format_entry(entry))
    
    # Ищем ключевые этапы
    print(f"\n{'='*80}")
    print("ПОИСК КЛЮЧЕВЫХ ЭТАПОВ:")
    print(f"{'='*80}\n")
    
    key_stages = {
        "DI_START": "Начало обработки - DI трек",
        "BEFORE_INPUT_GAIN": "До input gain",
        "AFTER_INPUT_GAIN": "После input gain",
        "AMP_BEFORE": "До AMP NAM",
        "AMP_AFTER": "После AMP NAM",
        "IR_BEFORE": "До IR convolution",
        "IR_SKIP_ZERO": "[WARNING] ПРОПУСК IR - аудио нулевое!"
    }
    
    found_stages = {}
    for entry in processor_entries:
        location = entry.get("location", "")
        for stage, desc in key_stages.items():
            if stage in location:
                if stage not in found_stages:
                    found_stages[stage] = []
                found_stages[stage].append(entry)
    
    for stage, desc in key_stages.items():
        if stage in found_stages:
            entries = found_stages[stage]
            print(f"✅ {desc}: найдено {len(entries)} записей")
            if entries:
                last = entries[-1]
                data = last.get("data", {})
                if "audio_all_zero" in data or "di_all_zero" in data or "result_all_zero" in data:
                    is_zero = data.get("audio_all_zero") or data.get("di_all_zero") or data.get("result_all_zero", False)
                    if is_zero:
                        print(f"   [WARNING] АУДИО НУЛЕВОЕ!")
                if "audio_rms" in data or "di_rms" in data:
                    rms = data.get("audio_rms") or data.get("di_rms", 0)
                    print(f"   RMS: {rms:.6f}")
        else:
            print(f"❌ {desc}: НЕ НАЙДЕНО")

if __name__ == "__main__":
    main()

