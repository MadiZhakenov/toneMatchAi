#!/usr/bin/env python3
"""
Скрипт для анализа логов обработки аудио.
Читает debug.log и python_output.log, анализирует этапы обработки.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def parse_debug_log(log_path: str) -> List[Dict[str, Any]]:
    """Парсит JSON строки из debug.log"""
    entries = []
    if not os.path.exists(log_path):
        return entries
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry["_line_num"] = line_num
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return entries

def parse_python_output(log_path: str) -> List[str]:
    """Читает строки из python_output.log"""
    lines = []
    if not os.path.exists(log_path):
        return lines
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
    
    return lines

def format_timestamp(ts: int) -> str:
    """Форматирует timestamp в читаемый формат"""
    try:
        dt = datetime.fromtimestamp(ts / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except:
        return str(ts)

def analyze_processing_chain(entries: List[Dict[str, Any]]) -> None:
    """Анализирует цепочку обработки из логов"""
    print("\n" + "="*80)
    print("АНАЛИЗ ЦЕПОЧКИ ОБРАБОТКИ АУДИО")
    print("="*80)
    
    # Группируем по этапам
    stages = {
        "DI_START": "Начало обработки - валидация DI трека",
        "BEFORE_INPUT_GAIN": "До применения input gain",
        "AFTER_INPUT_GAIN": "После применения input gain",
        "AMP_BEFORE": "До AMP NAM обработки",
        "AMP_AFTER": "После AMP NAM обработки",
        "IR_BEFORE": "До IR convolution",
        "IR_SKIP_ZERO": "ПРОПУСК IR - аудио нулевое!",
        "IR_FILE_NOT_FOUND": "IR файл не найден",
    }
    
    for entry in entries:
        location = entry.get("location", "")
        if ":" in location:
            stage = location.split(":")[-1]
        else:
            stage = location
        
        if stage in stages:
            print(f"\n[{format_timestamp(entry.get('timestamp', 0))}] {stages[stage]}")
            print(f"  Location: {location}")
            data = entry.get("data", {})
            
            # Выводим ключевые метрики
            if "di_len" in data or "audio_len" in data:
                length = data.get("di_len") or data.get("audio_len", 0)
                print(f"  Длина аудио: {length}")
            
            if "di_rms" in data or "audio_rms" in data:
                rms = data.get("di_rms") or data.get("audio_rms", 0)
                print(f"  RMS: {rms:.6f}")
            
            if "di_all_zero" in data or "audio_all_zero" in data or "result_all_zero" in data:
                is_zero = data.get("di_all_zero") or data.get("audio_all_zero") or data.get("result_all_zero", False)
                print(f"  Все нули: {is_zero}")
                if is_zero:
                    print(f"  ⚠️  ВНИМАНИЕ: Аудио полностью нулевое!")
            
            if "audio_min" in data and "audio_max" in data:
                print(f"  Min: {data.get('audio_min', 0):.6f}, Max: {data.get('audio_max', 0):.6f}")
            
            if "input_gain_db" in data:
                print(f"  Input gain: {data.get('input_gain_db', 0):.2f} dB")
            
            if "amp_nam_path" in data:
                amp_path = data.get("amp_nam_path", "None")
                amp_exists = data.get("amp_exists", False)
                print(f"  AMP NAM: {amp_path}")
                print(f"  AMP существует: {amp_exists}")
            
            if "ir_path" in data:
                ir_path = data.get("ir_path", "None")
                ir_exists = data.get("ir_exists", False)
                print(f"  IR: {ir_path}")
                print(f"  IR существует: {ir_exists}")

def analyze_python_output(lines: List[str]) -> None:
    """Анализирует вывод Python процесса"""
    print("\n" + "="*80)
    print("ВЫВОД PYTHON ПРОЦЕССА (stderr/stdout)")
    print("="*80)
    
    if not lines:
        print("  Файл python_output.log не найден или пуст.")
        print("  Это означает, что процесс еще не запускался или завершился с ошибкой до записи вывода.")
        return
    
    # Ищем ключевые сообщения
    error_keywords = ["ERROR", "WARNING", "PROCESSOR ERROR", "PROCESSOR WARNING"]
    important_keywords = ["START", "AMP", "IR", "gain", "zero", "empty", "invalid"]
    
    error_lines = []
    important_lines = []
    
    for line in lines:
        line_upper = line.upper()
        if any(kw in line_upper for kw in error_keywords):
            error_lines.append(line.rstrip())
        elif any(kw in line_upper for kw in important_keywords):
            important_lines.append(line.rstrip())
    
    if error_lines:
        print("\n⚠️  ОШИБКИ И ПРЕДУПРЕЖДЕНИЯ:")
        for line in error_lines:
            print(f"  {line}")
    
    if important_lines:
        print("\n📋 ВАЖНЫЕ СООБЩЕНИЯ:")
        for line in important_lines[:50]:  # Ограничиваем вывод
            print(f"  {line}")
    
    if not error_lines and not important_lines:
        print("\n  Ключевые сообщения не найдены. Полный вывод:")
        for line in lines[:100]:  # Первые 100 строк
            print(f"  {line.rstrip()}")

def main():
    base_dir = Path(__file__).parent
    debug_log = base_dir / "debug.log"
    python_output_log = base_dir / "python_output.log"
    
    print("="*80)
    print("АНАЛИЗ ЛОГОВ TONEMATCH AI")
    print("="*80)
    print(f"\nПроверяю файлы:")
    print(f"  debug.log: {debug_log.exists() and debug_log.stat().st_size > 0}")
    print(f"  python_output.log: {python_output_log.exists() and python_output_log.stat().st_size > 0}")
    
    # Парсим debug.log
    entries = parse_debug_log(str(debug_log))
    print(f"\nНайдено записей в debug.log: {len(entries)}")
    
    if entries:
        analyze_processing_chain(entries)
    else:
        print("\n⚠️  debug.log пуст или не содержит валидных записей.")
        print("   Это может означать:")
        print("   - Функция обработки еще не вызывалась")
        print("   - Ошибка при записи логов (проверьте права доступа)")
        print("   - Логирование отключено")
    
    # Парсим python_output.log
    lines = parse_python_output(str(python_output_log))
    print(f"\nНайдено строк в python_output.log: {len(lines)}")
    
    if lines:
        analyze_python_output(lines)
    
    # Итоговая диагностика
    print("\n" + "="*80)
    print("ДИАГНОСТИКА")
    print("="*80)
    
    if not entries and not lines:
        print("\n❌ Логи отсутствуют. Возможные причины:")
        print("   1. Плагин еще не запускался")
        print("   2. Ошибка при запуске Python процесса")
        print("   3. Логирование не настроено")
    elif entries:
        # Проверяем последнюю запись
        last_entry = entries[-1]
        location = last_entry.get("location", "")
        
        if "IR_SKIP_ZERO" in location:
            print("\n❌ ПРОБЛЕМА ОБНАРУЖЕНА:")
            print("   Аудио стало нулевым до IR convolution!")
            data = last_entry.get("data", {})
            print(f"   Этап: {location}")
            print(f"   Длина: {data.get('audio_len', 0)}")
            print(f"   RMS: {data.get('audio_rms', 0):.6f}")
            print("\n   Возможные причины:")
            print("   - AMP NAM модель вернула нули")
            print("   - Input gain слишком низкий")
            print("   - DI трек изначально был нулевым")
        elif "AMP_AFTER" in location:
            data = last_entry.get("data", {})
            if data.get("result_all_zero", False):
                print("\n❌ ПРОБЛЕМА ОБНАРУЖЕНА:")
                print("   AMP NAM вернул нулевой сигнал!")
            else:
                print("\n✅ AMP NAM обработал аудио успешно")
        else:
            print(f"\n📊 Последняя запись: {location}")

if __name__ == "__main__":
    main()

