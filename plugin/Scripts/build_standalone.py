#!/usr/bin/env python3
"""
Build standalone executable for ToneMatch AI using PyInstaller.
Run this script to create tone_matcher.exe
"""

import os
import sys
import subprocess
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist", "ToneMatchAI")

def main():
    print("=" * 60)
    print("Building ToneMatch AI Standalone Executable")
    print("=" * 60)
    
    # Check PyInstaller
    try:
        import PyInstaller
        print(f"[OK] PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("[ERROR] PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)
    
    # Source file
    source_file = os.path.join(SCRIPT_DIR, "tone_matcher.py")
    if not os.path.exists(source_file):
        print(f"[ERROR] Source file not found: {source_file}")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "tone_matcher",
        "--distpath", os.path.join(DIST_DIR, "Scripts"),
        "--workpath", os.path.join(PROJECT_ROOT, "build", "pyinstaller"),
        "--specpath", os.path.join(PROJECT_ROOT, "build", "pyinstaller"),
        "--clean",
        "--noconfirm",
        # Hidden imports for audio processing
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.signal",
        "--hidden-import", "scipy.optimize",
        "--hidden-import", "numpy",
        "--hidden-import", "soundfile",
        source_file
    ]
    
    print(f"\n[Building] Running PyInstaller...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"[ERROR] PyInstaller failed with code {result.returncode}")
        sys.exit(1)
    
    print(f"\n[OK] Executable built successfully!")
    print(f"Location: {os.path.join(DIST_DIR, 'Scripts', 'tone_matcher.exe')}")
    
    # Copy assets
    print(f"\n[Copying] Assets...")
    
    assets_src = os.path.join(PROJECT_ROOT, "assets")
    assets_dst = os.path.join(DIST_DIR, "assets")
    
    if os.path.exists(assets_dst):
        shutil.rmtree(assets_dst)
    
    if os.path.exists(assets_src):
        shutil.copytree(assets_src, assets_dst)
        print(f"[OK] Assets copied to {assets_dst}")
    else:
        print(f"[WARNING] Assets not found at {assets_src}")
    
    # Copy VST3
    print(f"\n[Copying] VST3 plugin...")
    
    vst3_src = os.path.join(PROJECT_ROOT, "plugin", "build", "ToneMatchAI_artefacts", "Release", "VST3", "ToneMatch AI.vst3")
    vst3_dst = os.path.join(DIST_DIR, "ToneMatch AI.vst3")
    
    if os.path.exists(vst3_dst):
        shutil.rmtree(vst3_dst)
    
    if os.path.exists(vst3_src):
        shutil.copytree(vst3_src, vst3_dst)
        print(f"[OK] VST3 copied to {vst3_dst}")
    else:
        print(f"[WARNING] VST3 not found at {vst3_src}")
    
    print(f"\n" + "=" * 60)
    print(f"BUILD COMPLETE!")
    print(f"=" * 60)
    print(f"\nDistribution folder: {DIST_DIR}")
    print(f"\nContents:")
    for item in os.listdir(DIST_DIR):
        print(f"  - {item}")


if __name__ == "__main__":
    main()

