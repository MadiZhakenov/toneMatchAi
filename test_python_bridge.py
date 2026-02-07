#!/usr/bin/env python3
"""
Test script to verify Python bridge (run_match.py) works standalone.
This simulates what the C++ plugin will do when calling the Python optimizer.
"""

import os
import sys
import json
import tempfile
import subprocess

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def test_run_match_script():
    """Test that run_match.py can be executed and produces valid JSON."""
    
    print("=" * 70)
    print("Testing Python Bridge (run_match.py)")
    print("=" * 70)
    
    # Check if script exists
    script_path = os.path.join(SCRIPT_DIR, "plugin", "Scripts", "run_match.py")
    if not os.path.exists(script_path):
        print(f"❌ ERROR: run_match.py not found at {script_path}")
        return False
    
    print(f"✓ Found script: {script_path}")
    
    # Check if we can import the optimizer
    try:
        from src.core.optimizer import ToneOptimizer
        print("✓ ToneOptimizer can be imported")
    except ImportError as e:
        print(f"❌ ERROR: Cannot import ToneOptimizer: {e}")
        return False
    
    # Check if test audio files exist (optional - skip if not available)
    test_di = os.path.join(SCRIPT_DIR, "assets", "demo_di.wav")
    test_ref = os.path.join(SCRIPT_DIR, "assets", "demo_ref.wav")
    
    if not os.path.exists(test_di) or not os.path.exists(test_ref):
        print("⚠ WARNING: Test audio files not found. Skipping full execution test.")
        print("  To test fully, ensure assets/demo_di.wav and assets/demo_ref.wav exist")
        return True  # Not a failure, just can't test execution
    
    # Test script execution
    print("\nTesting script execution...")
    output_json = os.path.join(tempfile.gettempdir(), "test_tonematch_result.json")
    
    try:
        cmd = [
            sys.executable,
            script_path,
            "--di", test_di,
            "--ref", test_ref,
            "--out", output_json,
            "--duration", "2.0",  # Short duration for testing
            "--iterations", "10"   # Few iterations for testing
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"❌ ERROR: Script failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("✓ Script executed successfully")
        
        # Verify JSON output
        if not os.path.exists(output_json):
            print(f"❌ ERROR: Output JSON not created at {output_json}")
            return False
        
        print(f"✓ Output JSON created: {output_json}")
        
        # Parse and validate JSON
        with open(output_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = {
            'rig': ['fx_nam', 'amp_nam', 'ir', 'fx_nam_path', 'amp_nam_path', 'ir_path'],
            'params': ['input_gain_db', 'pre_eq_gain_db', 'pre_eq_freq_hz', 
                      'reverb_wet', 'reverb_room_size', 'delay_time_ms', 
                      'delay_mix', 'final_eq_gain_db'],
            'loss': None
        }
        
        for section, fields in required_fields.items():
            if section not in data:
                print(f"❌ ERROR: Missing section '{section}' in JSON")
                return False
            
            if fields:
                for field in fields:
                    if field not in data[section]:
                        print(f"❌ ERROR: Missing field '{section}.{field}' in JSON")
                        return False
        
        print("✓ JSON structure is valid")
        print(f"  Loss: {data.get('loss', 'N/A')}")
        print(f"  FX NAM: {data['rig'].get('fx_nam', 'N/A')}")
        print(f"  AMP NAM: {data['rig'].get('amp_nam', 'N/A')}")
        print(f"  IR: {data['rig'].get('ir', 'N/A')}")
        
        # Clean up
        if os.path.exists(output_json):
            os.remove(output_json)
        
        print("\n" + "=" * 70)
        print("✅ Python Bridge Test PASSED")
        print("=" * 70)
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Script execution timed out")
        return False
    except Exception as e:
        print(f"❌ ERROR: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_run_match_script()
    sys.exit(0 if success else 1)

