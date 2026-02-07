# Pre-Compilation Verification Script for ToneMatch AI VST
# This script verifies all prerequisites before compilation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ToneMatch AI - Pre-Compilation Check" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$errors = @()
$warnings = @()

# 1. File Structure Check
Write-Host "[1.1] Checking file structure..." -ForegroundColor Yellow

if (Test-Path "plugin/Scripts/run_match.py") {
    Write-Host "  ✓ plugin/Scripts/run_match.py exists" -ForegroundColor Green
} else {
    $errors += "plugin/Scripts/run_match.py not found"
    Write-Host "  ✗ plugin/Scripts/run_match.py NOT FOUND" -ForegroundColor Red
}

$namCount = (Get-ChildItem -Path "assets/nam_models" -Filter "*.nam" -ErrorAction SilentlyContinue).Count
if ($namCount -ge 250) {
    Write-Host "  ✓ assets/nam_models/ contains $namCount .nam files" -ForegroundColor Green
} else {
    $warnings += "Expected ~259 NAM models, found $namCount"
    Write-Host "  ⚠ assets/nam_models/ contains only $namCount .nam files (expected ~259)" -ForegroundColor Yellow
}

if (Test-Path "assets/impulse_responses") {
    $irCount = (Get-ChildItem -Path "assets/impulse_responses" -Filter "*.wav" -ErrorAction SilentlyContinue).Count
    Write-Host "  ✓ assets/impulse_responses/ contains $irCount IR files" -ForegroundColor Green
} else {
    $warnings += "assets/impulse_responses/ directory not found"
    Write-Host "  ⚠ assets/impulse_responses/ directory not found" -ForegroundColor Yellow
}

if (Test-Path "src/core/optimizer.py") {
    Write-Host "  ✓ src/core/optimizer.py exists" -ForegroundColor Green
} else {
    $errors += "src/core/optimizer.py not found"
    Write-Host "  ✗ src/core/optimizer.py NOT FOUND" -ForegroundColor Red
}

if (Test-Path "plugin/CMakeLists.txt") {
    $cmakeSize = (Get-Item "plugin/CMakeLists.txt").Length
    if ($cmakeSize -gt 0) {
        Write-Host "  ✓ plugin/CMakeLists.txt exists and has content" -ForegroundColor Green
    } else {
        $warnings += "plugin/CMakeLists.txt is empty"
        Write-Host "  ⚠ plugin/CMakeLists.txt exists but is empty" -ForegroundColor Yellow
    }
} else {
    $warnings += "plugin/CMakeLists.txt not found (may use JUCE Projucer instead)"
    Write-Host "  ⚠ plugin/CMakeLists.txt not found" -ForegroundColor Yellow
}

# 2. Python Environment Check
Write-Host ""
Write-Host "[1.2] Checking Python environment..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.([0-9]+)") {
        $minorVersion = [int]$matches[1]
        if ($minorVersion -ge 9) {
            Write-Host "  ✓ Python version: $pythonVersion" -ForegroundColor Green
        } else {
            $errors += "Python 3.9+ required, found $pythonVersion"
            Write-Host "  ✗ Python version too old: $pythonVersion (need 3.9+)" -ForegroundColor Red
        }
    } else {
        $errors += "Could not parse Python version"
        Write-Host "  ✗ Could not parse Python version" -ForegroundColor Red
    }
} catch {
    $errors += "Python not found in PATH"
    Write-Host "  ✗ Python not found in PATH" -ForegroundColor Red
}

# Check Python dependencies
Write-Host "  Checking Python dependencies..." -ForegroundColor Gray
$deps = @("torch", "librosa", "soundfile", "numpy", "scipy", "pedalboard", "auraloss")
foreach ($dep in $deps) {
    try {
        python -c "import $dep" 2>&1 | Out-Null
        Write-Host "    ✓ $dep" -ForegroundColor Green
    } catch {
        $errors += "Python dependency missing: $dep"
        Write-Host "    ✗ $dep NOT FOUND" -ForegroundColor Red
    }
}

# Test run_match.py
Write-Host "  Testing run_match.py..." -ForegroundColor Gray
try {
    $helpOutput = python plugin/Scripts/run_match.py --help 2>&1
    if ($helpOutput -match "usage:") {
        Write-Host "    ✓ run_match.py is executable" -ForegroundColor Green
    } else {
        $warnings += "run_match.py help output unexpected"
        Write-Host "    ⚠ run_match.py help output unexpected" -ForegroundColor Yellow
    }
} catch {
    $errors += "run_match.py failed to execute"
    Write-Host "    ✗ run_match.py failed to execute" -ForegroundColor Red
}

# Test Python module import
Write-Host "  Testing Python module imports..." -ForegroundColor Gray
try {
    python -c "import sys; sys.path.insert(0, '.'); from src.core.optimizer import ToneOptimizer; print('OK')" 2>&1 | Out-Null
    Write-Host "    ✓ src.core.optimizer imports successfully" -ForegroundColor Green
} catch {
    $errors += "Failed to import src.core.optimizer"
    Write-Host "    ✗ Failed to import src.core.optimizer" -ForegroundColor Red
}

# 3. Build Dependencies Check
Write-Host ""
Write-Host "[1.3] Checking build dependencies..." -ForegroundColor Yellow

# Check CMake
try {
    $cmakeVersion = cmake --version 2>&1 | Select-Object -First 1
    if ($cmakeVersion -match "version ([0-9]+\.[0-9]+)") {
        $version = [version]$matches[1]
        if ($version -ge [version]"3.20") {
            Write-Host "  ✓ CMake version: $version" -ForegroundColor Green
        } else {
            $warnings += "CMake 3.20+ recommended, found $version"
            Write-Host "  ⚠ CMake version: $version (3.20+ recommended)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⚠ CMake found but version unclear" -ForegroundColor Yellow
    }
} catch {
    $warnings += "CMake not found in PATH (may use Visual Studio/Xcode directly)"
    Write-Host "  ⚠ CMake not found in PATH" -ForegroundColor Yellow
}

# Check NeuralAmpModelerCore
if (Test-Path "plugin/ThirdParty/NeuralAmpModelerCore/CMakeLists.txt") {
    Write-Host "  ✓ NeuralAmpModelerCore found" -ForegroundColor Green
} else {
    $errors += "NeuralAmpModelerCore not found"
    Write-Host "  ✗ NeuralAmpModelerCore NOT FOUND" -ForegroundColor Red
}

# Check Eigen
if (Test-Path "plugin/ThirdParty/NeuralAmpModelerCore/Dependencies/eigen") {
    Write-Host "  ✓ Eigen library found" -ForegroundColor Green
} else {
    $errors += "Eigen library not found"
    Write-Host "  ✗ Eigen library NOT FOUND" -ForegroundColor Red
}

# Check nlohmann/json
if (Test-Path "plugin/ThirdParty/NeuralAmpModelerCore/Dependencies/nlohmann/json.hpp") {
    Write-Host "  ✓ nlohmann/json found" -ForegroundColor Green
} else {
    $errors += "nlohmann/json not found"
    Write-Host "  ✗ nlohmann/json NOT FOUND" -ForegroundColor Red
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($errors.Count -eq 0) {
    Write-Host "✓ All critical checks passed!" -ForegroundColor Green
    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "Warnings ($($warnings.Count)):" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  ⚠ $warning" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "Ready to proceed with compilation." -ForegroundColor Green
    exit 0
} else {
    Write-Host "✗ Found $($errors.Count) error(s):" -ForegroundColor Red
    foreach ($error in $errors) {
        Write-Host "  ✗ $error" -ForegroundColor Red
    }
    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "Warnings ($($warnings.Count)):" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  ⚠ $warning" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host "Please fix errors before proceeding." -ForegroundColor Red
    exit 1
}

