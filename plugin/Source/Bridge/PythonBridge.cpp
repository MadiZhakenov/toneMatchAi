/*
  ==============================================================================
    PythonBridge.cpp
    Launches the Python optimizer as a child process, reads the resulting
    preset JSON, and delivers parsed parameters back to the message thread.
  ==============================================================================
*/

#include "PythonBridge.h"

// #region agent log - debug helper for PythonBridge
#include <fstream>
#include <chrono>
static void dbgLogBridge(const char* loc, const char* msg, float v1 = 0.f, float v2 = 0.f, float v3 = 0.f, int i1 = 0, int i2 = 0) {
    std::ofstream f("e:\\Users\\Desktop\\toneMatchAi\\.cursor\\debug.log", std::ios::app);
    if (!f.is_open()) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    f << "{\"location\":\"" << loc << "\",\"message\":\"" << msg
      << "\",\"hypothesisId\":\"K\",\"data\":{\"v1\":" << v1 << ",\"v2\":" << v2 << ",\"v3\":" << v3
      << ",\"i1\":" << i1 << ",\"i2\":" << i2
      << "},\"timestamp\":" << ms << "}\n";
    f.close();
}
// #endregion

//==============================================================================
PythonBridge::PythonBridge()
    : juce::Thread("ToneMatch-PythonBridge")
{
}

PythonBridge::~PythonBridge()
{
    stopThread(timeoutMs + 5000);
}

//==============================================================================
void PythonBridge::startMatch(const juce::File& diFile,
                              const juce::File& refFile,
                              bool forceHighGain,
                              Callback onComplete)
{
    if (isThreadRunning())
        return;   // already in progress

    diFilePending      = diFile;
    refFilePending     = refFile;
    forceHighGainPending = forceHighGain;
    callbackPending    = std::move(onComplete);

    startThread(juce::Thread::Priority::normal);
}

//==============================================================================
void PythonBridge::run()
{
    DBG("[PythonBridge] Starting run()...");
    
    // CRITICAL: Validate input files BEFORE starting Python
    if (!diFilePending.existsAsFile())
    {
        DBG("[PythonBridge] ERROR: DI file does not exist: " + diFilePending.getFullPathName());
        dbgLogBridge("PythonBridge:DI_NOT_FOUND", ("DI file not found: " + diFilePending.getFullPathName()).toRawUTF8(), 0, 0, 0, 0, 0);
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "DI file not found: " + diFilePending.getFullPathName();
        
        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    if (diFilePending.getSize() == 0)
    {
        DBG("[PythonBridge] ERROR: DI file is empty: " + diFilePending.getFullPathName());
        dbgLogBridge("PythonBridge:DI_EMPTY", ("DI file is empty: " + diFilePending.getFullPathName()).toRawUTF8(), 0, 0, 0, 0, 0);
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "DI file is empty (no audio recorded)";
        
        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    if (!refFilePending.existsAsFile())
    {
        DBG("[PythonBridge] ERROR: Reference file does not exist: " + refFilePending.getFullPathName());
        dbgLogBridge("PythonBridge:REF_NOT_FOUND", ("Reference file not found: " + refFilePending.getFullPathName()).toRawUTF8(), 0, 0, 0, 0, 0);
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "Reference file not found: " + refFilePending.getFullPathName();
        
        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    if (refFilePending.getSize() == 0)
    {
        DBG("[PythonBridge] ERROR: Reference file is empty: " + refFilePending.getFullPathName());
        dbgLogBridge("PythonBridge:REF_EMPTY", ("Reference file is empty: " + refFilePending.getFullPathName()).toRawUTF8(), 0, 0, 0, 0, 0);
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "Reference file is empty";
        
        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    DBG("[PythonBridge] Files validated:");
    DBG("[PythonBridge]   DI: " + diFilePending.getFullPathName() + " (" + juce::String(diFilePending.getSize()) + " bytes)");
    DBG("[PythonBridge]   Ref: " + refFilePending.getFullPathName() + " (" + juce::String(refFilePending.getSize()) + " bytes)");
    
    dbgLogBridge("PythonBridge:VALIDATED", "Files validated", 0, 0, 0, diFilePending.getSize(), refFilePending.getSize());
    
    // Get plugin location - VST3 bundle structure
    auto appFile = juce::File::getSpecialLocation(juce::File::currentApplicationFile);
    auto appDir = appFile.getParentDirectory();
    
    // VST3 structure: ToneMatch AI.vst3/Contents/x86_64-win/ToneMatch AI.vst3
    // We need to go up to ToneMatch AI.vst3 root, then find Scripts folder nearby
    auto vst3Root = appDir.getParentDirectory().getParentDirectory();  // Up from x86_64-win/Contents
    auto pluginFolder = vst3Root.getParentDirectory();  // Folder containing the .vst3
    
    DBG("[PythonBridge] App file: " + appFile.getFullPathName());
    DBG("[PythonBridge] Plugin folder: " + pluginFolder.getFullPathName());
    
    // Try to find tone_matcher.exe (standalone) or run_match.py (development)
    juce::File executable;
    juce::String executableType;
    
    // Paths to try for STANDALONE exe (priority)
    juce::Array<juce::File> exePaths;
    exePaths.add(pluginFolder.getChildFile("Scripts").getChildFile("tone_matcher.exe"));
    exePaths.add(vst3Root.getChildFile("Scripts").getChildFile("tone_matcher.exe"));
    exePaths.add(appDir.getChildFile("Scripts").getChildFile("tone_matcher.exe"));
    
    // Paths to try for DEVELOPMENT script
    juce::Array<juce::File> scriptPaths;
    scriptPaths.add(juce::File("E:/Users/Desktop/toneMatchAi/plugin/Scripts/run_match.py"));
    scriptPaths.add(pluginFolder.getChildFile("Scripts").getChildFile("run_match.py"));
    scriptPaths.add(vst3Root.getChildFile("Scripts").getChildFile("run_match.py"));
    if (scriptPath.existsAsFile())
        scriptPaths.add(scriptPath);
    
    // First try standalone exe
    for (const auto& path : exePaths)
    {
        DBG("[PythonBridge] Checking exe: " + path.getFullPathName());
        if (path.existsAsFile())
        {
            executable = path;
            executableType = "exe";
            DBG("[PythonBridge] Found standalone exe: " + executable.getFullPathName());
            break;
        }
    }
    
    // If no exe, try Python script
    if (! executable.existsAsFile())
    {
        for (const auto& path : scriptPaths)
        {
            DBG("[PythonBridge] Checking script: " + path.getFullPathName());
            if (path.existsAsFile())
            {
                executable = path;
                executableType = "python";
                DBG("[PythonBridge] Found Python script: " + executable.getFullPathName());
                break;
            }
        }
    }

    if (! executable.existsAsFile())
    {
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "Tone matcher not found";
        
        DBG("[PythonBridge] Neither exe nor script found!");

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }
    
    // Use script variable for compatibility with rest of code
    juce::File script = executable;

    // Temporary output file
    juce::File outputJson = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                .getChildFile("tonematch_result.json");
    outputJson.deleteFile();

    // Build command line - different for exe vs python script
    // Use faster settings for plugin: 2.0 sec duration, 30 iterations (vs 5.0/50 in streamlit)
    // This makes it ~2.5x faster while still getting good results
    juce::String durationArg = " --duration 2.0";
    juce::String iterationsArg = " --iterations 30";
    
    juce::String forceHighGainArg = forceHighGainPending ? " --force_high_gain" : "";
    
    juce::String cmd;
    if (executableType == "exe")
    {
        // Standalone exe - run directly
        cmd = "\"" + script.getFullPathName() + "\""
            + " --di \"" + diFilePending.getFullPathName() + "\""
            + " --ref \"" + refFilePending.getFullPathName() + "\""
            + " --out \"" + outputJson.getFullPathName() + "\""
            + durationArg + iterationsArg + forceHighGainArg;
    }
    else
    {
        // Python script - need python interpreter
        cmd = pythonPath
            + " \"" + script.getFullPathName() + "\""
            + " --di \"" + diFilePending.getFullPathName() + "\""
            + " --ref \"" + refFilePending.getFullPathName() + "\""
            + " --out \"" + outputJson.getFullPathName() + "\""
            + durationArg + iterationsArg + forceHighGainArg;
    }

    DBG("[PythonBridge] Launching: " + cmd);
    DBG("[PythonBridge] DI file: " + diFilePending.getFullPathName());
    DBG("[PythonBridge] Ref file: " + refFilePending.getFullPathName());
    DBG("[PythonBridge] Output JSON: " + outputJson.getFullPathName());
    DBG("[PythonBridge] Timeout: " + juce::String(timeoutMs / 1000) + " seconds");
    
    dbgLogBridge("PythonBridge:LAUNCHING", ("Command: " + cmd).toRawUTF8(), 0, 0, 0, 0, 0);

    // Launch child process
    juce::ChildProcess process;
    bool started = process.start(cmd);

    if (! started)
    {
        DBG("[PythonBridge] ERROR: Failed to start process!");
        dbgLogBridge("PythonBridge:START_FAILED", ("Failed to start: " + cmd).toRawUTF8(), 0, 0, 0, 0, 0);
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "Failed to start Python process. Command: " + cmd;

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    DBG("[PythonBridge] Process started successfully, waiting for completion...");
    dbgLogBridge("PythonBridge:STARTED", "Process started", 0, 0, 0, 0, 0);

    // Continuously read output to prevent pipe buffer from filling up (which causes hangs)
    juce::String processOutput;
    const int sleepInterval = 50;
    int elapsedTime = 0;
    bool finished = false;
    
    // Log file for Python output
    juce::File logFile = juce::File("e:\\Users\\Desktop\\toneMatchAi\\.cursor\\python_output.log");
    logFile.deleteFile(); // Clear previous log
    juce::FileOutputStream logStream(logFile);
    if (!logStream.openedOk())
    {
        DBG("[PythonBridge] WARNING: Could not open log file: " + logFile.getFullPathName());
    }

    while (process.isRunning())
    {
        // Read available output
        juce::String newOutput = process.readAllProcessOutput();
        if (newOutput.isNotEmpty())
        {
            processOutput += newOutput;
            // Log to file immediately
            if (logStream.openedOk())
            {
                logStream << newOutput;
                logStream.flush();
            }
        }

        if (threadShouldExit())
        {
            DBG("[PythonBridge] Thread should exit, killing process...");
            process.kill();
            return;
        }

        if (elapsedTime >= timeoutMs)
        {
            DBG("[PythonBridge] Timeout reached, killing process...");
            process.kill();
            finished = false;
            break;
        }

        juce::Thread::sleep(sleepInterval);
        elapsedTime += sleepInterval;
    }

    // One final read after process finishes to ensure we got everything
    juce::String finalOutput = process.readAllProcessOutput();
    if (finalOutput.isNotEmpty())
    {
        processOutput += finalOutput;
        if (logStream.openedOk())
        {
            logStream << finalOutput;
            logStream.flush();
        }
    }
    logStream.flush();
    
    finished = !process.isRunning();
    juce::uint32 exitCode = process.getExitCode();

    DBG("[PythonBridge] Process finished. finished=" + juce::String(finished ? 1 : 0) + 
        ", exitCode=" + juce::String(exitCode));
    
    dbgLogBridge("PythonBridge:FINISHED", ("Process finished, exitCode=" + juce::String(exitCode)).toRawUTF8(), 0, 0, 0, finished ? 1 : 0, exitCode);

    if (! processOutput.isEmpty())
    {
        DBG("[PythonBridge] Process output length: " + juce::String(processOutput.length()) + " chars");
        // Log first 500 chars to debug log
        juce::String outputPreview = processOutput.substring(0, juce::jmin(500, processOutput.length()));
        dbgLogBridge("PythonBridge:OUTPUT", outputPreview.toRawUTF8(), 0, 0, 0, processOutput.length(), 0);
    }
    else
    {
        dbgLogBridge("PythonBridge:NO_OUTPUT", "No output from Python process", 0, 0, 0, 0, 0);
    }

    // Even if process failed, check if JSON was created (might contain error info)
    if (outputJson.existsAsFile())
    {
        DBG("[PythonBridge] JSON file exists, attempting to parse...");
        dbgLogBridge("PythonBridge:JSON_EXISTS", ("JSON file found: " + juce::String(outputJson.getSize()) + " bytes").toRawUTF8(), 0, 0, 0, outputJson.getSize(), 0);
        MatchResult result = parseResultJson(outputJson);
        
        // If JSON parsing succeeded but process failed, check for error field
        if (result.success || (!finished || exitCode != 0))
        {
            // If JSON has error field, use it
            if (!result.success && !result.errorMessage.isEmpty())
            {
                DBG("[PythonBridge] JSON contains error: " + result.errorMessage);
                auto cb = callbackPending;
                juce::MessageManager::callAsync([cb, result]() { if (cb) cb(result); });
                return;
            }
        }
        
        // If JSON parsing succeeded, use it even if process had non-zero exit
        if (result.success)
        {
            DBG("[PythonBridge] JSON parsed successfully despite process exit code: " + juce::String(exitCode));
            auto cb = callbackPending;
            juce::MessageManager::callAsync([cb, result]() { if (cb) cb(result); });
            return;
        }
    }

    // If we get here, process failed and no valid JSON
    if (! finished || exitCode != 0)
    {
        juce::String errorMsg = "Process failed. finished=" + juce::String(finished ? 1 : 0) + 
                                ", exitCode=" + juce::String(exitCode);
        DBG("[PythonBridge] " + errorMsg);
        DBG("[PythonBridge] Process failed. Exit code: " + juce::String(exitCode));
        DBG("[PythonBridge] Output: " + processOutput);
        dbgLogBridge("PythonBridge:PROCESS_FAILED", errorMsg.toRawUTF8(), 0, 0, 0, finished ? 1 : 0, exitCode);
        
        if (!processOutput.isEmpty())
        {
            // Try to extract error message from output
            juce::String errorOutput = processOutput.substring(0, juce::jmin(200, processOutput.length()));
            dbgLogBridge("PythonBridge:ERROR_OUTPUT", errorOutput.toRawUTF8(), 0, 0, 0, processOutput.length(), 0);
        }

        MatchResult fail;
        fail.success = false;
        // Show more detailed error message
        if (! finished)
        {
            fail.errorMessage = "Process timed out after " + juce::String(timeoutMs / 1000) + " seconds.";
        }
        else if (exitCode != 0)
        {
            // Try to extract error message from output
            if (processOutput.contains("ERROR"))
            {
                int errorStart = processOutput.indexOfIgnoreCase("[ERROR]");
                if (errorStart >= 0)
                {
                    int errorEnd = processOutput.indexOfAnyOf("\r\n", errorStart);
                    if (errorEnd > errorStart)
                        fail.errorMessage = processOutput.substring(errorStart, errorEnd);
                    else
                        fail.errorMessage = processOutput.substring(errorStart);
                }
                else
                {
                    fail.errorMessage = "Python process failed. Exit code: " + juce::String(exitCode);
                }
            }
            else
            {
                fail.errorMessage = "Python process failed. Exit code: " + juce::String(exitCode);
            }
        }
        else
        {
            fail.errorMessage = "Matching failed. Check logs.";
        }

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    DBG("[PythonBridge] Process completed successfully, parsing JSON...");

    // Parse the output JSON
    MatchResult result = parseResultJson(outputJson);

    // Deliver on the message thread
    auto cb = callbackPending;
    juce::MessageManager::callAsync([cb, result]() { if (cb) cb(result); });
}

//==============================================================================
MatchResult PythonBridge::parseResultJson(const juce::File& jsonFile)
{
    MatchResult result;

    if (! jsonFile.existsAsFile())
    {
        result.success = false;
        result.errorMessage = "Output JSON not found: " + jsonFile.getFullPathName();
        return result;
    }

    auto jsonText = jsonFile.loadFileAsString();
    
    // #region agent log - log raw JSON
    dbgLogBridge("parseResultJson:RAW_JSON", jsonText.substring(0, 200).toRawUTF8(), 0, 0, 0, jsonText.length(), 0);
    // #endregion
    
    auto parsed   = juce::JSON::parse(jsonText);

    if (! parsed.isObject())
    {
        result.success = false;
        result.errorMessage = "Failed to parse JSON: " + jsonText.substring(0, 200);
        // #region agent log
        dbgLogBridge("parseResultJson:PARSE_FAILED", "JSON parse failed", 0, 0, 0, 0, 0);
        // #endregion
        return result;
    }

    // Check for error field first
    if (parsed.hasProperty("error"))
    {
        result.success = false;
        result.errorMessage = parsed.getProperty("error", "Unknown error").toString();
        DBG("[PythonBridge] JSON contains error: " + result.errorMessage);
        return result;
    }

    // ── Parse "rig" ──────────────────────────────────────────────────────────
    if (auto* rig = parsed.getProperty("rig", {}).getDynamicObject())
    {
        result.fxNamName  = rig->getProperty("fx_nam").toString();
        result.ampNamName = rig->getProperty("amp_nam").toString();
        result.irName     = rig->getProperty("ir").toString();
        result.fxNamPath  = rig->getProperty("fx_nam_path").toString();
        result.ampNamPath = rig->getProperty("amp_nam_path").toString();
        result.irPath     = rig->getProperty("ir_path").toString();
        
        // #region agent log
        dbgLogBridge("parseResultJson:RIG", "parsed rig from JSON", 0, 0, 0, 
                     result.fxNamPath.length(), result.ampNamPath.length());
        // #endregion
    }
    else
    {
        // #region agent log
        dbgLogBridge("parseResultJson:NO_RIG", "rig object not found", 0, 0, 0, 0, 0);
        // #endregion
    }

    // ── Parse "params" ───────────────────────────────────────────────────────
    if (auto* params = parsed.getProperty("params", {}).getDynamicObject())
    {
        result.inputGainDb    = static_cast<float>((double) params->getProperty("input_gain_db"));
        
        // Parse overdrive_db if present, otherwise use input_gain_db as fallback
        // This allows explicit overdrive_db in JSON while maintaining backward compatibility
        if (params->hasProperty("overdrive_db"))
        {
            result.overdriveDb = static_cast<float>((double) params->getProperty("overdrive_db"));
        }
        else
        {
            // Fallback: use input_gain_db as overdrive_db (backward compatibility)
            result.overdriveDb = result.inputGainDb;
        }
        
        result.preEqGainDb    = static_cast<float>((double) params->getProperty("pre_eq_gain_db"));
        result.preEqFreqHz    = static_cast<float>((double) params->getProperty("pre_eq_freq_hz"));
        result.reverbWet      = static_cast<float>((double) params->getProperty("reverb_wet"));
        result.reverbRoomSize = static_cast<float>((double) params->getProperty("reverb_room_size"));
        result.delayTimeMs    = static_cast<float>((double) params->getProperty("delay_time_ms"));
        result.delayMix       = static_cast<float>((double) params->getProperty("delay_mix"));
        result.finalEqGainDb  = static_cast<float>((double) params->getProperty("final_eq_gain_db"));
        
        // #region agent log
        dbgLogBridge("parseResultJson:PARAMS", "parsed params from JSON", 
                     result.inputGainDb, result.preEqGainDb, result.preEqFreqHz,
                     (int)(result.reverbWet * 100), (int)(result.delayMix * 100));
        DBG("[PythonBridge] Parsed overdriveDb: " + juce::String(result.overdriveDb, 2) + 
            " dB (from JSON field: " + (params->hasProperty("overdrive_db") ? "overdrive_db" : "input_gain_db") + ")");
        // #endregion
    }
    else
    {
        // #region agent log
        dbgLogBridge("parseResultJson:NO_PARAMS", "params object not found", 0, 0, 0, 0, 0);
        // #endregion
    }

    result.loss    = static_cast<float>((double) parsed.getProperty("loss", 0.0));
    result.success = true;
    
    // #region agent log
    dbgLogBridge("parseResultJson:SUCCESS", "parse complete", 
                 result.loss, 0, 0, 
                 result.fxNamPath.length(), result.ampNamPath.length());
    // #endregion
    
    return result;
}




