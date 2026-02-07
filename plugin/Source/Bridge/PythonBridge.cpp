/*
  ==============================================================================
    PythonBridge.cpp
    Launches the Python optimizer as a child process, reads the resulting
    preset JSON, and delivers parsed parameters back to the message thread.
  ==============================================================================
*/

#include "PythonBridge.h"

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
                              Callback onComplete)
{
    if (isThreadRunning())
        return;   // already in progress

    diFilePending   = diFile;
    refFilePending  = refFile;
    callbackPending = std::move(onComplete);

    startThread(juce::Thread::Priority::normal);
}

//==============================================================================
void PythonBridge::run()
{
    DBG("[PythonBridge] Starting run()...");
    
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
    juce::String cmd;
    if (executableType == "exe")
    {
        // Standalone exe - run directly
        cmd = "\"" + script.getFullPathName() + "\""
            + " --di \"" + diFilePending.getFullPathName() + "\""
            + " --ref \"" + refFilePending.getFullPathName() + "\""
            + " --out \"" + outputJson.getFullPathName() + "\"";
    }
    else
    {
        // Python script - need python interpreter
        cmd = pythonPath
            + " \"" + script.getFullPathName() + "\""
            + " --di \"" + diFilePending.getFullPathName() + "\""
            + " --ref \"" + refFilePending.getFullPathName() + "\""
            + " --out \"" + outputJson.getFullPathName() + "\"";
    }

    DBG("[PythonBridge] Launching: " + cmd);

    // Launch child process
    juce::ChildProcess process;
    bool started = process.start(cmd);

    if (! started)
    {
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "Failed to start Python process. Command: " + cmd;

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    // Wait for completion (blocking, but we're on our own thread)
    bool finished = process.waitForProcessToFinish(timeoutMs);
    auto exitCode = process.getExitCode();

    if (threadShouldExit())
        return;

    if (! finished || exitCode != 0)
    {
        juce::String stdErr = process.readAllProcessOutput();
        DBG("[PythonBridge] Process failed. Exit code: " + juce::String(exitCode));
        DBG("[PythonBridge] stderr: " + stdErr);

        MatchResult fail;
        fail.success = false;
        // Don't show full stderr in UI - just a simple message
        fail.errorMessage = finished
            ? "Matching failed. Check logs."
            : "Process timed out.";

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

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
    auto parsed   = juce::JSON::parse(jsonText);

    if (! parsed.isObject())
    {
        result.success = false;
        result.errorMessage = "Failed to parse JSON: " + jsonText.substring(0, 200);
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
    }

    // ── Parse "params" ───────────────────────────────────────────────────────
    if (auto* params = parsed.getProperty("params", {}).getDynamicObject())
    {
        result.inputGainDb    = static_cast<float>((double) params->getProperty("input_gain_db"));
        result.preEqGainDb    = static_cast<float>((double) params->getProperty("pre_eq_gain_db"));
        result.preEqFreqHz    = static_cast<float>((double) params->getProperty("pre_eq_freq_hz"));
        result.reverbWet      = static_cast<float>((double) params->getProperty("reverb_wet"));
        result.reverbRoomSize = static_cast<float>((double) params->getProperty("reverb_room_size"));
        result.delayTimeMs    = static_cast<float>((double) params->getProperty("delay_time_ms"));
        result.delayMix       = static_cast<float>((double) params->getProperty("delay_mix"));
        result.finalEqGainDb  = static_cast<float>((double) params->getProperty("final_eq_gain_db"));
    }

    result.loss    = static_cast<float>((double) parsed.getProperty("loss", 0.0));
    result.success = true;
    return result;
}




