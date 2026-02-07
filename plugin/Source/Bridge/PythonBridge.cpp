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
    // Resolve script path — default: Scripts/run_match.py next to the binary
    juce::File script = scriptPath;
    if (! script.existsAsFile())
    {
        script = juce::File::getSpecialLocation(juce::File::currentApplicationFile)
                     .getParentDirectory()
                     .getChildFile("Scripts")
                     .getChildFile("run_match.py");
    }

    if (! script.existsAsFile())
    {
        MatchResult fail;
        fail.success = false;
        fail.errorMessage = "run_match.py not found at: " + script.getFullPathName();

        auto cb = callbackPending;
        juce::MessageManager::callAsync([cb, fail]() { if (cb) cb(fail); });
        return;
    }

    // Temporary output file
    juce::File outputJson = juce::File::getSpecialLocation(juce::File::tempDirectory)
                                .getChildFile("tonematch_result.json");
    outputJson.deleteFile();

    // Build command line
    juce::String cmd = pythonPath
        + " \"" + script.getFullPathName() + "\""
        + " --di \"" + diFilePending.getFullPathName() + "\""
        + " --ref \"" + refFilePending.getFullPathName() + "\""
        + " --out \"" + outputJson.getFullPathName() + "\"";

    DBG("PythonBridge: launching: " + cmd);

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

        MatchResult fail;
        fail.success = false;
        fail.errorMessage = finished
            ? ("Python exited with code " + juce::String(exitCode) + ": " + stdErr)
            : "Python process timed out after 10 minutes.";

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


