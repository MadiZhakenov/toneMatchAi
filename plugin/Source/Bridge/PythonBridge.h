/*
  ==============================================================================
    PythonBridge.h
    Launches the Python optimizer as a child process, reads the resulting
    preset JSON, and delivers parsed parameters back to the message thread.
  ==============================================================================
*/

#pragma once

#include <juce_core/juce_core.h>
#include <juce_events/juce_events.h>
#include <functional>

//==============================================================================
/**
 * Parsed result from the Python tone-matching script.
 */
struct MatchResult
{
    // Rig info
    juce::String fxNamName;
    juce::String ampNamName;
    juce::String irName;
    juce::String fxNamPath;
    juce::String ampNamPath;
    juce::String irPath;

    // DSP parameters
    float inputGainDb    = 0.0f;  // Input gain from optimizer (used as overdrive_db)
    float overdriveDb     = 0.0f;  // Explicit overdrive parameter (if present in JSON, otherwise = inputGainDb)
    float preEqGainDb    = 0.0f;
    float preEqFreqHz    = 800.0f;
    float reverbWet      = 0.0f;
    float reverbRoomSize = 0.5f;
    float delayTimeMs    = 100.0f;
    float delayMix       = 0.0f;
    float finalEqGainDb  = 0.0f;

    float loss           = 0.0f;

    bool  success        = false;
    juce::String errorMessage;
};

//==============================================================================
/**
 * Runs the Python optimizer in a background thread via juce::ChildProcess.
 *
 * Usage:
 *   bridge.startMatch(diFile, refFile, [this](const MatchResult& r){ applyResult(r); });
 *
 * The callback is invoked on the JUCE message thread.
 */
class PythonBridge : private juce::Thread
{
public:
    using Callback = std::function<void(const MatchResult&)>;

    PythonBridge();
    ~PythonBridge() override;

    //==========================================================================
    /** Kick off a match.  Non-blocking — work happens on a background thread. */
    void startMatch(const juce::File& diFile,
                    const juce::File& refFile,
                    Callback onComplete);

    /** Returns true while the background process is running. */
    bool isRunning() const noexcept { return isThreadRunning(); }

    /** Set the path to the Python executable (default: "python"). */
    void setPythonPath(const juce::String& path) { pythonPath = path; }

    /** Set the path to the run_match.py script. */
    void setScriptPath(const juce::File& path) { scriptPath = path; }

private:
    void run() override;   // juce::Thread

    //==========================================================================
    /** Parse the JSON output written by the Python script. */
    static MatchResult parseResultJson(const juce::File& jsonFile);

    //==========================================================================
    juce::File   diFilePending;
    juce::File   refFilePending;
    Callback     callbackPending;

    juce::String pythonPath  { "python" };
    juce::File   scriptPath;

    static constexpr int timeoutMs = 600000;  // 10 minutes

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PythonBridge)
};



