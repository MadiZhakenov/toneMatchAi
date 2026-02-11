/*
  ==============================================================================
    PluginProcessor.h
    ToneMatch AI — main AudioProcessor.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "DSP/NAMProcessor.h"
#include "DSP/SimpleGate.h"
#include "Bridge/PythonBridge.h"
#include "Preset/PresetManager.h"

//==============================================================================
/**
 * Parameters structure matching the JSON format from Python optimizer.
 */
struct RigParameters
{
    juce::String fx_path;
    juce::String amp_path;
    juce::String ir_path;
    float reverb_wet = 0.0f;
    float reverb_room_size = 0.5f;
    float delay_time_ms = 100.0f;
    float delay_mix = 0.0f;
    float overdrive_db = 0.0f;      // Applied before NAM for overdrive
    float input_gain_db = 0.0f;     // Applied at the end for volume
    float pre_eq_gain_db = 0.0f;
    float pre_eq_freq_hz = 800.0f;
    float hpf_freq = 70.0f;
    float lpf_freq = 8000.0f;
    bool ai_lock = false;
    bool cab_lock = false;
};

//==============================================================================
/**
 * Model entry for the library browser.
 */
struct ModelEntry
{
    juce::File file;              // Real file path
    juce::String displayName;     // Formatted name for UI display
};

//==============================================================================
class ToneMatchAudioProcessor : public juce::AudioProcessor
{
public:
    ToneMatchAudioProcessor();
    ~ToneMatchAudioProcessor() override;

    //==========================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==========================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    //==========================================================================
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi()  const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==========================================================================
    int getNumPrograms() override    { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    //==========================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    //==========================================================================
    // Public API for the Editor / Bridge

    /** Trigger a tone-match analysis in the background. */
    void triggerMatch(const juce::File& refFile, bool forceHighGain = false);

    /** Access the APVTS (for slider attachments in the editor). */
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }

    /** Access the preset manager. */
    PresetManager& getPresetManager() { return presetManager; }

    /** Load a preset from file and apply it to the processor. */
    bool loadPresetToProcessor(const juce::File& file);

    //==========================================================================
    // DSP Chain Access Methods

    /** Apply a complete rig configuration from RigParameters. */
    void applyNewRig(const RigParameters& params);

    /** Load a pedal NAM model. */
    bool loadPedalModel(const juce::File& namFile);

    /** Load an amp NAM model. */
    bool loadAmpModel(const juce::File& namFile);

    /** Load an IR cabinet file. */
    bool loadIR(const juce::File& irFile);

    /** Get the name of the currently loaded pedal model. */
    juce::String getPedalModelName() const { return pedal.getModelName(); }

    /** Get the name of the currently loaded amp model. */
    juce::String getAmpModelName() const { return amp.getModelName(); }

    /** Get the name of the currently loaded IR. */
    juce::String getIRName() const { return currentIRName; }

    /** Get the path of the currently loaded FX (pedal) model. */
    juce::String getCurrentFxPath() const { return currentFxPath; }

    /** Get the path of the currently loaded AMP model. */
    juce::String getCurrentAmpPath() const { return currentAmpPath; }

    /** Get the path of the currently loaded IR. */
    juce::String getCurrentIrPath() const { return currentIrPath; }

    /** Get the name of the last matched amp (from AI analysis). */
    juce::String getLastAmpName() const { return lastAmpName; }

    /** Get the name of the last matched cab (from AI analysis). */
    juce::String getLastCabName() const { return lastCabName; }

    /** Whether a match is currently running. */
    bool isMatchRunning() const { return pythonBridge.isRunning(); }

    /** Start capturing DI audio (call before processing audio you want to match). */
    void startCapturingDI();

    /** Stop capturing DI audio. */
    void stopCapturingDI();

    /** Whether DI audio is currently being captured. */
    bool isCapturingDI() const { return capturing.load(std::memory_order_acquire); }

    /** Get the number of samples captured so far. */
    int getCapturedDISamples() const { return capturedDIWritePos.load(std::memory_order_acquire); }

    /** Get the total capacity of the DI buffer. */
    int getDIBufferSize() const { return capturedDI.getNumSamples(); }

    /** Get current auto-compensation gain in dB (for UI display). */
    float getAutoCompensationDb() const { return autoCompensationDb.load(std::memory_order_acquire); }

    //==========================================================================
    // Progress State Management

    /** Get the progress state ValueTree (for Editor subscription). */
    juce::ValueTree& getProgressState() { return progressState; }

    /** Get current progress stage (0=Idle, 1=GridSearch, 2=Optimizing, 3=Done). */
    int getProgressStage() const;

    /** Set progress stage and status text. */
    void setProgressStage(int stage, const juce::String& statusText = {});

    /** Sync lock states from APVTS parameters. */
    void syncLockStates();

    //==========================================================================
    // Model Library Management

    /** Scan the assets/nam_models folder and populate availableModels. */
    void scanModels();

    /** Get the list of available models. */
    const std::vector<ModelEntry>& getAvailableModels() const { return availableModels; }

    /** Rescan models (useful for refreshing the list). */
    void rescanModels() { scanModels(); }

    /** Bypass amp model (unload and set to direct input). */
    void bypassAmpModel();

private:
    //==========================================================================
    /** Build the APVTS parameter layout. */
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    /** Called on the message thread when the Python bridge finishes. */
    void onMatchComplete(const MatchResult& result);

    //==========================================================================
    juce::AudioProcessorValueTreeState apvts;
    PythonBridge   pythonBridge;
    PresetManager  presetManager;

    // Progress state for UI updates
    juce::ValueTree progressState;

    //==========================================================================
    // DSP Chain Components

    // NAM Processors
    NAMProcessor pedal;
    NAMProcessor amp;

    // IR Cabinet (Convolution)
    juce::dsp::Convolution ir_cabinet;

    // Delay Line
    juce::dsp::DelayLine<float> delay_line { 44100 };  // max 1 sec at 44.1kHz

    // Reverb
    juce::dsp::Reverb reverb_unit;

    // Gain stages
    juce::dsp::Gain<float> overdrive_gain;  // Applied before NAM for overdrive
    juce::dsp::Gain<float> input_gain;      // Applied at the end for volume

    // Pre-EQ Filter
    juce::dsp::IIR::Filter<float> pre_eq_filter;
    juce::dsp::IIR::Coefficients<float>::Ptr pre_eq_coeffs;

    // HPF/LPF Filters (after IR)
    juce::dsp::IIR::Filter<float> hpf_filter;
    juce::dsp::IIR::Coefficients<float>::Ptr hpf_coeffs;
    juce::dsp::IIR::Filter<float> lpf_filter;
    juce::dsp::IIR::Coefficients<float>::Ptr lpf_coeffs;
    
    // Input Stage: DC Block & Safety HPF (80Hz, Q=0.7) - FIRST in chain
    juce::dsp::IIR::Filter<float> inputSafetyHPF;
    juce::dsp::IIR::Coefficients<float>::Ptr inputSafetyHPFCoeffs;
    
    // Pre-NAM Noise Gate (before any gain)
    SimpleGate preNAMNoiseGate;
    
    // Compensation Gain (RMS-based auto-compensation)
    juce::dsp::Gain<float> compensationGain;
    
    // Input Trim (manual user adjustment after auto-compensation)
    juce::dsp::Gain<float> inputTrimGain;
    
    // Aggressive high-pass filter for noise removal (before processing) - DEPRECATED, will be removed
    juce::dsp::IIR::Filter<float> aggressive_hpf;
    juce::dsp::IIR::Coefficients<float>::Ptr aggressive_hpf_coeffs;
    
    // Cached filter parameters to avoid recalculating coefficients every block
    float cachedPreEqGainDb = -999.0f;
    float cachedPreEqFreq = -999.0f;
    float cachedHpfFreq = -999.0f;
    float cachedLpfFreq = -999.0f;
    float cachedSampleRate = 0.0f;

    // Delay wet/dry mixing buffer
    juce::AudioBuffer<float> delay_dry_buffer;

    // Noise Gate state
    float noiseGateEnvelope = 0.0f;  // Current envelope value (0.0 to 1.0)

    // Current IR name (for UI display)
    juce::String currentIRName;

    // Current file paths (for preset saving)
    juce::String currentFxPath;
    juce::String currentAmpPath;
    juce::String currentIrPath;

    // Last match result (for UI display)
    juce::String lastAmpName;
    juce::String lastCabName;

    // Sample rate and block size (cached for processBlock)
    double currentSampleRate = 44100.0;
    int    currentBlockSize  = 512;

    //==========================================================================
    // For capturing DI audio to send to the Python bridge
    juce::AudioBuffer<float> capturedDI;
    std::atomic<bool> capturing { false };
    std::atomic<int> capturedDIWritePos { 0 };
    float capturedDIPeakDb = -100.0f;

    // Lock states for AI/CAB
    std::atomic<bool> aiLockEnabled { false };
    std::atomic<bool> cabLockEnabled { false };

    // Compensation gain: RMS-based linear gain applied to input signal to match target level.
    // Calculated once during DI capture and applied to all incoming audio before processing chain.
    std::atomic<float> inputCompensationLinear { 1.0f };
    
    // Auto-compensation in dB for UI display
    std::atomic<float> autoCompensationDb { 0.0f };

    //==========================================================================
    // Model Library
    std::vector<ModelEntry> availableModels;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatchAudioProcessor)
};


