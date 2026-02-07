/*
  ==============================================================================
    PluginProcessor.h
    ToneMatch AI — main AudioProcessor.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "DSP/DSPChain.h"
#include "Bridge/PythonBridge.h"
#include "Preset/PresetManager.h"

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
    void triggerMatch(const juce::File& refFile);

    /** Access the APVTS (for slider attachments in the editor). */
    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }

    /** Access the DSP chain (for model-name queries, loading models). */
    DSPChain& getDSPChain() { return dspChain; }

    /** Access the preset manager. */
    PresetManager& getPresetManager() { return presetManager; }

    /** Whether a match is currently running. */
    bool isMatchRunning() const { return pythonBridge.isRunning(); }

private:
    //==========================================================================
    /** Build the APVTS parameter layout. */
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    /** Called on the message thread when the Python bridge finishes. */
    void onMatchComplete(const MatchResult& result);

    //==========================================================================
    juce::AudioProcessorValueTreeState apvts;
    DSPChain       dspChain;
    PythonBridge   pythonBridge;
    PresetManager  presetManager;

    // For capturing DI audio to send to the Python bridge
    juce::AudioBuffer<float> capturedDI;
    bool capturing = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatchAudioProcessor)
};


