/*
  ==============================================================================
    PresetManager.h
    Save / load plugin state as .json preset files on disk.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_core/juce_core.h>

class DSPChain;  // forward

//==============================================================================
/**
 * Manages preset persistence (local disk only, no cloud).
 *
 * Preset JSON layout matches the Python bridge contract:
 * {
 *   "name": "My Preset",
 *   "rig":  { "fx_nam": "...", ... },
 *   "params": { "input_gain_db": 0.0, ... }
 * }
 */
class PresetManager
{
public:
    PresetManager();
    ~PresetManager();

    //==========================================================================
    /** Save current state to a .json file. */
    bool savePreset(const juce::File& file,
                    const juce::AudioProcessorValueTreeState& apvts,
                    const DSPChain& chain) const;

    /** Load state from a .json file.  Updates APVTS params and reloads models. */
    bool loadPreset(const juce::File& file,
                    juce::AudioProcessorValueTreeState& apvts,
                    DSPChain& chain) const;

    //==========================================================================
    /** Serialize APVTS state to a juce::var (for getStateInformation). */
    static juce::var stateToVar(const juce::AudioProcessorValueTreeState& apvts,
                                const DSPChain& chain);

    /** Restore APVTS state from a juce::var (for setStateInformation). */
    static void varToState(const juce::var& data,
                           juce::AudioProcessorValueTreeState& apvts,
                           DSPChain& chain);

    //==========================================================================
    /** Returns default preset directory (user documents / ToneMatchAI / Presets). */
    static juce::File getDefaultPresetDirectory();

private:
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PresetManager)
};


