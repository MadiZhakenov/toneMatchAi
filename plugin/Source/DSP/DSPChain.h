/*
  ==============================================================================
    DSPChain.h
    Six-stage real-time signal chain:
      Input Gain -> NAM Pedal -> NAM Amp -> IR Cabinet -> Delay -> Reverb
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>

#include "NAMProcessor.h"

//==============================================================================
/**
 * Owns all DSP stages and processes audio blocks through the full chain.
 *
 * All parameter pointers come from AudioProcessorValueTreeState atomics,
 * so reads are lock-free and real-time safe.
 */
class DSPChain
{
public:
    DSPChain();
    ~DSPChain();

    //==========================================================================
    /** Call from AudioProcessor::prepareToPlay. */
    void prepare(const juce::dsp::ProcessSpec& spec);

    /** Call from AudioProcessor::releaseResources. */
    void reset();

    /** Process one block through the full chain.  Real-time safe. */
    void process(juce::AudioBuffer<float>& buffer);

    //==========================================================================
    // Parameter pointer setters — call once after APVTS is created.

    void setInputGainParameter(std::atomic<float>* p)      { inputGainDb       = p; }
    void setPreEqGainParameter(std::atomic<float>* p)      { preEqGainDb       = p; }
    void setPreEqFreqParameter(std::atomic<float>* p)      { preEqFreqHz       = p; }
    void setReverbWetParameter(std::atomic<float>* p)      { reverbWet         = p; }
    void setReverbRoomSizeParameter(std::atomic<float>* p) { reverbRoomSize    = p; }
    void setDelayTimeParameter(std::atomic<float>* p)      { delayTimeMs       = p; }
    void setDelayMixParameter(std::atomic<float>* p)       { delayMix          = p; }
    void setFinalEqGainParameter(std::atomic<float>* p)    { finalEqGainDb     = p; }

    //==========================================================================
    // Model / IR loading (called from message thread or background thread).

    bool loadPedalModel(const juce::File& namFile);
    bool loadAmpModel(const juce::File& namFile);
    bool loadIR(const juce::File& irFile);

    void clearPedalModel();
    void clearAmpModel();
    void clearIR();

    //==========================================================================
    // Accessors for UI

    juce::String getPedalModelName() const { return pedalNAM.getModelName(); }
    juce::String getAmpModelName()   const { return ampNAM.getModelName(); }
    juce::String getIRName()         const { return currentIRName; }

private:
    //==========================================================================
    // DSP stages
    juce::dsp::Gain<float>       inputGain;
    NAMProcessor                 pedalNAM;
    NAMProcessor                 ampNAM;
    juce::dsp::Convolution       irCabinet;
    juce::dsp::DelayLine<float>  delayLine { 44100 };   // max 1 sec
    juce::dsp::Reverb            reverb;

    // Delay wet/dry mixing buffer
    juce::AudioBuffer<float>     delayDryBuffer;

    //==========================================================================
    // Atomic parameter pointers (non-owning, point into APVTS storage)
    std::atomic<float>* inputGainDb    = nullptr;
    std::atomic<float>* preEqGainDb    = nullptr;
    std::atomic<float>* preEqFreqHz    = nullptr;
    std::atomic<float>* reverbWet      = nullptr;
    std::atomic<float>* reverbRoomSize = nullptr;
    std::atomic<float>* delayTimeMs    = nullptr;
    std::atomic<float>* delayMix       = nullptr;
    std::atomic<float>* finalEqGainDb  = nullptr;

    //==========================================================================
    double currentSampleRate = 44100.0;
    int    currentBlockSize  = 512;
    juce::String currentIRName;

    // Pre-EQ biquad filter
    juce::dsp::IIR::Filter<float>            preEqFilter;
    juce::dsp::IIR::Coefficients<float>::Ptr preEqCoeffs;

    // Final EQ (simple gain)
    juce::dsp::Gain<float> finalGain;

    //==========================================================================
    /** Helper: read an atomic<float>* safely, returning `fallback` if null. */
    static float readParam(std::atomic<float>* p, float fallback = 0.0f)
    {
        return p ? p->load(std::memory_order_relaxed) : fallback;
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DSPChain)
};


