/*
  ==============================================================================
    NAMProcessor.h
    Wrapper around NeuralAmpModelerCore for loading .nam files and
    running real-time inference inside a JUCE audio graph.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <filesystem>

#if NAM_CORE_AVAILABLE
#include "NAM/dsp.h"   // NeuralAmpModelerCore public header
#endif

//==============================================================================
/**
 * Wraps NeuralAmpModelerCore so it can be dropped into a JUCE DSP chain.
 *
 * Usage:
 *   NAMProcessor pedal;
 *   pedal.prepare(spec);
 *   pedal.loadModel(juce::File("DS1.nam"));   // background-safe
 *   pedal.process(block);                      // real-time safe
 */
class NAMProcessor
{
public:
    NAMProcessor();
    ~NAMProcessor();

    //==========================================================================
    /** Prepare the processor (call from prepareToPlay). */
    void prepare(const juce::dsp::ProcessSpec& spec);

    /** Reset internal state. */
    void reset();

    /** Process a block of audio in-place.  Real-time safe. */
    void process(juce::dsp::AudioBlock<float>& block);

    //==========================================================================
    /** Load a .nam model file.  Thread-safe (uses a lock for the swap). */
    bool loadModel(const juce::File& namFile);

    /** Unload the current model. */
    void clearModel();

    /** Returns true when a model is loaded and ready for processing. */
    bool isModelLoaded() const noexcept { return modelReady.load(); }

    /** Returns the name of the currently loaded model (empty if none). */
    juce::String getModelName() const;

private:
    //==========================================================================
#if NAM_CORE_AVAILABLE
    std::unique_ptr<nam::DSP> namModel;
#endif

    std::mutex modelMutex;           // guards model swap only
    std::atomic<bool> modelReady { false };

    double sampleRate = 44100.0;
    int    blockSize  = 512;

    juce::String currentModelName;

    // Scratch buffer for NAM (needs contiguous float*)
    std::vector<float> scratchIn;
    std::vector<float> scratchOut;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NAMProcessor)
};



