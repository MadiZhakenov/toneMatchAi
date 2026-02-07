/*
  ==============================================================================
    NAMProcessor.cpp
    Wrapper around NeuralAmpModelerCore for real-time NAM inference.
  ==============================================================================
*/

#include "NAMProcessor.h"

#if NAM_CORE_AVAILABLE
#include "NAM/dsp.h"
#include "NAM/activations.h"
#endif

//==============================================================================
NAMProcessor::NAMProcessor()  = default;
NAMProcessor::~NAMProcessor() = default;

//==============================================================================
void NAMProcessor::prepare(const juce::dsp::ProcessSpec& spec)
{
    sampleRate = spec.sampleRate;
    blockSize  = static_cast<int>(spec.maximumBlockSize);

    scratchIn.resize(static_cast<size_t>(blockSize), 0.0f);
    scratchOut.resize(static_cast<size_t>(blockSize), 0.0f);
}

void NAMProcessor::reset()
{
    std::fill(scratchIn.begin(), scratchIn.end(), 0.0f);
    std::fill(scratchOut.begin(), scratchOut.end(), 0.0f);
}

//==============================================================================
void NAMProcessor::process(juce::dsp::AudioBlock<float>& block)
{
    if (! modelReady.load(std::memory_order_acquire))
        return;   // pass-through when no model is loaded

#if NAM_CORE_AVAILABLE
    // NAMCore processes mono — use channel 0
    const auto numSamples = static_cast<int>(block.getNumSamples());

    // Ensure scratch buffers are large enough
    if (static_cast<int>(scratchIn.size()) < numSamples)
    {
        scratchIn.resize(static_cast<size_t>(numSamples));
        scratchOut.resize(static_cast<size_t>(numSamples));
    }

    // Copy channel 0 into scratch input
    auto* channelData = block.getChannelPointer(0);
    std::copy(channelData, channelData + numSamples, scratchIn.begin());

    {
        // Lock-free fast path: if we can't lock (model being swapped), skip
        std::unique_lock<std::mutex> lock(modelMutex, std::try_to_lock);
        if (lock.owns_lock() && namModel)
        {
            // NeuralAmpModelerCore processes a buffer of floats
            namModel->process(scratchIn.data(), scratchOut.data(), numSamples);
            namModel->finalize_(numSamples);
        }
        else
        {
            // Model is being swapped — pass through
            std::copy(scratchIn.begin(), scratchIn.begin() + numSamples, scratchOut.begin());
        }
    }

    // Write processed mono back to all channels
    for (size_t ch = 0; ch < block.getNumChannels(); ++ch)
    {
        auto* dest = block.getChannelPointer(ch);
        std::copy(scratchOut.begin(), scratchOut.begin() + numSamples, dest);
    }

#else
    // NAM Core not available — pass through (audio is untouched)
    juce::ignoreUnused(block);
#endif
}

//==============================================================================
bool NAMProcessor::loadModel(const juce::File& namFile)
{
    if (! namFile.existsAsFile())
        return false;

#if NAM_CORE_AVAILABLE
    try
    {
        auto newModel = nam::get_dsp(namFile.getFullPathName().toStdString());
        if (newModel == nullptr)
            return false;

        {
            std::lock_guard<std::mutex> lock(modelMutex);
            namModel = std::move(newModel);
        }

        currentModelName = namFile.getFileNameWithoutExtension();
        modelReady.store(true, std::memory_order_release);
        return true;
    }
    catch (const std::exception& e)
    {
        DBG("NAMProcessor::loadModel failed: " << e.what());
        return false;
    }
#else
    // Without NAM Core we just record the name for UI purposes
    currentModelName = namFile.getFileNameWithoutExtension();
    modelReady.store(true, std::memory_order_release);
    return true;
#endif
}

void NAMProcessor::clearModel()
{
    modelReady.store(false, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(modelMutex);
#if NAM_CORE_AVAILABLE
        namModel.reset();
#endif
    }

    currentModelName.clear();
}

juce::String NAMProcessor::getModelName() const
{
    return currentModelName;
}


