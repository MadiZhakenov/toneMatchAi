/*
  ==============================================================================
    NAMProcessor.cpp
    Wrapper around NeuralAmpModelerCore for real-time NAM inference.
  ==============================================================================
*/

#include "NAMProcessor.h"

#if NAM_CORE_AVAILABLE
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/activations.h"
// Force WaveNet factory registration by including wavenet.h
// This ensures the static registration happens before get_dsp is called
#include "NAM/wavenet.h"
#include "NAM/registry.h"
#include <fstream>
#include <sstream>

// Force registration of WaveNet factory by referencing it
// This prevents the compiler from optimizing away the static registration
namespace {
    // This function will be called to ensure WaveNet factory is registered
    void ensureWaveNetRegistered() {
        // Access the registry to trigger any static initializations
        auto& registry = nam::factory::FactoryRegistry::instance();
        // The static registration in wavenet.cpp should have already happened
        // by the time this function is called (due to include order)
        (void)registry; // Suppress unused variable warning
    }
    
    // Call this at file scope to ensure registration happens
    static const bool _waveNetRegistrationEnsured = []() {
        ensureWaveNetRegistered();
        return true;
    }();
}
#endif

// #region agent log - debug helper for NAMProcessor
#include <fstream>
#include <chrono>
static void dbgLogNAM(const char* loc, const char* msg, int i1 = 0, int i2 = 0) {
    std::ofstream f("e:\\Users\\Desktop\\toneMatchAi\\.cursor\\debug.log", std::ios::app);
    if (!f.is_open()) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    f << "{\"location\":\"" << loc << "\",\"message\":\"" << msg
      << "\",\"hypothesisId\":\"J\",\"data\":{\"v1\":0,\"v2\":0,\"v3\":0"
      << ",\"i1\":" << i1 << ",\"i2\":" << i2
      << "},\"timestamp\":" << ms << "}\n";
    f.close();
}
// #endregion

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
    // Pass-through when no model is loaded - signal continues unchanged
    if (! modelReady.load(std::memory_order_acquire))
    {
        // Occasional logging to check why it's passing through
        static int passThroughCount = 0;
        if ((passThroughCount++ % 1000) == 0) {
            dbgLogNAM("NAMProcessor:process:PASS_THROUGH", "no model ready", 0, 0);
        }
        return;
    }

#if NAM_CORE_AVAILABLE
    const auto numSamples = static_cast<int>(block.getNumSamples());
    if (numSamples == 0)
        return;
    
    // Occasional logging to verify NAM is active
    static int activeCount = 0;
    if ((activeCount++ % 1000) == 0) {
        dbgLogNAM("NAMProcessor:process:ACTIVE", "NAM Core processing", 0, 0);
    }

    // Ensure scratch buffers are large enough
    if (static_cast<int>(scratchIn.size()) < numSamples)
    {
        scratchIn.resize(static_cast<size_t>(numSamples));
        scratchOut.resize(static_cast<size_t>(numSamples));
    }

    // Copy channel 0 into scratch input (save original for fallback)
    auto* channelData = block.getChannelPointer(0);
    std::copy(channelData, channelData + numSamples, scratchIn.begin());

    bool processed = false;
    {
        // Lock-free fast path: if we can't lock (model being swapped), skip
        std::unique_lock<std::mutex> lock(modelMutex, std::try_to_lock);
        if (lock.owns_lock() && namModel)
        {
            try
            {
                // NeuralAmpModelerCore processes double** buffers
                std::vector<double> tempIn(numSamples);
                std::vector<double> tempOut(numSamples, 0.0);
                
                for (int i = 0; i < numSamples; ++i)
                    tempIn[i] = static_cast<double>(scratchIn[i]);
                
                double* inputPtrs[1] = { tempIn.data() };
                double* outputPtrs[1] = { tempOut.data() };
                namModel->process(inputPtrs, outputPtrs, numSamples);
                
                // Convert back to float
                for (int i = 0; i < numSamples; ++i)
                    scratchOut[i] = static_cast<float>(tempOut[i]);
                
                processed = true;
            }
            catch (...)
            {
                // NAM processing failed - will fall back to pass-through
                processed = false;
            }
        }
    }

    if (!processed)
    {
        // Model is being swapped, not available, or failed — pass through unchanged
        std::copy(scratchIn.begin(), scratchIn.begin() + numSamples, scratchOut.begin());
    }

    // Write processed mono back to all channels
    for (size_t ch = 0; ch < block.getNumChannels(); ++ch)
    {
        auto* dest = block.getChannelPointer(ch);
        std::copy(scratchOut.begin(), scratchOut.begin() + numSamples, dest);
    }

#else
    // Occasional logging to check if Core is missing
    static int missingCoreCount = 0;
    if ((missingCoreCount++ % 1000) == 0) {
        dbgLogNAM("NAMProcessor:process:MISSING_CORE", "NAM_CORE_AVAILABLE NOT DEFINED", 0, 0);
    }
#endif
    // If NAM_CORE_AVAILABLE is not defined, audio passes through unchanged (block is untouched)
}

//==============================================================================
bool NAMProcessor::loadModel(const juce::File& namFile)
{
    // #region agent log
    dbgLogNAM("NAMProcessor:loadModel:START", "load model start", 0, 0);
    // #endregion
    
    if (! namFile.existsAsFile())
    {
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:FILE_NOT_EXISTS", "file not exists", 0, 0);
        // #endregion
        return false;
    }

    // #region agent log
    dbgLogNAM("NAMProcessor:loadModel:FILE_EXISTS", "file exists", 1, 0);
    // #endregion

#if NAM_CORE_AVAILABLE
    try
    {
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:CALLING_GET_DSP", "calling get_dsp", 0, 0);
        // Read first 500 chars of file to log architecture
        std::ifstream file(namFile.getFullPathName().toStdString());
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            if (line.find("\"architecture\"") != std::string::npos) {
                size_t pos = line.find("\"architecture\"");
                size_t start = line.find('"', pos + 15) + 1;
                size_t end = line.find('"', start);
                if (end != std::string::npos) {
                    std::string arch = line.substr(start, end - start);
                    dbgLogNAM("NAMProcessor:loadModel:ARCH", arch.c_str(), arch.length(), 0);
                }
            }
            file.close();
        }
        // #endregion
        
        auto newModel = nam::get_dsp(std::filesystem::path(namFile.getFullPathName().toStdString()));
        
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:GET_DSP_RESULT", "get_dsp result", newModel ? 1 : 0, 0);
        // #endregion
        
        if (newModel == nullptr)
        {
            // #region agent log
            dbgLogNAM("NAMProcessor:loadModel:NULL_MODEL", "get_dsp returned null", 0, 0);
            // #endregion
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(modelMutex);
            namModel = std::move(newModel);
        }

        currentModelName = namFile.getFileNameWithoutExtension();
        modelReady.store(true, std::memory_order_release);
        
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:SUCCESS", "model loaded successfully", 1, 0);
        // #endregion
        
        return true;
    }
    catch (const std::exception& e)
    {
        juce::String errorMsg = e.what();
        DBG("NAMProcessor::loadModel failed: " << errorMsg);
        // #region agent log
        // Log error message - truncate to 100 chars and escape quotes
        juce::String safeMsg = errorMsg.substring(0, 100).replace("\"", "'");
        dbgLogNAM("NAMProcessor:loadModel:EXCEPTION", safeMsg.toRawUTF8(), errorMsg.length(), 0);
        // #endregion
        return false;
    }
    catch (...)
    {
        DBG("NAMProcessor::loadModel failed: unknown exception");
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:UNKNOWN_EXCEPTION", "unknown exception", 0, 0);
        // #endregion
        return false;
    }
#else
    // Without NAM Core we just record the name for UI purposes
    currentModelName = namFile.getFileNameWithoutExtension();
    modelReady.store(true, std::memory_order_release);
    // #region agent log
    dbgLogNAM("NAMProcessor:loadModel:NO_CORE", "NAM_CORE_AVAILABLE not defined", 0, 0);
    // #endregion
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



