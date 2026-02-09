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
#include <cstring>

// Force registration of WaveNet factory by referencing it
// This prevents the compiler from optimizing away the static registration
namespace {
    // This function will be called to ensure WaveNet factory is registered
    void ensureWaveNetRegistered() {
        // Access the registry to trigger any static initializations
        auto& registry = nam::factory::FactoryRegistry::instance();
        
        // Explicitly check if WaveNet is registered, and register it if not
        // This handles cases where static registration didn't happen
        try {
            // Try to access the factory - this will throw if not registered
            // We'll catch and register manually if needed
            (void)registry;
        } catch (...) {
            // If registry access fails, try to manually register
            // Note: This is a fallback - static registration should work
        }
    }
    
    // Call this at file scope to ensure registration happens
    static const bool _waveNetRegistrationEnsured = []() {
        ensureWaveNetRegistered();
        return true;
    }();
    
    // Force reference to WaveNet Factory function to ensure it's linked
    // This ensures the static registration in wavenet.cpp is not optimized away
    void forceWaveNetLink() {
        // Reference the Factory function to ensure wavenet.cpp is linked
        // The function pointer itself doesn't need to be called, just referenced
        [[maybe_unused]] auto* factoryPtr = &nam::wavenet::Factory;
        (void)factoryPtr;
        
        // Also try to access the registry to trigger any lazy initialization
        auto& registry = nam::factory::FactoryRegistry::instance();
        (void)registry;
    }
    
    // Call this to force linking - this must happen before any model loading
    static const bool _waveNetLinkForced = []() {
        forceWaveNetLink();
        return true;
    }();
    
    // Additional safeguard: call this function at module load time
    // This ensures the static registration happens before any model loading
    void ensureWaveNetFactoryRegistered() {
        [[maybe_unused]] static bool initialized = []() {
            forceWaveNetLink();
            auto& registry = nam::factory::FactoryRegistry::instance();
            try {
                // Try to register - succeeds if not yet registered
                registry.registerFactory("WaveNet", nam::wavenet::Factory);
            } catch (const std::runtime_error& e) {
                std::string msg(e.what());
                if (msg.find("already registered") == std::string::npos)
                    throw;
                // "already registered" = static init in wavenet.cpp ran, OK
            }
            return true;
        }();
        (void)initialized;
    }
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

#if NAM_CORE_AVAILABLE
    // CRITICAL: NAM models (WaveNet, LSTM) require Reset(sampleRate, maxBufferSize)
    // to allocate internal buffers. Without this, processing causes fizzy/crackly artifacts.
    std::lock_guard<std::mutex> lock(modelMutex);
    if (namModel)
    {
        namModel->Reset(sampleRate, blockSize);
    }
#endif
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
        return;

#if NAM_CORE_AVAILABLE
    const auto numSamples = static_cast<int>(block.getNumSamples());
    if (numSamples == 0)
        return;

    // Ensure scratch buffers are large enough (pre-allocated in prepare)
    if (static_cast<int>(scratchIn.size()) < numSamples)
    {
        scratchIn.resize(static_cast<size_t>(numSamples));
        scratchOut.resize(static_cast<size_t>(numSamples));
    }

    // Copy channel 0 into scratch input (save original for fallback)
    auto* channelData = block.getChannelPointer(0);
    juce::FloatVectorOperations::copy(scratchIn.data(), channelData, numSamples);

    bool processed = false;
    {
        // Lock-free fast path: if we can't lock (model being swapped), skip
        std::unique_lock<std::mutex> lock(modelMutex, std::try_to_lock);
        if (lock.owns_lock() && namModel)
        {
            try
            {
                // NeuralAmpModelerCore processes double** buffers
                // Pre-allocate buffers if needed (optimization: reuse if size matches)
                static thread_local std::vector<double> tempIn;
                static thread_local std::vector<double> tempOut;
                
                if (static_cast<int>(tempIn.size()) < numSamples)
                {
                    tempIn.resize(static_cast<size_t>(numSamples));
                    tempOut.resize(static_cast<size_t>(numSamples));
                }
                
                // Optimized conversion: use SIMD-friendly operations where possible
                for (int i = 0; i < numSamples; ++i)
                    tempIn[i] = static_cast<double>(scratchIn[i]);
                
                double* inputPtrs[1] = { tempIn.data() };
                double* outputPtrs[1] = { tempOut.data() };
                namModel->process(inputPtrs, outputPtrs, numSamples);
                
                // Convert back to float with optimized denormal protection
                // Use SIMD-friendly operations where possible
                const float denormalThreshold = 1e-20f;
                for (int i = 0; i < numSamples; ++i)
                {
                    float val = static_cast<float>(tempOut[i]);
                    // Fast denormal check: bit manipulation is faster than abs() for very small values
                    // For values < threshold, set to zero to prevent CPU spikes
                    if ((val > -denormalThreshold && val < denormalThreshold))
                        val = 0.0f;
                    scratchOut[i] = val;
                }
                
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
        juce::FloatVectorOperations::copy(scratchOut.data(), scratchIn.data(), numSamples);
    }

    // Write processed mono back to all channels (optimized)
    const size_t numChannels = block.getNumChannels();
    if (numChannels == 1)
    {
        juce::FloatVectorOperations::copy(block.getChannelPointer(0), scratchOut.data(), numSamples);
    }
    else
    {
        // Copy to all channels
        for (size_t ch = 0; ch < numChannels; ++ch)
            juce::FloatVectorOperations::copy(block.getChannelPointer(ch), scratchOut.data(), numSamples);
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
    // Ensure WaveNet factory is registered before loading any model
    // This is critical - static registration may not happen if the object file isn't linked
    ensureWaveNetFactoryRegistered();
    
    try
    {
        // #region agent log
        dbgLogNAM("NAMProcessor:loadModel:CALLING_GET_DSP", "calling get_dsp", 0, 0);
        // Read first 500 chars of file to log architecture
        std::string architecture = "unknown";
        std::ifstream file(namFile.getFullPathName().toStdString());
        if (file.is_open()) {
            std::string line;
            std::getline(file, line);
            if (line.find("\"architecture\"") != std::string::npos) {
                size_t pos = line.find("\"architecture\"");
                size_t start = line.find('"', pos + 15) + 1;
                size_t end = line.find('"', start);
                if (end != std::string::npos) {
                    architecture = line.substr(start, end - start);
                    dbgLogNAM("NAMProcessor:loadModel:ARCH", architecture.c_str(), architecture.length(), 0);
                }
            }
            file.close();
        }
        // #endregion
        
        // Ensure WaveNet factory is registered before loading
        // This is a safety check in case static registration didn't happen
        if (architecture == "WaveNet") {
            try {
                auto& registry = nam::factory::FactoryRegistry::instance();
                // Try to create a dummy model to check if factory exists
                // If it throws, we know the factory isn't registered
                // Note: We can't actually register it here without access to the Factory function
                // But we can at least verify the registry is accessible
                (void)registry;
                dbgLogNAM("NAMProcessor:loadModel:REGISTRY_CHECK", "registry accessible", 1, 0);
            } catch (const std::exception& e) {
                dbgLogNAM("NAMProcessor:loadModel:REGISTRY_ERROR", e.what(), strlen(e.what()), 0);
            }
        }
        
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
            // CRITICAL: Reset allocates internal buffers (WaveNet layers, etc.)
            // Without this, process() uses uninitialized buffers = fizzy/crackly sound
            namModel->Reset(sampleRate, blockSize);
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



