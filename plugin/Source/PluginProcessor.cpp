/*
  ==============================================================================
    PluginProcessor.cpp
    ToneMatch AI — main AudioProcessor implementation.
  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
ToneMatchAudioProcessor::ToneMatchAudioProcessor()
    : AudioProcessor(BusesProperties()
          .withInput ("Input",  juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout()),
      progressState("ProgressState")
{
    // Initialize progress state
    progressState.setProperty("progressStage", 0, nullptr);  // Idle
    progressState.setProperty("statusText", "Ready", nullptr);
    progressState.setProperty("progress", 0.0, nullptr);
}

ToneMatchAudioProcessor::~ToneMatchAudioProcessor() = default;

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout
ToneMatchAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("inputGain", 1), "Input Gain",
        juce::NormalisableRange<float>(-24.0f, 24.0f, 0.1f), 0.0f, "dB"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("preEqGainDb", 1), "Pre-EQ Gain",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f, "dB"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("preEqFreqHz", 1), "Pre-EQ Freq",
        juce::NormalisableRange<float>(400.0f, 3000.0f, 1.0f, 0.5f), 800.0f, "Hz"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("reverbWet", 1), "Reverb Wet",
        juce::NormalisableRange<float>(0.0f, 0.7f, 0.01f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("reverbRoomSize", 1), "Reverb Room",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("delayTimeMs", 1), "Delay Time",
        juce::NormalisableRange<float>(50.0f, 500.0f, 1.0f), 100.0f, "ms"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("delayMix", 1), "Delay Mix",
        juce::NormalisableRange<float>(0.0f, 0.5f, 0.01f), 0.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("finalEqGainDb", 1), "Final EQ Gain",
        juce::NormalisableRange<float>(-3.0f, 3.0f, 0.1f), 0.0f, "dB"));

    return { params.begin(), params.end() };
}

//==============================================================================
void ToneMatchAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::dsp::ProcessSpec spec;
    spec.sampleRate       = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels      = static_cast<juce::uint32>(getTotalNumOutputChannels());

    currentSampleRate = sampleRate;
    currentBlockSize  = samplesPerBlock;

    // Prepare all DSP components
    input_gain.prepare(spec);
    input_gain.setRampDurationSeconds(0.02);

    pedal.prepare(spec);
    amp.prepare(spec);

    ir_cabinet.prepare(spec);

    delay_line.prepare(spec);
    delay_line.setMaximumDelayInSamples(static_cast<int>(sampleRate));  // max 1 sec
    delay_dry_buffer.setSize(static_cast<int>(spec.numChannels),
                             static_cast<int>(spec.maximumBlockSize));

    reverb_unit.prepare(spec);

    // Pre-EQ filter
    pre_eq_coeffs = juce::dsp::IIR::Coefficients<float>::makePeakFilter(
        sampleRate, 800.0f, 0.707f, 1.0f);
    pre_eq_filter.prepare(spec);
    pre_eq_filter.coefficients = pre_eq_coeffs;

    // Final gain
    final_eq_gain.prepare(spec);
    final_eq_gain.setRampDurationSeconds(0.02);

    // Prepare DI capture buffer (store up to 10 seconds for matching)
    capturedDI.setSize(1, static_cast<int>(sampleRate * 10.0));
    capturedDI.clear();
}

void ToneMatchAudioProcessor::releaseResources()
{
    input_gain.reset();
    pedal.reset();
    amp.reset();
    ir_cabinet.reset();
    delay_line.reset();
    reverb_unit.reset();
    pre_eq_filter.reset();
    final_eq_gain.reset();
}

bool ToneMatchAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Support mono-in/stereo-out or stereo/stereo
    const auto& mainInput  = layouts.getMainInputChannelSet();
    const auto& mainOutput = layouts.getMainOutputChannelSet();

    if (mainOutput != juce::AudioChannelSet::mono()
        && mainOutput != juce::AudioChannelSet::stereo())
        return false;

    // Input can be mono or match output
    if (mainInput != juce::AudioChannelSet::mono()
        && mainInput != mainOutput)
        return false;

    return true;
}

void ToneMatchAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer& /*midi*/)
{
    juce::ScopedNoDenormals noDenormals;

    // Clear extra output channels
    for (int ch = getTotalNumInputChannels(); ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, buffer.getNumSamples());

    // ── CAPTURE DI (before processing) ────────────────────────────────────────
    if (capturing.load(std::memory_order_acquire))
    {
        const int numSamples = buffer.getNumSamples();
        const int numChannels = getTotalNumInputChannels();
        
        // Convert to mono and append to capturedDI
        int writePos = capturedDIWritePos.load(std::memory_order_relaxed);
        int available = capturedDI.getNumSamples() - writePos;
        int toWrite = juce::jmin(numSamples, available);
        
        if (toWrite > 0 && numChannels > 0)
        {
            // Sum to mono (or use channel 0 if mono)
            if (numChannels == 1)
            {
                capturedDI.copyFrom(0, writePos, buffer, 0, 0, toWrite);
            }
            else
            {
                // Sum stereo to mono
                for (int i = 0; i < toWrite; ++i)
                {
                    float sum = 0.0f;
                    for (int ch = 0; ch < numChannels; ++ch)
                        sum += buffer.getSample(ch, i);
                    capturedDI.setSample(0, writePos + i, sum / numChannels);
                }
            }
            capturedDIWritePos.store(writePos + toWrite, std::memory_order_release);
        }
    }
    // ── END CAPTURE ───────────────────────────────────────────────────────────

    // ── Full DSP Chain Processing ─────────────────────────────────────────────
    const auto numSamples  = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    // Read parameters from APVTS (lock-free)
    const float gainDb      = apvts.getRawParameterValue("inputGain")->load();
    const float eqGainDb    = apvts.getRawParameterValue("preEqGainDb")->load();
    const float eqFreq      = apvts.getRawParameterValue("preEqFreqHz")->load();
    const float revWet      = apvts.getRawParameterValue("reverbWet")->load();
    const float revRoom     = apvts.getRawParameterValue("reverbRoomSize")->load();
    const float delTime     = apvts.getRawParameterValue("delayTimeMs")->load();
    const float delMixVal   = apvts.getRawParameterValue("delayMix")->load();
    const float finGainDb   = apvts.getRawParameterValue("finalEqGainDb")->load();

    juce::dsp::AudioBlock<float> block(buffer);

    // ── Stage 1: Input Gain ──────────────────────────────────────────────────
    input_gain.setGainDecibels(gainDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        input_gain.process(ctx);
    }

    // ── Stage 1b: Pre-EQ (parametric peak) ───────────────────────────────────
    {
        const float linearGain = juce::Decibels::decibelsToGain(eqGainDb);
        *pre_eq_coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            currentSampleRate, juce::jlimit(20.0f, 20000.0f, eqFreq), 0.707f, linearGain);

        // Process mono (channel 0) through IIR; copy to other channels
        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            pre_eq_filter.process(ctx);

            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 2: NAM Pedal ───────────────────────────────────────────────────
    pedal.process(block);

    // ── Stage 3: NAM Amp ─────────────────────────────────────────────────────
    amp.process(block);

    // ── Stage 4: IR Cabinet (Convolution) ────────────────────────────────────
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        ir_cabinet.process(ctx);
    }

    // ── Stage 5: Delay ───────────────────────────────────────────────────────
    {
        // Save dry signal for wet/dry mixing
        for (int ch = 0; ch < numChannels; ++ch)
            delay_dry_buffer.copyFrom(ch, 0, buffer, ch, 0, numSamples);

        const float delaySamples = (delTime / 1000.0f) * static_cast<float>(currentSampleRate);
        delay_line.setDelay(juce::jlimit(1.0f, static_cast<float>(currentSampleRate), delaySamples));

        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                const float dry = data[i];
                delay_line.pushSample(ch, dry);
                const float wet = delay_line.popSample(ch);
                data[i] = dry * (1.0f - delMixVal) + wet * delMixVal;
            }
        }
    }

    // ── Stage 6: Reverb ──────────────────────────────────────────────────────
    {
        juce::dsp::Reverb::Parameters reverbParams;
        reverbParams.wetLevel  = revWet;
        reverbParams.dryLevel  = 1.0f - revWet;
        reverbParams.roomSize  = revRoom;
        reverbParams.damping   = 0.5f;
        reverbParams.width     = 1.0f;
        reverb_unit.setParameters(reverbParams);

        juce::dsp::ProcessContextReplacing<float> ctx(block);
        reverb_unit.process(ctx);
    }

    // ── Final Gain ───────────────────────────────────────────────────────────
    final_eq_gain.setGainDecibels(finGainDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        final_eq_gain.process(ctx);
    }
}

//==============================================================================
juce::AudioProcessorEditor* ToneMatchAudioProcessor::createEditor()
{
    return new ToneMatchAudioProcessorEditor(*this);
}

//==============================================================================
void ToneMatchAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = PresetManager::stateToVar(apvts, *this);
    auto json  = juce::JSON::toString(state, false);
    destData.append(json.toRawUTF8(), json.getNumBytesAsUTF8());
}

void ToneMatchAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto json   = juce::String::fromUTF8(static_cast<const char*>(data), sizeInBytes);
    auto parsed = juce::JSON::parse(json);

    if (parsed.isObject())
        PresetManager::varToState(parsed, apvts, *this);
}

//==============================================================================
int ToneMatchAudioProcessor::getProgressStage() const
{
    return progressState.getProperty("progressStage", 0);
}

void ToneMatchAudioProcessor::setProgressStage(int stage, const juce::String& statusText)
{
    progressState.setProperty("progressStage", stage, nullptr);
    if (statusText.isNotEmpty())
        progressState.setProperty("statusText", statusText, nullptr);
    
    // Set progress percentage based on stage
    double progress = 0.0;
    if (stage == 1) progress = 0.3;      // GridSearch
    else if (stage == 2) progress = 0.7; // Optimizing
    else if (stage == 3) progress = 1.0; // Done
    
    progressState.setProperty("progress", progress, nullptr);
    
    // Debug logging for testing
    DBG("[Progress] Stage: " + juce::String(stage) + 
        ", Status: " + statusText + 
        ", Progress: " + juce::String(progress * 100.0, 1) + "%");
}

void ToneMatchAudioProcessor::triggerMatch(const juce::File& refFile)
{
    DBG("[ToneMatch] triggerMatch called with file: " + refFile.getFullPathName());
    
    if (pythonBridge.isRunning())
    {
        DBG("[ToneMatch] Python bridge already running, ignoring");
        return;
    }

    // Stop capturing before saving
    stopCapturingDI();

    // Save the captured DI buffer to a temp .wav file
    juce::File diTemp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                            .getChildFile("tonematch_di.wav");

    // Get the actual number of samples captured
    int numCapturedSamples = capturedDIWritePos.load(std::memory_order_acquire);
    DBG("[ToneMatch] Captured DI samples: " + juce::String(numCapturedSamples));
    
    // If no DI captured, use the reference file as DI for testing
    if (numCapturedSamples == 0)
    {
        DBG("[ToneMatch] No DI audio captured - using reference file as DI for testing");
        diTemp = refFile;  // Use reference as DI for testing
    }
    else
    {
        // Write captured mono DI to file (only the captured portion)
        if (auto writer = std::unique_ptr<juce::AudioFormatWriter>(
                juce::WavAudioFormat().createWriterFor(
                    new juce::FileOutputStream(diTemp),
                    getSampleRate(), 1, 16, {}, 0)))
        {
            writer->writeFromAudioSampleBuffer(capturedDI, 0, numCapturedSamples);
            DBG("[ToneMatch] Saved " + juce::String(numCapturedSamples) + " samples to " + diTemp.getFullPathName());
        }
        else
        {
            DBG("[ToneMatch] Failed to create WAV writer for DI file");
            setProgressStage(0, "Error: Failed to save DI");
            return;
        }
    }

    // Set initial progress state
    DBG("[ToneMatch] Setting progress stage to 1 (Grid Search)");
    setProgressStage(1, "Grid Search...");

    DBG("[ToneMatch] Starting Python bridge...");
    pythonBridge.startMatch(diTemp, refFile,
        [this](const MatchResult& result) { 
            DBG("[ToneMatch] Python bridge callback received");
            onMatchComplete(result); 
        });
}

void ToneMatchAudioProcessor::onMatchComplete(const MatchResult& result)
{
    // This runs on the message thread
    if (! result.success)
    {
        DBG("Match failed: " + result.errorMessage);
        setProgressStage(0, "Error: " + result.errorMessage);
        return;
    }
    
    // Update progress to optimizing stage (if not already done)
    setProgressStage(2, "Optimizing...");

    // Build RigParameters from MatchResult
    RigParameters params;
    params.fx_path = result.fxNamPath;
    params.amp_path = result.ampNamPath;
    params.ir_path = result.irPath;
    params.reverb_wet = result.reverbWet;
    params.reverb_room_size = result.reverbRoomSize;
    params.delay_time_ms = result.delayTimeMs;
    params.delay_mix = result.delayMix;
    params.input_gain_db = result.inputGainDb;
    params.pre_eq_gain_db = result.preEqGainDb;
    params.pre_eq_freq_hz = result.preEqFreqHz;
    params.final_eq_gain_db = result.finalEqGainDb;

    // Store last match result for UI display
    lastAmpName = result.ampNamName;
    lastCabName = result.irName;

    // Apply the complete rig
    applyNewRig(params);
    
    // Set progress to done
    setProgressStage(3, "Done");

    DBG("Match complete! Loss = " + juce::String(result.loss, 6));
}

//==============================================================================
void ToneMatchAudioProcessor::startCapturingDI()
{
    capturing.store(true, std::memory_order_release);
    capturedDIWritePos.store(0, std::memory_order_relaxed);
    capturedDI.clear();
}

void ToneMatchAudioProcessor::stopCapturingDI()
{
    capturing.store(false, std::memory_order_release);
}

//==============================================================================
void ToneMatchAudioProcessor::applyNewRig(const RigParameters& params)
{
    // Load NAM models and save paths
    if (params.fx_path.isNotEmpty())
    {
        if (! pedal.loadModel(juce::File(params.fx_path)))
            DBG("Failed to load pedal model: " + params.fx_path);
        else
        {
            currentFxPath = params.fx_path;
            DBG("Loaded pedal model: " + pedal.getModelName());
        }
    }

    if (params.amp_path.isNotEmpty())
    {
        if (! amp.loadModel(juce::File(params.amp_path)))
            DBG("Failed to load amp model: " + params.amp_path);
        else
        {
            currentAmpPath = params.amp_path;
            DBG("Loaded amp model: " + amp.getModelName());
        }
    }

    // Load IR and save path
    if (params.ir_path.isNotEmpty())
    {
        if (! loadIR(juce::File(params.ir_path)))
            DBG("Failed to load IR: " + params.ir_path);
        else
        {
            currentIrPath = params.ir_path;
            DBG("Loaded IR: " + currentIRName);
        }
    }

    // Update all parameters via APVTS
    auto setParam = [this](const juce::String& id, float value)
    {
        if (auto* p = apvts.getParameter(id))
            p->setValueNotifyingHost(p->convertTo0to1(value));
    };

    setParam("inputGain",      params.input_gain_db);
    setParam("preEqGainDb",    params.pre_eq_gain_db);
    setParam("preEqFreqHz",    params.pre_eq_freq_hz);
    setParam("reverbWet",      params.reverb_wet);
    setParam("reverbRoomSize", params.reverb_room_size);
    setParam("delayTimeMs",    params.delay_time_ms);
    setParam("delayMix",       params.delay_mix);
    setParam("finalEqGainDb",  params.final_eq_gain_db);
}

//==============================================================================
bool ToneMatchAudioProcessor::loadPresetToProcessor(const juce::File& file)
{
    return presetManager.loadPreset(file, apvts, *this);
}

//==============================================================================
bool ToneMatchAudioProcessor::loadPedalModel(const juce::File& namFile)
{
    if (pedal.loadModel(namFile))
    {
        currentFxPath = namFile.getFullPathName();
        return true;
    }
    return false;
}

bool ToneMatchAudioProcessor::loadAmpModel(const juce::File& namFile)
{
    if (amp.loadModel(namFile))
    {
        currentAmpPath = namFile.getFullPathName();
        return true;
    }
    return false;
}

bool ToneMatchAudioProcessor::loadIR(const juce::File& irFile)
{
    if (! irFile.existsAsFile())
        return false;

    ir_cabinet.loadImpulseResponse(irFile,
                                  juce::dsp::Convolution::Stereo::no,
                                  juce::dsp::Convolution::Trim::yes,
                                  0);   // 0 = use full IR
    currentIRName = irFile.getFileNameWithoutExtension();
    currentIrPath = irFile.getFullPathName();
    return true;
}

//==============================================================================
// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ToneMatchAudioProcessor();
}


