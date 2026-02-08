/*
  ==============================================================================
    PluginProcessor.cpp
    ToneMatch AI — main AudioProcessor implementation.
  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

// #region agent log - debug helper
#include <fstream>
#include <chrono>
static int dbgBlockCount = 0;
static void dbgLog(const char* loc, const char* msg, const char* hyp,
                   float val1 = 0.f, float val2 = 0.f, float val3 = 0.f,
                   int i1 = 0, int i2 = 0, bool forceLog = false) {
    // Always log if forceLog=true (for non-audio-thread calls), otherwise log every 500th block
    if (!forceLog && (dbgBlockCount % 500) != 0) return;
    std::ofstream f("e:\\Users\\Desktop\\toneMatchAi\\.cursor\\debug.log", std::ios::app);
    if (!f.is_open()) return;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    f << "{\"location\":\"" << loc << "\",\"message\":\"" << msg
      << "\",\"hypothesisId\":\"" << hyp
      << "\",\"data\":{\"v1\":" << val1 << ",\"v2\":" << val2 << ",\"v3\":" << val3
      << ",\"i1\":" << i1 << ",\"i2\":" << i2
      << "},\"timestamp\":" << ms << "}\n";
    f.close();
}
static float peakOf(const juce::AudioBuffer<float>& buf, int ch) {
    if (ch >= buf.getNumChannels() || buf.getNumSamples() == 0) return 0.f;
    float pk = 0.f;
    auto* r = buf.getReadPointer(ch);
    for (int i = 0; i < buf.getNumSamples(); ++i)
        pk = std::max(pk, std::abs(r[i]));
    return pk;
}
static float peakToDb(float peak) {
    if (peak <= 0.f) return -200.f;
    return 20.f * std::log10(peak);
}
// #endregion

//==============================================================================
ToneMatchAudioProcessor::ToneMatchAudioProcessor()
    : AudioProcessor(BusesProperties()
          .withInput ("Input",  juce::AudioChannelSet::mono(), true)  // Mono input for guitar
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
        juce::ParameterID("overdrive", 1), "Overdrive",
        juce::NormalisableRange<float>(-12.0f, 40.0f, 0.1f), 0.0f, "dB"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("inputGain", 1), "Input Gain",
        juce::NormalisableRange<float>(-24.0f, 80.0f, 0.1f), 0.0f, "dB"));

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
        juce::ParameterID("hpfFreq", 1), "HPF Frequency",
        juce::NormalisableRange<float>(70.0f, 150.0f, 1.0f), 70.0f, "Hz"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("lpfFreq", 1), "LPF Frequency",
        juce::NormalisableRange<float>(4000.0f, 8000.0f, 1.0f), 8000.0f, "Hz"));

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("aiLock", 1), "AI Lock", false));

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("cabLock", 1), "CAB Lock", false));

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
    overdrive_gain.prepare(spec);
    overdrive_gain.setRampDurationSeconds(0.02);
    
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

    // HPF/LPF filters (after IR)
    hpf_coeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(
        sampleRate, 70.0f);
    hpf_filter.prepare(spec);
    hpf_filter.coefficients = hpf_coeffs;

    lpf_coeffs = juce::dsp::IIR::Coefficients<float>::makeLowPass(
        sampleRate, 8000.0f);
    lpf_filter.prepare(spec);
    lpf_filter.coefficients = lpf_coeffs;

    // Prepare DI capture buffer (store up to 30 seconds for matching)
    capturedDI.setSize(1, static_cast<int>(sampleRate * 30.0));
    capturedDI.clear();
}

void ToneMatchAudioProcessor::releaseResources()
{
    overdrive_gain.reset();
    input_gain.reset();
    pedal.reset();
    amp.reset();
    ir_cabinet.reset();
    delay_line.reset();
    reverb_unit.reset();
    pre_eq_filter.reset();
    hpf_filter.reset();
    lpf_filter.reset();
}

bool ToneMatchAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Support mono-in/stereo-out or stereo/stereo
    const auto& mainInput  = layouts.getMainInputChannelSet();
    const auto& mainOutput = layouts.getMainOutputChannelSet();

    if (mainOutput != juce::AudioChannelSet::mono()
        && mainOutput != juce::AudioChannelSet::stereo())
        return false;

    // Input can be mono (guitar) or stereo
    if (mainInput != juce::AudioChannelSet::mono()
        && mainInput != juce::AudioChannelSet::stereo())
        return false;

    return true;
}

void ToneMatchAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                           juce::MidiBuffer& /*midi*/)
{
    juce::ScopedNoDenormals noDenormals;

    // Copy mono input to stereo output if needed
    const int numInputChannels = getTotalNumInputChannels();
    const int numOutputChannels = getTotalNumOutputChannels();
    if (numInputChannels == 1 && numOutputChannels == 2)
    {
        // Copy mono input to both output channels
        buffer.copyFrom(1, 0, buffer, 0, 0, buffer.getNumSamples());
    }
    else
    {
        // Clear extra output channels
        for (int ch = numInputChannels; ch < numOutputChannels; ++ch)
            buffer.clear(ch, 0, buffer.getNumSamples());
    }

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

    // Early exit if no samples
    if (numSamples == 0 || numChannels == 0)
        return;

    // #region agent log
    dbgBlockCount++;
    float inputPeakCh0 = peakOf(buffer, 0);
    float inputPeakCh1 = (numChannels > 1) ? peakOf(buffer, 1) : 0.f;
    float inputMax = std::max(inputPeakCh0, inputPeakCh1);
    float inputDbCh0 = peakToDb(inputPeakCh0);
    float inputDbCh1 = peakToDb(inputPeakCh1);
    float inputDbMax = peakToDb(inputMax);
    dbgLog("processBlock:ENTRY", "input signal dB", "A",
           inputDbCh0, inputDbCh1, inputDbMax,
           getTotalNumInputChannels(), getTotalNumOutputChannels());
    
    // Check if input is too quiet (likely routing issue)
    static int quietWarningCount = 0;
    if (inputDbMax < -60.0f && (quietWarningCount % 1000) == 0) {
        dbgLog("processBlock:WARNING", "input too quiet", "G",
               inputDbMax, 0, 0, 0, 0);
        quietWarningCount++;
    }
    // #endregion

    // ── Conditional input normalization (only after Match Tone) ────────────────
    // Normalize input signal to -1dB only if inputNormalized flag is set.
    // This flag is set after successful Match Tone, when gain was calculated for normalized DI.
    // For normal operation (clean sound, manual presets), input is NOT normalized.
    bool shouldNormalize = inputNormalized.load(std::memory_order_acquire);
    // #region agent log
    static int normalizationLogCount = 0;
    if ((normalizationLogCount % 1000) == 0)
    {
        dbgLog("processBlock:NORMALIZATION_FLAG", "input normalization flag state", "N",
               shouldNormalize ? 1.0f : 0.0f, inputDbMax, 0, 0, 0);
        normalizationLogCount++;
    }
    // #endregion
    
    if (shouldNormalize)
    {
        if (inputMax > 1e-6f) // Only if not too quiet
        {
            const float targetLinear = juce::Decibels::decibelsToGain(-1.0f); // ≈ 0.89125
            const float normalizeGain = targetLinear / inputMax;
            
            // Limit normalization gain to prevent excessive amplification (max +40dB)
            const float maxNormalizeGainDb = 40.0f;
            const float maxNormalizeGainLinear = juce::Decibels::decibelsToGain(maxNormalizeGainDb);
            const float clampedNormalizeGain = juce::jmin(normalizeGain, maxNormalizeGainLinear);
            
            // Apply normalization to all channels
            for (int ch = 0; ch < numChannels; ++ch)
                buffer.applyGain(ch, 0, numSamples, clampedNormalizeGain);
            
            // #region agent log
            float normalizeGainDb = 20.0f * std::log10(clampedNormalizeGain);
            float normalizedPeak = peakOf(buffer, 0);
            float normalizedDb = peakToDb(normalizedPeak);
            dbgLog("processBlock:INPUT_NORMALIZED", "input normalized to -1dB (after Match)", "N",
                   inputDbMax, normalizeGainDb, normalizedDb, 1, 0);
            // #endregion
        }
    }

    // Read parameters from APVTS (lock-free)
    const float overdriveDb = apvts.getRawParameterValue("overdrive")->load();
    const float gainDb       = apvts.getRawParameterValue("inputGain")->load();
    const float eqGainDb    = apvts.getRawParameterValue("preEqGainDb")->load();
    const float eqFreq      = apvts.getRawParameterValue("preEqFreqHz")->load();
    // #region agent log
    float inputDbBeforeOverdrive = peakToDb(peakOf(buffer, 0));
    dbgLog("processBlock:PARAMS", "params and input", "H",
           overdriveDb, inputDbBeforeOverdrive, eqGainDb, 0, 0);
    // #endregion
    const float revWet      = apvts.getRawParameterValue("reverbWet")->load();
    const float revRoom     = apvts.getRawParameterValue("reverbRoomSize")->load();
    const float delTime     = apvts.getRawParameterValue("delayTimeMs")->load();
    const float delMixVal   = apvts.getRawParameterValue("delayMix")->load();
    const float hpfFreq     = apvts.getRawParameterValue("hpfFreq")->load();
    const float lpfFreq     = apvts.getRawParameterValue("lpfFreq")->load();

    juce::dsp::AudioBlock<float> block(buffer);

    // ── Stage 1: Overdrive (before NAM, for overdrive/distortion) ────────────
    overdrive_gain.setGainDecibels(overdriveDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        overdrive_gain.process(ctx);
    }
    // #region agent log
    float afterOverdrivePeak = peakOf(buffer, 0);
    float afterOverdriveDb = peakToDb(afterOverdrivePeak);
    dbgLog("processBlock:AFTER_OVERDRIVE", "after overdrive dB", "O",
           afterOverdriveDb, overdriveDb, afterOverdrivePeak, 0, 0);
    // #endregion

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

    // #region agent log
    dbgLog("processBlock:AFTER_EQ", "after pre-eq", "C",
           peakOf(buffer, 0), eqGainDb, eqFreq);
    // #endregion

    // ── Stage 2: NAM Pedal ───────────────────────────────────────────────────
    {
        bool pedalLoaded = pedal.isModelLoaded();
        // #region agent log
        dbgLog("processBlock:NAM_PEDAL", "pedal check", "D",
               peakOf(buffer, 0), 0, 0, pedalLoaded ? 1 : 0);
        // #endregion
        if (pedalLoaded)
            pedal.process(block);
        // #region agent log
        dbgLog("processBlock:AFTER_PEDAL", "after pedal", "D",
               peakOf(buffer, 0), 0, 0, pedalLoaded ? 1 : 0);
        // #endregion
    }

    // ── Stage 3: NAM Amp ─────────────────────────────────────────────────────
    {
        bool ampLoaded = amp.isModelLoaded();
        // #region agent log
        dbgLog("processBlock:NAM_AMP", "amp check", "D",
               peakOf(buffer, 0), 0, 0, ampLoaded ? 1 : 0);
        // #endregion
        if (ampLoaded)
            amp.process(block);
        // #region agent log
        dbgLog("processBlock:AFTER_AMP", "after amp", "D",
               peakOf(buffer, 0), 0, 0, ampLoaded ? 1 : 0);
        // #endregion
    }

    // ── Stage 4: IR Cabinet (Convolution) ────────────────────────────────────
    {
        int irSize = ir_cabinet.getCurrentIRSize();
        // #region agent log
        dbgLog("processBlock:IR_CAB", "ir check", "E",
               peakOf(buffer, 0), 0, 0, irSize);
        // #endregion
        if (irSize > 0)
        {
            juce::dsp::ProcessContextReplacing<float> ctx(block);
            ir_cabinet.process(ctx);
        }
        // #region agent log
        dbgLog("processBlock:AFTER_IR", "after ir", "E",
               peakOf(buffer, 0), 0, 0, irSize);
        // #endregion
    }

    // ── Stage 4b: HPF (High-Pass Filter) ───────────────────────────────────────
    {
        *hpf_coeffs = *juce::dsp::IIR::Coefficients<float>::makeHighPass(
            currentSampleRate, juce::jlimit(20.0f, 20000.0f, hpfFreq));

        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            hpf_filter.process(ctx);

            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 4c: LPF (Low-Pass Filter) ────────────────────────────────────────
    {
        *lpf_coeffs = *juce::dsp::IIR::Coefficients<float>::makeLowPass(
            currentSampleRate, juce::jlimit(20.0f, 20000.0f, lpfFreq));

        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            lpf_filter.process(ctx);

            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }
    // #region agent log
    dbgLog("processBlock:AFTER_FILTERS", "after hpf+lpf", "F",
           peakOf(buffer, 0), hpfFreq, lpfFreq);
    // #endregion

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

    // ── Stage 7: Input Gain (volume control, applied at the end) ──────────────
    input_gain.setGainDecibels(gainDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        input_gain.process(ctx);
    }
    // #region agent log
    float afterVolumeGainPeak = peakOf(buffer, 0);
    float afterVolumeGainDb = peakToDb(afterVolumeGainPeak);
    dbgLog("processBlock:AFTER_VOLUME_GAIN", "after volume gain dB", "B",
           afterVolumeGainDb, gainDb, afterVolumeGainPeak, 0, 0);
    // #endregion

    // ── Stage 8: Safety Limiter ──────────────────────────────────────────────
    // Prevent harsh digital clipping if gain is set too high
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            // Soft clipping (tanh-like) for safety
            float val = data[i];
            if (val > 1.0f) val = 1.0f;
            else if (val < -1.0f) val = -1.0f;
            data[i] = val;
        }
    }

    // #region agent log
    dbgLog("processBlock:FINAL_OUTPUT", "final output after all", "G",
           peakOf(buffer, 0), revWet, delMixVal);
    // #endregion
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
    {
        // Reset input normalization flag when loading state
        // (state loading typically means manual preset change)
        inputNormalized.store(false, std::memory_order_release);
        PresetManager::varToState(parsed, apvts, *this);
    }
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

void ToneMatchAudioProcessor::syncLockStates()
{
    if (auto* aiLockParam = apvts.getRawParameterValue("aiLock"))
        aiLockEnabled.store(aiLockParam->load() > 0.5f, std::memory_order_release);
    
    if (auto* cabLockParam = apvts.getRawParameterValue("cabLock"))
        cabLockEnabled.store(cabLockParam->load() > 0.5f, std::memory_order_release);
}

void ToneMatchAudioProcessor::triggerMatch(const juce::File& refFile)

{

    dbgLog("triggerMatch:START", "triggerMatch called", "J", 0, 0, 0, refFile.existsAsFile() ? 1 : 0, 0, true);

    DBG("[ToneMatch] triggerMatch called with file: " + refFile.getFullPathName());
    dbgLog("triggerMatch:REF_FILE", ("Ref file: " + refFile.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, refFile.existsAsFile() ? 1 : 0, refFile.getSize(), true);

    // Validate reference file FIRST
    if (!refFile.existsAsFile())
    {
        DBG("[ToneMatch] ERROR: Reference file does not exist: " + refFile.getFullPathName());
        dbgLog("triggerMatch:REF_NOT_FOUND", ("Reference file not found: " + refFile.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: Reference file not found!");
        return;
    }

    // Check file size - empty files are invalid
    if (refFile.getSize() == 0)
    {
        DBG("[ToneMatch] ERROR: Reference file is empty: " + refFile.getFullPathName());
        dbgLog("triggerMatch:REF_EMPTY", ("Reference file is empty: " + refFile.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: Reference file is empty!");
        return;
    }
    
    dbgLog("triggerMatch:REF_VALID", "Reference file validated", "J", 0, 0, 0, refFile.getSize(), 0, true);

    if (pythonBridge.isRunning())

    {

        dbgLog("triggerMatch:ALREADY_RUNNING", "Python bridge already running", "J", 0, 0, 0, 0, 0, true);

        DBG("[ToneMatch] Python bridge already running, ignoring");

        return;

    }



    // Save the captured DI buffer to a temp .wav file

    juce::File diTemp = juce::File::getSpecialLocation(juce::File::tempDirectory)

                            .getChildFile("tonematch_di.wav");

    // Delete old file if exists to ensure clean write
    if (diTemp.existsAsFile())
    {
        diTemp.deleteFile();
    }

    // Get the actual number of samples captured

    int numCapturedSamples = capturedDIWritePos.load(std::memory_order_acquire);

    DBG("[ToneMatch] Captured DI samples: " + juce::String(numCapturedSamples));
    dbgLog("triggerMatch:DI_SAMPLES", "Captured DI samples", "J", 0, 0, 0, numCapturedSamples, getSampleRate(), true);

    

    // If no DI captured, warn the user

    if (numCapturedSamples == 0)

    {

        DBG("[ToneMatch] No DI audio captured - please record some DI first");
        dbgLog("triggerMatch:NO_DI", "No DI audio captured", "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: Record DI first!");

        return;

    }

    // Check if captured audio is not all zeros (very quiet audio)
    float peakLinear = capturedDI.getMagnitude(0, 0, numCapturedSamples);
    float peakDb = juce::Decibels::gainToDecibels(peakLinear, -100.0f);
    dbgLog("triggerMatch:DI_PEAK", "DI peak level", "J", peakLinear, peakDb, 0, numCapturedSamples, 0, true);
    
    if (peakLinear < 1e-6f) // Very quiet or silent
    {
        DBG("[ToneMatch] WARNING: Captured DI is very quiet (peak: " + 
            juce::String(peakDb, 1) + " dB)");
        dbgLog("triggerMatch:DI_QUIET", "DI is very quiet", "J", peakLinear, peakDb, 0, 0, 0, true);
        // Continue anyway - might be intentional
    }

    // Normalize captured DI to -1dB before writing to file
    // Python optimizer expects DI normalized to -1dB (as done in src/core/io.py)
    // This ensures the optimizer calculates inputGainDb correctly for overdrive
    if (peakLinear > 1e-6f) // Only if not too quiet
    {
        const float targetLinear = juce::Decibels::decibelsToGain(-1.0f); // ≈ 0.89125
        const float normalizeGain = targetLinear / peakLinear;
        
        // Apply normalization to capturedDI
        capturedDI.applyGain(0, 0, numCapturedSamples, normalizeGain);
        
        // Update capturedDIPeakDb to -1.0f (normalized level)
        capturedDIPeakDb = -1.0f;
        
        float normalizeGainDb = 20.0f * std::log10(normalizeGain);
        DBG("[ToneMatch] Normalized DI to -1dB (gain: " + 
            juce::String(normalizeGainDb, 1) + " dB, original peak: " + 
            juce::String(peakDb, 1) + " dB)");
        dbgLog("triggerMatch:DI_NORMALIZED", "DI normalized to -1dB", "J", 
               capturedDIPeakDb, normalizeGainDb, peakDb, 0, 0, true);
    }
    else
    {
        // If too quiet, use raw level but warn
        capturedDIPeakDb = peakDb;
        DBG("[ToneMatch] WARNING: DI too quiet to normalize, using raw level: " + 
            juce::String(peakDb, 1) + " dB");
        dbgLog("triggerMatch:DI_NOT_NORMALIZED", "DI too quiet, using raw level", "J", 
               capturedDIPeakDb, peakLinear, 0, 0, 0, true);
    }

    // Write captured mono DI to file (only the captured portion)
    dbgLog("triggerMatch:WRITING_DI", ("Writing DI to: " + diTemp.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, numCapturedSamples, getSampleRate(), true);
    
    bool diFileWritten = false;
    if (auto writer = std::unique_ptr<juce::AudioFormatWriter>(
            juce::WavAudioFormat().createWriterFor(
                new juce::FileOutputStream(diTemp),
                getSampleRate(), 1, 16, {}, 0)))
    {
        writer->writeFromAudioSampleBuffer(capturedDI, 0, numCapturedSamples);
        // Data will be written when writer is destroyed (RAII)
        
        DBG("[ToneMatch] Saved " + juce::String(numCapturedSamples) + " samples to " + diTemp.getFullPathName());
        
        // capturedDIPeakDb is already set during normalization (or raw level if too quiet)
        // No need to recalculate here
        
        diFileWritten = true;
        dbgLog("triggerMatch:DI_WRITTEN", "DI file written successfully", "J", capturedDIPeakDb, 0, 0, numCapturedSamples, 0, true);
    }
    else
    {
        DBG("[ToneMatch] Failed to create WAV writer for DI file");
        dbgLog("triggerMatch:DI_WRITE_FAILED", ("Failed to create WAV writer: " + diTemp.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: Failed to save DI");
        return;
    }

    // CRITICAL: Verify DI file was actually written
    if (!diTemp.existsAsFile())
    {
        DBG("[ToneMatch] ERROR: DI file was not created: " + diTemp.getFullPathName());
        dbgLog("triggerMatch:DI_NOT_CREATED", ("DI file not created: " + diTemp.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: Failed to create DI file");
        return;
    }

    juce::int64 diFileSize = diTemp.getSize();
    if (diFileSize == 0)
    {
        DBG("[ToneMatch] ERROR: DI file is empty: " + diTemp.getFullPathName());
        dbgLog("triggerMatch:DI_EMPTY", ("DI file is empty: " + diTemp.getFullPathName()).toRawUTF8(), "J", 0, 0, 0, 0, 0, true);
        setProgressStage(0, "Error: DI file is empty");
        return;
    }

    DBG("[ToneMatch] DI file verified: " + juce::String(diFileSize) + " bytes");
    dbgLog("triggerMatch:DI_VERIFIED", "DI file verified", "J", 0, 0, 0, diFileSize, 0, true);

    // Set initial progress state
    DBG("[ToneMatch] Setting progress stage to 1 (Grid Search)");
    setProgressStage(1, "Grid Search...");

    dbgLog("triggerMatch:STARTING_BRIDGE", "Starting Python bridge", "J", 0, 0, 0, 0, 0, true);
    DBG("[ToneMatch] Starting Python bridge...");
    DBG("[ToneMatch] DI file: " + diTemp.getFullPathName() + " (" + juce::String(diTemp.getSize()) + " bytes)");
    DBG("[ToneMatch] Ref file: " + refFile.getFullPathName() + " (" + juce::String(refFile.getSize()) + " bytes)");
    dbgLog("triggerMatch:BRIDGE_START", "Calling pythonBridge.startMatch", "J", 0, 0, 0, diTemp.getSize(), refFile.getSize(), true);
    
    pythonBridge.startMatch(diTemp, refFile,
        [this](const MatchResult& result) { 
            dbgLog("triggerMatch:CALLBACK", "Python bridge callback received", "J", 0, 0, 0, result.success ? 1 : 0, 0, true);
            DBG("[ToneMatch] Python bridge callback received");
            onMatchComplete(result); 
        });
}

void ToneMatchAudioProcessor::onMatchComplete(const MatchResult& result)
{
    // This runs on the message thread
    // #region agent log
    dbgLog("onMatchComplete:START", "match complete", "J",
           0, 0, 0, result.success ? 1 : 0, 0, true);
    // #endregion
    
    if (! result.success)
    {
        DBG("Match failed: " + result.errorMessage);
        setProgressStage(0, "Error: " + result.errorMessage);
        // #region agent log
        dbgLog("onMatchComplete:FAIL", "match failed", "J", 0, 0, 0, 0, 0, true);
        // #endregion
        return;
    }
    
    // Update progress to optimizing stage (if not already done)
    setProgressStage(2, "Optimizing...");

    // Build RigParameters from MatchResult
    RigParameters params;
    params.fx_path = result.fxNamPath;
    params.amp_path = result.ampNamPath;
    params.ir_path = result.irPath;
    
    // #region agent log
    bool fxExists = params.fx_path.isNotEmpty() ? juce::File(params.fx_path).existsAsFile() : false;
    bool ampExists = params.amp_path.isNotEmpty() ? juce::File(params.amp_path).existsAsFile() : false;
    bool irExists = params.ir_path.isNotEmpty() ? juce::File(params.ir_path).existsAsFile() : false;
    dbgLog("onMatchComplete:PATHS", "match paths", "J",
           0, 0, 0, (fxExists ? 1 : 0) | ((ampExists ? 1 : 0) << 1) | ((irExists ? 1 : 0) << 2), 0, true);
    // #endregion
    
    params.reverb_wet = result.reverbWet;
    params.reverb_room_size = result.reverbRoomSize;
    params.delay_time_ms = result.delayTimeMs;
    params.delay_mix = result.delayMix;
    
    // Apply gain compensation: result.inputGainDb is relative to a DI normalized to -1dB.
    // After normalization in triggerMatch(), capturedDIPeakDb should be -1.0f,
    // so compensationDb should be ~0.0f. We keep this for safety/backward compatibility.
    float compensationDb = -1.0f - capturedDIPeakDb;
    // Limit compensation to a reasonable range (max +60dB) to allow sufficient gain for overdrive
    compensationDb = juce::jlimit(-12.0f, 60.0f, compensationDb);
    
    // After DI normalization, capturedDIPeakDb should be -1.0f, so compensation should be 0.0f
    // result.inputGainDb becomes overdrive_db (applied before NAM for overdrive)
    params.overdrive_db = result.inputGainDb + compensationDb;
    
    // Input gain (volume) is set to 0dB after Match Tone
    // User can adjust it manually if needed, but it doesn't affect overdrive
    params.input_gain_db = 0.0f;
    
    // Verify compensation is correct (should be ~0dB after normalization)
    if (std::abs(compensationDb) > 1.0f)
    {
        DBG("[ToneMatch] WARNING: Gain compensation is " + juce::String(compensationDb, 1) + 
            " dB (expected ~0dB). DI peak was " + juce::String(capturedDIPeakDb, 1) + " dB.");
    }
    params.pre_eq_gain_db = result.preEqGainDb;
    params.pre_eq_freq_hz = result.preEqFreqHz;
    
    // #region agent log - Gain compensation diagnostics
    // Log DI peak level, compensation, optimizer result, and final gain
    // After normalization, capturedDIPeakDb should be -1.0f and compensationDb should be ~0.0f
    dbgLog("onMatchComplete:GAIN_COMP", "gain compensation", "L",
           capturedDIPeakDb, compensationDb, result.inputGainDb,
           0, 0, true);
    DBG("[ToneMatch] Gain compensation: DI peak=" + juce::String(capturedDIPeakDb, 1) + 
        " dB, compensation=" + juce::String(compensationDb, 1) + 
        " dB, optimizer gain=" + juce::String(result.inputGainDb, 1) + " dB");
    dbgLog("onMatchComplete:OVERDRIVE_FINAL", "final overdrive", "L",
           params.overdrive_db, 0, 0, 0, 0, true);
    DBG("[ToneMatch] Final overdrive: " + juce::String(params.overdrive_db, 1) + " dB");
    dbgLog("onMatchComplete:VOLUME_GAIN_FINAL", "final volume gain", "L",
           params.input_gain_db, 0, 0, 0, 0, true);
    DBG("[ToneMatch] Final volume gain: " + juce::String(params.input_gain_db, 1) + " dB (set to 0 after Match)");
    
    // Log expected behavior after normalization
    if (std::abs(capturedDIPeakDb + 1.0f) < 0.1f) // DI was normalized to -1dB
    {
        DBG("[ToneMatch] DI was normalized to -1dB, compensation should be ~0dB");
        if (std::abs(compensationDb) > 0.5f)
        {
            DBG("[ToneMatch] WARNING: Compensation is not ~0dB despite normalized DI!");
        }
    }
    // #endregion
    
    // #region agent log - Check if parameters are always the same
    dbgLog("onMatchComplete:PARAMS", "match params", "K",
           params.overdrive_db, params.pre_eq_gain_db, params.pre_eq_freq_hz,
           (int)(params.reverb_wet * 100), (int)(params.delay_mix * 100), true);
    // #endregion
    // Use default values for HPF/LPF (will be controlled by UI)
    params.hpf_freq = 70.0f;
    params.lpf_freq = 8000.0f;
    // Preserve current lock states
    params.ai_lock = aiLockEnabled.load(std::memory_order_acquire);
    params.cab_lock = cabLockEnabled.load(std::memory_order_acquire);

    // Store last match result for UI display
    lastAmpName = result.ampNamName;
    lastCabName = result.irName;

    // Apply the complete rig
    applyNewRig(params);
    
    // Set inputNormalized flag to true after successful Match Tone
    // This enables input normalization in processBlock, since gain was calculated for normalized DI
    inputNormalized.store(true, std::memory_order_release);
    DBG("[ToneMatch] Input normalization enabled (gain calculated for normalized DI)");
    dbgLog("onMatchComplete:INPUT_NORMALIZED_FLAG", "input normalization enabled", "N",
           0, 0, 0, 1, 0, true);
    
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
    // #region agent log
    bool aiLocked = aiLockEnabled.load(std::memory_order_acquire);
    dbgLog("applyNewRig:START", "apply new rig", "J",
           0, 0, 0, aiLocked ? 1 : 0, 0, true);
    // #endregion
    
    // Load NAM models and save paths (only if not locked)
    if (params.fx_path.isNotEmpty() && !aiLocked)
    {
        juce::File fxFile(params.fx_path);
        bool fileExists = fxFile.existsAsFile();
        // #region agent log
        dbgLog("applyNewRig:PEDAL_FILE", "pedal file check", "J",
               0, 0, 0, fileExists ? 1 : 0, 0, true);
        // #endregion
        
        // #region agent log
        juce::String fxPathStr = fxFile.getFullPathName();
        dbgLog("applyNewRig:PEDAL_LOAD_START", "pedal load start", "J",
               0, 0, 0, fxPathStr.length(), 0, true);
        // #endregion
        
        if (! pedal.loadModel(fxFile))
        {
            DBG("Failed to load pedal model: " + params.fx_path);
            // #region agent log
            dbgLog("applyNewRig:PEDAL_FAIL", "pedal load failed", "J", 0, 0, 0, 0, 0, true);
            // #endregion
        }
        else
        {
            currentFxPath = params.fx_path;
            bool isLoaded = pedal.isModelLoaded();
            DBG("Loaded pedal model: " + pedal.getModelName());
            // #region agent log
            dbgLog("applyNewRig:PEDAL_OK", "pedal loaded", "J",
                   0, 0, 0, isLoaded ? 1 : 0, 0, true);
            // #endregion
        }
    }
    else if (aiLocked)
    {
        // #region agent log
        dbgLog("applyNewRig:PEDAL_SKIP", "pedal skip (lock)", "J", 0, 0, 0, 0, 0, true);
        // #endregion
    }
    else
    {
        // #region agent log
        dbgLog("applyNewRig:PEDAL_EMPTY", "pedal path empty", "J", 0, 0, 0, 0, 0, true);
        // #endregion
    }

    if (params.amp_path.isNotEmpty() && !aiLocked)
    {
        juce::File ampFile(params.amp_path);
        bool fileExists = ampFile.existsAsFile();
        // #region agent log
        dbgLog("applyNewRig:AMP_FILE", "amp file check", "J",
               0, 0, 0, fileExists ? 1 : 0, 0, true);
        // #endregion
        
        // #region agent log
        juce::String ampPathStr = ampFile.getFullPathName();
        dbgLog("applyNewRig:AMP_LOAD_START", "amp load start", "J",
               0, 0, 0, ampPathStr.length(), 0, true);
        // #endregion
        
        if (! amp.loadModel(ampFile))
        {
            DBG("Failed to load amp model: " + params.amp_path);
            // #region agent log
            dbgLog("applyNewRig:AMP_FAIL", "amp load failed", "J", 0, 0, 0, 0, 0, true);
            // #endregion
        }
        else
        {
            currentAmpPath = params.amp_path;
            bool isLoaded = amp.isModelLoaded();
            DBG("Loaded amp model: " + amp.getModelName());
            // #region agent log
            dbgLog("applyNewRig:AMP_OK", "amp loaded", "J",
                   0, 0, 0, isLoaded ? 1 : 0, 0, true);
            // #endregion
        }
    }
    else if (aiLocked)
    {
        // #region agent log
        dbgLog("applyNewRig:AMP_SKIP", "amp skip (lock)", "J", 0, 0, 0, 0, 0, true);
        // #endregion
    }
    else
    {
        // #region agent log
        dbgLog("applyNewRig:AMP_EMPTY", "amp path empty", "J", 0, 0, 0, 0, 0, true);
        // #endregion
    }

    // Load IR and save path (only if not locked)
    if (params.ir_path.isNotEmpty() && !cabLockEnabled.load(std::memory_order_acquire))
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
        {
            // Check if value is within parameter range
            auto range = apvts.getParameterRange(id);
            float clampedValue = juce::jlimit(range.start, range.end, value);
            if (clampedValue != value)
            {
                DBG("[applyNewRig] WARNING: Parameter " + id + " value " + 
                    juce::String(value, 2) + " dB was clamped to " + 
                    juce::String(clampedValue, 2) + " dB (range: " + 
                    juce::String(range.start, 2) + " to " + 
                    juce::String(range.end, 2) + " dB)");
                dbgLog("applyNewRig:PARAM_CLAMPED", ("Parameter " + id + " clamped").toRawUTF8(), "M",
                       value, clampedValue, range.start, (int)range.end, true);
            }
            p->setValueNotifyingHost(p->convertTo0to1(clampedValue));
        }
    };

    auto setBoolParam = [this](const juce::String& id, bool value)
    {
        if (auto* p = apvts.getParameter(id))
            p->setValueNotifyingHost(value ? 1.0f : 0.0f);
    };

    setParam("overdrive",      params.overdrive_db);
    setParam("inputGain",      params.input_gain_db);
    setParam("preEqGainDb",    params.pre_eq_gain_db);
    setParam("preEqFreqHz",    params.pre_eq_freq_hz);
    setParam("reverbWet",      params.reverb_wet);
    setParam("reverbRoomSize", params.reverb_room_size);
    setParam("delayTimeMs",    params.delay_time_ms);
    setParam("delayMix",       params.delay_mix);
    setParam("hpfFreq",        params.hpf_freq);
    setParam("lpfFreq",        params.lpf_freq);
    setBoolParam("aiLock",     params.ai_lock);
    setBoolParam("cabLock",    params.cab_lock);

    // Update lock states
    aiLockEnabled.store(params.ai_lock, std::memory_order_release);
    cabLockEnabled.store(params.cab_lock, std::memory_order_release);
}

//==============================================================================
bool ToneMatchAudioProcessor::loadPresetToProcessor(const juce::File& file)
{
    // Reset input normalization flag when loading preset manually
    // (presets don't assume normalized input)
    inputNormalized.store(false, std::memory_order_release);
    DBG("[ToneMatch] Input normalization disabled (preset loaded manually)");
    
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


