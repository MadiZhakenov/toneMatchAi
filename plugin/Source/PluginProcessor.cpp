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
        juce::ParameterID("inputTrim", 1), "Input Trim",
        juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), 0.0f, "dB"));

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
        juce::NormalisableRange<float>(3000.0f, 8000.0f, 1.0f), 6000.0f, "Hz"));

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("aiLock", 1), "AI Lock", false));

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("cabLock", 1), "CAB Lock", false));

    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID("noiseGateEnabled", 1), "Noise Gate", true));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("noiseGateThreshold", 1), "Noise Gate Threshold",
        juce::NormalisableRange<float>(-100.0f, -40.0f, 0.1f), -90.0f, "dB"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("noiseGateAttack", 1), "Noise Gate Attack",
        juce::NormalisableRange<float>(0.001f, 0.050f, 0.001f), 0.005f, "s"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("noiseGateRelease", 1), "Noise Gate Release",
        juce::NormalisableRange<float>(0.010f, 1.000f, 0.010f), 0.100f, "s"));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID("noiseGateRange", 1), "Noise Gate Range",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f), -60.0f, "dB"));  // More aggressive default: -90dB

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
    
    // Reset cached parameters to force recalculation
    cachedPreEqGainDb = -999.0f;
    cachedPreEqFreq = -999.0f;
    cachedHpfFreq = -999.0f;
    cachedLpfFreq = -999.0f;
    cachedSampleRate = sampleRate;

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

    // Input Stage: DC Block & Safety HPF (80Hz, Q=0.7) - FIRST in chain
    inputSafetyHPFCoeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(
        sampleRate, 80.0f, 0.707f);  // 80Hz high-pass to remove rumble and low-frequency noise
    inputSafetyHPF.prepare(spec);
    inputSafetyHPF.coefficients = inputSafetyHPFCoeffs;

    // Pre-NAM Noise Gate (before any gain)
    preNAMNoiseGate.prepare(spec);
    preNAMNoiseGate.setThreshold(-65.0f);
    preNAMNoiseGate.setAttack(0.003f);
    preNAMNoiseGate.setRelease(0.100f);
    preNAMNoiseGate.setRange(-60.0f);

    // Compensation Gain (RMS-based auto-compensation)
    compensationGain.prepare(spec);
    compensationGain.setRampDurationSeconds(0.02);

    // Input Trim (manual user adjustment)
    inputTrimGain.prepare(spec);
    inputTrimGain.setRampDurationSeconds(0.02);

    // Aggressive high-pass filter for noise removal (DEPRECATED - will be removed)
    aggressive_hpf_coeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(
        sampleRate, 80.0f, 0.707f);  // 80Hz high-pass to remove rumble and low-frequency noise
    aggressive_hpf.prepare(spec);
    aggressive_hpf.coefficients = aggressive_hpf_coeffs;

    // Prepare DI capture buffer (store up to 30 seconds for matching)
    capturedDI.setSize(1, static_cast<int>(sampleRate * 30.0));
    capturedDI.clear();
    
    // Reset noise gate state
    noiseGateEnvelope = 0.0f;
}

void ToneMatchAudioProcessor::releaseResources()
{
    // Reset new input stage modules
    inputSafetyHPF.reset();
    preNAMNoiseGate.reset();
    compensationGain.reset();
    inputTrimGain.reset();
    
    // Reset existing modules
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
    aggressive_hpf.reset();  // Keep for now (deprecated)
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

    // Copy mono input to stereo output if needed (optimized)
    const int numInputChannels = getTotalNumInputChannels();
    const int numOutputChannels = getTotalNumOutputChannels();
    const int numSamples = buffer.getNumSamples();
    
    if (numInputChannels == 1 && numOutputChannels == 2)
    {
        // Copy mono input to both output channels (SIMD-optimized)
        buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
    }
    else if (numInputChannels < numOutputChannels)
    {
        // Clear extra output channels (optimized)
        for (int ch = numInputChannels; ch < numOutputChannels; ++ch)
            buffer.clear(ch, 0, numSamples);
    }

    // ── CAPTURE DI (before processing, with real-time normalization) ───────────
    if (capturing.load(std::memory_order_acquire))
    {
        const int numChannels = getTotalNumInputChannels();
        
        // Convert to mono and append to capturedDI
        int writePos = capturedDIWritePos.load(std::memory_order_relaxed);
        int available = capturedDI.getNumSamples() - writePos;
        int toWrite = juce::jmin(numSamples, available);
        
        if (toWrite > 0 && numChannels > 0)
        {
            // Calculate peak of incoming block for normalization
            float blockPeak = buffer.getMagnitude(0, 0, toWrite);
            
            // Sum to mono (or use channel 0 if mono) and normalize to -1dB in real-time
            const float targetPeakLinear = juce::Decibels::decibelsToGain(-1.0f); // ≈ 0.89125
            
            if (numChannels == 1)
            {
                if (blockPeak > 1e-6f)  // Only normalize if not silent
                {
                    // Copy and normalize in one step
                    float normalizeGain = targetPeakLinear / blockPeak;
                    for (int i = 0; i < toWrite; ++i)
                    {
                        float sample = buffer.getSample(0, i) * normalizeGain;
                        capturedDI.setSample(0, writePos + i, sample);
                    }
                }
                else
                {
                    // Silent - just copy zeros
                    capturedDI.copyFrom(0, writePos, buffer, 0, 0, toWrite);
                }
            }
            else
            {
                // Sum stereo to mono and normalize
                if (blockPeak > 1e-6f)
                {
                    float normalizeGain = targetPeakLinear / blockPeak;
                    for (int i = 0; i < toWrite; ++i)
                    {
                        float sum = 0.0f;
                        for (int ch = 0; ch < numChannels; ++ch)
                            sum += buffer.getSample(ch, i);
                        float mono = (sum / numChannels) * normalizeGain;
                        capturedDI.setSample(0, writePos + i, mono);
                    }
                }
                else
                {
                    // Silent - just set zeros
                    for (int i = 0; i < toWrite; ++i)
                        capturedDI.setSample(0, writePos + i, 0.0f);
                }
            }
            capturedDIWritePos.store(writePos + toWrite, std::memory_order_release);
        }
    }
    // ── END CAPTURE ───────────────────────────────────────────────────────────

    // ── Full DSP Chain Processing ─────────────────────────────────────────────
    const auto numChannels = buffer.getNumChannels();

    // Early exit if no samples
    if (numSamples == 0 || numChannels == 0)
        return;
    
    juce::dsp::AudioBlock<float> block(buffer);

    // ── NEW INPUT STAGE: Proper Gain Staging ──────────────────────────────────
    
    // Stage 1: DC Block & Safety HPF (80Hz, Q=0.7) - FIRST in chain
    // This removes network hum/rumble and low-frequency noise BEFORE any gain
    if (numChannels > 0)
    {
        auto monoBlock = block.getSingleChannelBlock(0);
        juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
        inputSafetyHPF.process(ctx);
        
        // Copy to other channels
        if (numChannels == 2)
            buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
        else
            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
    }

    // Stage 2: Pre-NAM Noise Gate - BEFORE any gain
    // This cuts audio interface noise in pauses before signal is boosted
    if (apvts.getRawParameterValue("noiseGateEnabled")->load() > 0.5f)
    {
        const float ngThreshold = apvts.getRawParameterValue("noiseGateThreshold")->load();
        const float ngAttack    = apvts.getRawParameterValue("noiseGateAttack")->load();
        const float ngRelease   = apvts.getRawParameterValue("noiseGateRelease")->load();
        const float ngRange     = apvts.getRawParameterValue("noiseGateRange")->load();

        preNAMNoiseGate.setThreshold(ngThreshold);
        preNAMNoiseGate.setAttack(ngAttack);
        preNAMNoiseGate.setRelease(ngRelease);
        preNAMNoiseGate.setRange(ngRange);

        juce::dsp::ProcessContextReplacing<float> ctx(block);
        preNAMNoiseGate.process(ctx);
    }

    // Stage 3: Compensation Gain - RMS-based auto-compensation
    // This applies the calculated compensation gain to match target RMS level
    {
        const float compensationLinear = inputCompensationLinear.load(std::memory_order_acquire);
        compensationGain.setGainLinear(compensationLinear);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        compensationGain.process(ctx);
    }

    // Stage 4: Input Trim - manual user adjustment after auto-compensation
    // This allows user to fine-tune level after auto-compensation, but before NAM
    {
        const float inputTrimDb = apvts.getRawParameterValue("inputTrim")->load();
        inputTrimGain.setGainDecibels(inputTrimDb);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        inputTrimGain.process(ctx);
    }

    // ── END NEW INPUT STAGE ───────────────────────────────────────────────────

    // Read parameters from APVTS (lock-free)
    const float overdriveDb = apvts.getRawParameterValue("overdrive")->load();
    const float gainDb       = apvts.getRawParameterValue("inputGain")->load();
    const float eqGainDb    = apvts.getRawParameterValue("preEqGainDb")->load();
    const float eqFreq      = apvts.getRawParameterValue("preEqFreqHz")->load();
    const float revWet      = apvts.getRawParameterValue("reverbWet")->load();
    const float revRoom     = apvts.getRawParameterValue("reverbRoomSize")->load();
    const float delTime     = apvts.getRawParameterValue("delayTimeMs")->load();
    const float delMixVal   = apvts.getRawParameterValue("delayMix")->load();
    const float hpfFreq     = apvts.getRawParameterValue("hpfFreq")->load();
    const float lpfFreq     = apvts.getRawParameterValue("lpfFreq")->load();

    // ── Stage 1: Overdrive (before NAM, for overdrive/distortion) ────────────
    overdrive_gain.setGainDecibels(overdriveDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        overdrive_gain.process(ctx);
    }

    // ── Stage 1b: Pre-EQ (parametric peak) ───────────────────────────────────
    {
        // Cache coefficients and only update when parameters change
        if (cachedPreEqGainDb != eqGainDb || cachedPreEqFreq != eqFreq || cachedSampleRate != currentSampleRate)
        {
            const float linearGain = juce::Decibels::decibelsToGain(eqGainDb);
            *pre_eq_coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
                currentSampleRate, juce::jlimit(20.0f, 20000.0f, eqFreq), 0.707f, linearGain);
            cachedPreEqGainDb = eqGainDb;
            cachedPreEqFreq = eqFreq;
            cachedSampleRate = currentSampleRate;
        }

        // Process mono (channel 0) through IIR; copy to other channels
        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            pre_eq_filter.process(ctx);

            // Optimized: use SIMD-friendly copy for stereo
            if (numChannels == 2)
                buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
            else
                for (int ch = 1; ch < numChannels; ++ch)
                    buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 2: NAM Pedal ───────────────────────────────────────────────────
    if (pedal.isModelLoaded())
        pedal.process(block);

    // ── Stage 3: NAM Amp ─────────────────────────────────────────────────────
    if (amp.isModelLoaded())
        amp.process(block);

    // ── Stage 4: IR Cabinet (Convolution) ────────────────────────────────────
    if (ir_cabinet.getCurrentIRSize() > 0)
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        ir_cabinet.process(ctx);
    }

    // ── Stage 4b: HPF (High-Pass Filter) ───────────────────────────────────────
    {
        // Cache coefficients and only update when parameters change
        if (cachedHpfFreq != hpfFreq || cachedSampleRate != currentSampleRate)
        {
            *hpf_coeffs = *juce::dsp::IIR::Coefficients<float>::makeHighPass(
                currentSampleRate, juce::jlimit(20.0f, 20000.0f, hpfFreq));
            cachedHpfFreq = hpfFreq;
            cachedSampleRate = currentSampleRate;
        }

        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            hpf_filter.process(ctx);

            if (numChannels == 2)
                buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
            else
                for (int ch = 1; ch < numChannels; ++ch)
                    buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 4c: LPF (Low-Pass Filter) ────────────────────────────────────────
    {
        // Cache coefficients and only update when parameters change
        if (cachedLpfFreq != lpfFreq || cachedSampleRate != currentSampleRate)
        {
            *lpf_coeffs = *juce::dsp::IIR::Coefficients<float>::makeLowPass(
                currentSampleRate, juce::jlimit(20.0f, 20000.0f, lpfFreq));
            cachedLpfFreq = lpfFreq;
            cachedSampleRate = currentSampleRate;
        }

        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            lpf_filter.process(ctx);

            if (numChannels == 2)
                buffer.copyFrom(1, 0, buffer, 0, 0, numSamples);
            else
                for (int ch = 1; ch < numChannels; ++ch)
                    buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 5: Delay ───────────────────────────────────────────────────────
    if (delMixVal > 0.001f)  // Skip if delay mix is effectively zero
    {
        const float delaySamples = (delTime / 1000.0f) * static_cast<float>(currentSampleRate);
        delay_line.setDelay(juce::jlimit(1.0f, static_cast<float>(currentSampleRate), delaySamples));

        const float dryMix = 1.0f - delMixVal;
        const float wetMix = delMixVal;

        // Optimized delay processing - vectorized where possible
        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                const float dry = data[i];
                delay_line.pushSample(ch, dry);
                const float wet = delay_line.popSample(ch);
                data[i] = dry * dryMix + wet * wetMix;
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

    // ── Stage 8: Safety Limiter ──────────────────────────────────────────────
    // Prevent harsh digital clipping if gain is set too high
    // Optimized: use SIMD-friendly clamping
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* data = buffer.getWritePointer(ch);
        juce::FloatVectorOperations::clip(data, data, -1.0f, 1.0f, numSamples);
    }
    
    // ── Stage 9: Final Soft Noise Gate Check (after all processing) ───────────
    // Final check: if output is extremely quiet, gently reduce it
    // This catches any noise that might have been generated in the processing chain
    float outputPeak = buffer.getMagnitude(0, 0, numSamples);
    float outputPeakDb = juce::Decibels::gainToDecibels(outputPeak, -100.0f);
    
    // If output is quieter than -60 dB, it's likely just noise - gently reduce it
    const float outputMuteThresholdDb = -60.0f;
    if (outputPeakDb < outputMuteThresholdDb)
    {
        // Soft reduction instead of hard mute
        float reductionFactor = juce::jmap(outputPeakDb, -100.0f, outputMuteThresholdDb, 0.0f, 1.0f);
        buffer.applyGain(reductionFactor);
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
    {
        // Reset compensation gain when loading state
        // (state loading typically means manual preset change)
        inputCompensationLinear.store(1.0f, std::memory_order_release);
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

    // ── RMS-based Compensation Calculation ────────────────────────────────────
    // Calculate RMS of captured DI signal (more accurate than peak for gain staging)
    float rmsLevel = capturedDI.getRMSLevel(0, 0, numCapturedSamples);
    float rmsDb = juce::Decibels::gainToDecibels(rmsLevel, -100.0f);
    
    // Target RMS level: -18.0 dB (industry standard for "hot" signal)
    const float targetRmsDb = -18.0f;
    
    // Calculate raw compensation: TargetRMS - UserRMS
    float rawCompensationDb = targetRmsDb - rmsDb;
    
    // SAFETY CLAMP: Limit compensation to maximum +18.0 dB
    // Logic: If signal is quieter than -36dB RMS, it's a bad signal.
    // We won't boost it by +40dB to avoid raising noise. We'll boost by +18dB max,
    // and the user can adjust the rest on their audio interface.
    float compensationDb = juce::jlimit(-60.0f, 18.0f, rawCompensationDb);
    
    // Convert to linear gain
    float compensationLinear = juce::Decibels::decibelsToGain(compensationDb);
    
    // Store compensation values
    inputCompensationLinear.store(compensationLinear, std::memory_order_release);
    autoCompensationDb.store(compensationDb, std::memory_order_release);
    
    // Also store peak for compatibility (used in onMatchComplete)
    float peakLinear = capturedDI.getMagnitude(0, 0, numCapturedSamples);
    float peakDb = juce::Decibels::gainToDecibels(peakLinear, -100.0f);
    capturedDIPeakDb = peakDb;
    
    // Logging
    dbgLog("triggerMatch:DI_RMS", "DI RMS level (RMS-based compensation)", "J", rmsLevel, rmsDb, compensationDb, numCapturedSamples, 0, true);
    
    if (rawCompensationDb != compensationDb)
    {
        DBG("[ToneMatch] WARNING: Compensation was clamped from " + 
            juce::String(rawCompensationDb, 1) + " dB to " + 
            juce::String(compensationDb, 1) + " dB (signal RMS: " + 
            juce::String(rmsDb, 1) + " dB)");
        dbgLog("triggerMatch:COMP_CLAMPED", "Compensation clamped", "J", rawCompensationDb, compensationDb, rmsDb, 0, 0, true);
    }
    
    if (rmsLevel < 1e-6f) // Very quiet or silent
    {
        DBG("[ToneMatch] WARNING: Captured DI is very quiet (RMS: " + 
            juce::String(rmsDb, 1) + " dB) - may not match well");
        dbgLog("triggerMatch:DI_QUIET", "DI is very quiet", "J", rmsLevel, rmsDb, 0, 0, 0, true);
    }
    else
    {
        DBG("[ToneMatch] DI captured - RMS: " + juce::String(rmsDb, 1) + 
            " dB, Target: " + juce::String(targetRmsDb, 1) + 
            " dB, Compensation: " + juce::String(compensationDb, 1) + " dB");
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

    // ── DETAILED PARAMETER LOGGING ─────────────────────────────────────────────
    // Log all parameters received from Python optimizer for diagnostics
    DBG("[onMatchComplete] ========== ALL PARAMETERS RECEIVED FROM PYTHON ==========");
    DBG("  Rig Info:");
    DBG("    - FX NAM: " + result.fxNamName + " (" + result.fxNamPath + ")");
    DBG("    - AMP NAM: " + result.ampNamName + " (" + result.ampNamPath + ")");
    DBG("    - IR: " + result.irName + " (" + result.irPath + ")");
    DBG("  DSP Parameters (from JSON):");
    DBG("    - inputGainDb (from optimizer): " + juce::String(result.inputGainDb, 2) + " dB");
    DBG("    - overdriveDb (explicit field): " + juce::String(result.overdriveDb, 2) + " dB");
    DBG("    - preEqGainDb: " + juce::String(result.preEqGainDb, 2) + " dB");
    DBG("    - preEqFreqHz: " + juce::String(result.preEqFreqHz, 2) + " Hz");
    DBG("    - reverbWet: " + juce::String(result.reverbWet, 3));
    DBG("    - reverbRoomSize: " + juce::String(result.reverbRoomSize, 3));
    DBG("    - delayTimeMs: " + juce::String(result.delayTimeMs, 2) + " ms");
    DBG("    - delayMix: " + juce::String(result.delayMix, 3));
    DBG("    - finalEqGainDb: " + juce::String(result.finalEqGainDb, 2) + " dB");
    DBG("    - loss: " + juce::String(result.loss, 6));
    DBG("  DI Capture State:");
    DBG("    - capturedDIPeakDb: " + juce::String(capturedDIPeakDb, 2) + " dB");
    DBG("================================================================");
    // #endregion

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
    
    // ── Calculate gain compensation ──────────────────────────────────────────
    // result.inputGainDb from Python optimizer is relative to a DI normalized to -1dB.
    // After normalization in triggerMatch(), capturedDIPeakDb should be -1.0f,
    // so compensationDb should be ~0.0f.
    // 
    // Compensation formula: compensationDb = targetLevelDb - actualLevelDb
    // where targetLevelDb = -1.0f (what Python optimizer expects)
    // and actualLevelDb = capturedDIPeakDb (what we actually have)
    const float targetLevelDb = -1.0f;
    float compensationDb = targetLevelDb - capturedDIPeakDb;
    
    // Verify that DI was normalized correctly
    // If capturedDIPeakDb is close to -1.0f, compensation should be ~0.0f
    const float normalizationTolerance = 0.1f;  // Allow 0.1 dB tolerance
    bool diWasNormalized = std::abs(capturedDIPeakDb - targetLevelDb) < normalizationTolerance;
    
    if (diWasNormalized)
    {
        // DI was normalized to -1dB, so compensation should be ~0dB
        // Force it to 0.0f to avoid floating point errors
        if (std::abs(compensationDb) < 0.5f)
        {
            compensationDb = 0.0f;
            DBG("[onMatchComplete] DI was normalized to -1dB, compensation set to 0.0 dB");
        }
        else
        {
            DBG("[onMatchComplete] WARNING: DI appears normalized but compensation is " + 
                juce::String(compensationDb, 2) + " dB (expected ~0 dB)");
        }
    }
    else
    {
        // DI was not normalized (too quiet), use calculated compensation
        DBG("[onMatchComplete] WARNING: DI was NOT normalized (peak: " + 
            juce::String(capturedDIPeakDb, 2) + " dB). Compensation: " + 
            juce::String(compensationDb, 2) + " dB");
    }
    
    // Limit compensation to a reasonable range (max +60dB) to allow sufficient gain for overdrive
    // This prevents extreme compensation values that could cause issues
    float compensationDbBeforeLimit = compensationDb;
    compensationDb = juce::jlimit(-12.0f, 60.0f, compensationDb);
    
    if (compensationDb != compensationDbBeforeLimit)
    {
        DBG("[onMatchComplete] WARNING: Compensation was clamped from " + 
            juce::String(compensationDbBeforeLimit, 2) + " dB to " + 
            juce::String(compensationDb, 2) + " dB");
    }
    
    // After DI normalization, capturedDIPeakDb should be -1.0f, so compensation should be 0.0f
    // Use overdriveDb if available (explicit field), otherwise fallback to inputGainDb
    // Both represent the same value from Python optimizer, but overdriveDb is more explicit
    float optimizerOverdriveDb = (result.overdriveDb != 0.0f || result.inputGainDb == 0.0f) 
                                  ? result.overdriveDb 
                                  : result.inputGainDb;
    params.overdrive_db = optimizerOverdriveDb + compensationDb;
    
    // ── DETAILED OVERDRIVE CALCULATION LOGGING ────────────────────────────────
    DBG("[onMatchComplete] ========== OVERDRIVE CALCULATION ==========");
    DBG("  Step 1: Python optimizer returned:");
    DBG("    - inputGainDb = " + juce::String(result.inputGainDb, 2) + " dB");
    DBG("    - overdriveDb = " + juce::String(result.overdriveDb, 2) + " dB");
    DBG("    - Using overdriveDb value: " + juce::String(optimizerOverdriveDb, 2) + " dB");
    DBG("  Step 2: DI was normalized to capturedDIPeakDb = " + juce::String(capturedDIPeakDb, 2) + " dB");
    DBG("  Step 3: Compensation calculated: compensationDb = -1.0 - " + 
        juce::String(capturedDIPeakDb, 2) + " = " + juce::String(compensationDb, 2) + " dB");
    DBG("  Step 4: Final overdrive_db = optimizerOverdriveDb + compensationDb = " + 
        juce::String(optimizerOverdriveDb, 2) + " + " + juce::String(compensationDb, 2) + 
        " = " + juce::String(params.overdrive_db, 2) + " dB");
    
    // Check if overdrive is suspiciously low for metal tones
    if (params.overdrive_db < 10.0f)
    {
        DBG("[onMatchComplete] WARNING: Final overdrive_db is very low (" + 
            juce::String(params.overdrive_db, 2) + " dB). For metal tones, expected 15-30 dB.");
        DBG("  This might indicate:");
        DBG("    1. Python optimizer did not find high enough input_gain_db");
        DBG("    2. Compensation calculation issue");
        DBG("    3. Reference tone might not be metal");
    }
    else if (params.overdrive_db >= 10.0f && params.overdrive_db < 15.0f)
    {
        DBG("[onMatchComplete] INFO: Overdrive_db is moderate (" + 
            juce::String(params.overdrive_db, 2) + " dB). May be sufficient for some tones.");
    }
    else
    {
        DBG("[onMatchComplete] INFO: Overdrive_db is high (" + 
            juce::String(params.overdrive_db, 2) + " dB). Should produce significant distortion.");
    }
    DBG("================================================================");
    // #endregion
    
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
    
    // ── FINAL PARAMETERS SUMMARY ──────────────────────────────────────────────
    DBG("[onMatchComplete] ========== FINAL PARAMETERS TO APPLY ==========");
    DBG("  Overdrive: " + juce::String(params.overdrive_db, 2) + " dB");
    DBG("  Input Gain (Volume): " + juce::String(params.input_gain_db, 2) + " dB");
    DBG("  Pre-EQ Gain: " + juce::String(params.pre_eq_gain_db, 2) + " dB @ " + 
        juce::String(params.pre_eq_freq_hz, 1) + " Hz");
    DBG("  Reverb: " + juce::String(params.reverb_wet * 100.0f, 1) + "% wet, room size " + 
        juce::String(params.reverb_room_size, 2));
    DBG("  Delay: " + juce::String(params.delay_mix * 100.0f, 1) + "% mix, " + 
        juce::String(params.delay_time_ms, 1) + " ms");
    DBG("  HPF: " + juce::String(params.hpf_freq, 1) + " Hz, LPF: " + juce::String(params.lpf_freq, 1) + " Hz");
    DBG("================================================================");
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
    
    // Note: Compensation gain was already calculated and stored in triggerMatch()
    // No need to set any flags here - compensation is applied automatically in processBlock()
    
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
    // Reset compensation gain when loading preset manually
    // (presets don't assume compensation gain)
    inputCompensationLinear.store(1.0f, std::memory_order_release);
    DBG("[ToneMatch] Compensation gain reset (preset loaded manually)");
    
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


