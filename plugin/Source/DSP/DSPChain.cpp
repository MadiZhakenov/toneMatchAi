/*
  ==============================================================================
    DSPChain.cpp
    Six-stage real-time signal chain implementation.
  ==============================================================================
*/

#include "DSPChain.h"

//==============================================================================
DSPChain::DSPChain()  = default;
DSPChain::~DSPChain() = default;

//==============================================================================
void DSPChain::prepare(const juce::dsp::ProcessSpec& spec)
{
    currentSampleRate = spec.sampleRate;
    currentBlockSize  = static_cast<int>(spec.maximumBlockSize);

    // 1. Input Gain
    inputGain.prepare(spec);
    inputGain.setRampDurationSeconds(0.02);

    // 2-3. NAM processors
    pedalNAM.prepare(spec);
    ampNAM.prepare(spec);

    // 4. IR Convolution
    irCabinet.prepare(spec);

    // 5. Delay Line
    delayLine.prepare(spec);
    delayLine.setMaximumDelayInSamples(static_cast<int>(spec.sampleRate));  // max 1 sec
    delayDryBuffer.setSize(static_cast<int>(spec.numChannels),
                           static_cast<int>(spec.maximumBlockSize));

    // 6. Reverb
    reverb.prepare(spec);

    // Pre-EQ filter
    preEqCoeffs = juce::dsp::IIR::Coefficients<float>::makePeakFilter(
        spec.sampleRate, 800.0f, 0.707f, 1.0f);
    preEqFilter.prepare(spec);
    preEqFilter.coefficients = preEqCoeffs;

    // Final Gain
    finalGain.prepare(spec);
    finalGain.setRampDurationSeconds(0.02);
}

void DSPChain::reset()
{
    inputGain.reset();
    pedalNAM.reset();
    ampNAM.reset();
    irCabinet.reset();
    delayLine.reset();
    reverb.reset();
    preEqFilter.reset();
    finalGain.reset();
}

//==============================================================================
void DSPChain::process(juce::AudioBuffer<float>& buffer)
{
    const auto numSamples  = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    // ── Read parameters (lock-free) ──────────────────────────────────────────
    const float gainDb      = readParam(inputGainDb, 0.0f);
    const float eqGainDb    = readParam(preEqGainDb, 0.0f);
    const float eqFreq      = readParam(preEqFreqHz, 800.0f);
    const float revWet      = readParam(reverbWet, 0.0f);
    const float revRoom     = readParam(reverbRoomSize, 0.5f);
    const float delTime     = readParam(delayTimeMs, 100.0f);
    const float delMixVal   = readParam(delayMix, 0.0f);
    const float finGainDb   = readParam(finalEqGainDb, 0.0f);

    juce::dsp::AudioBlock<float> block(buffer);

    // ── Stage 1: Input Gain ──────────────────────────────────────────────────
    inputGain.setGainDecibels(gainDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        inputGain.process(ctx);
    }

    // ── Stage 1b: Pre-EQ (parametric peak) ───────────────────────────────────
    {
        const float linearGain = juce::Decibels::decibelsToGain(eqGainDb);
        *preEqCoeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            currentSampleRate, juce::jlimit(20.0f, 20000.0f, eqFreq), 0.707f, linearGain);

        // Process mono (channel 0) through IIR; copy to other channels
        if (numChannels > 0)
        {
            auto monoBlock = block.getSingleChannelBlock(0);
            juce::dsp::ProcessContextReplacing<float> ctx(monoBlock);
            preEqFilter.process(ctx);

            for (int ch = 1; ch < numChannels; ++ch)
                buffer.copyFrom(ch, 0, buffer, 0, 0, numSamples);
        }
    }

    // ── Stage 2: NAM Pedal ───────────────────────────────────────────────────
    pedalNAM.process(block);

    // ── Stage 3: NAM Amp ─────────────────────────────────────────────────────
    ampNAM.process(block);

    // ── Stage 4: IR Cabinet (Convolution) ────────────────────────────────────
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        irCabinet.process(ctx);
    }

    // ── Stage 5: Delay ───────────────────────────────────────────────────────
    {
        // Save dry signal for wet/dry mixing
        for (int ch = 0; ch < numChannels; ++ch)
            delayDryBuffer.copyFrom(ch, 0, buffer, ch, 0, numSamples);

        const float delaySamples = (delTime / 1000.0f) * static_cast<float>(currentSampleRate);
        delayLine.setDelay(juce::jlimit(1.0f, static_cast<float>(currentSampleRate), delaySamples));

        for (int ch = 0; ch < numChannels; ++ch)
        {
            auto* data = buffer.getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                const float dry = data[i];
                delayLine.pushSample(ch, dry);
                const float wet = delayLine.popSample(ch);
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
        reverb.setParameters(reverbParams);

        juce::dsp::ProcessContextReplacing<float> ctx(block);
        reverb.process(ctx);
    }

    // ── Final Gain ───────────────────────────────────────────────────────────
    finalGain.setGainDecibels(finGainDb);
    {
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        finalGain.process(ctx);
    }
}

//==============================================================================
bool DSPChain::loadPedalModel(const juce::File& namFile)
{
    return pedalNAM.loadModel(namFile);
}

bool DSPChain::loadAmpModel(const juce::File& namFile)
{
    return ampNAM.loadModel(namFile);
}

bool DSPChain::loadIR(const juce::File& irFile)
{
    if (! irFile.existsAsFile())
        return false;

    irCabinet.loadImpulseResponse(irFile,
                                  juce::dsp::Convolution::Stereo::no,
                                  juce::dsp::Convolution::Trim::yes,
                                  0);   // 0 = use full IR
    currentIRName = irFile.getFileNameWithoutExtension();
    return true;
}

void DSPChain::clearPedalModel() { pedalNAM.clearModel(); }
void DSPChain::clearAmpModel()   { ampNAM.clearModel(); }
void DSPChain::clearIR()
{
    irCabinet.reset();
    currentIRName.clear();
}


















