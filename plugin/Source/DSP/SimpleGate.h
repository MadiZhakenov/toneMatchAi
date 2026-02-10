/*
  ==============================================================================
    SimpleGate.h
    Robust Noise Gate implementation with hysteresis and smooth gain control.
  ==============================================================================
*/

#pragma once

#include <juce_dsp/juce_dsp.h>
#include <cmath>

class SimpleGate
{
public:
    SimpleGate() = default;

    void prepare(const juce::dsp::ProcessSpec& spec) {
        sampleRate = spec.sampleRate;
        updateCoefficients();
        reset();
    }

    void reset() {
        envelope = 0.0f;
        gain = 0.0f;
    }

    // Parameters setters
    void setThreshold(float thresholdDb) {
        thresholdLinear = juce::Decibels::decibelsToGain(thresholdDb);
    }
    
    void setAttack(float attackSeconds) {
        attackTime = attackSeconds;
        updateCoefficients();
    }
    
    void setRelease(float releaseSeconds) {
        releaseTime = releaseSeconds;
        updateCoefficients();
    }
    
    void setRange(float rangeDb) {
        // Range is the floor gain (e.g., -60dB)
        floorGain = juce::Decibels::decibelsToGain(rangeDb);
    }

    void process(juce::dsp::AudioBlock<float>& block) {
        // Safe check for valid sample rate
        if (sampleRate <= 0.0) return;
        
        const int numChannels = static_cast<int>(block.getNumChannels());
        const int numSamples = static_cast<int>(block.getNumSamples());
        
        for (int i = 0; i < numSamples; ++i) {
            float input = 0.0f;
            // Sum channels for detection (sidechain)
            for (int ch = 0; ch < numChannels; ++ch)
                input += std::abs(block.getSample(ch, i));
            
            if (numChannels > 1) input /= (float)numChannels;

            // Envelope follower
            if (input > envelope) 
                envelope = alphaAttack * envelope + (1.0 - alphaAttack) * input;
            else                  
                envelope = alphaRelease * envelope + (1.0 - alphaRelease) * input;

            // Gate logic
            // If signal (envelope) is above threshold, gain is 1.0 (open)
            // Otherwise, gain is floorGain (closed/attenuated)
            float targetGain = (envelope > thresholdLinear) ? 1.0f : floorGain;
            
            // Smooth the gain change to prevent clicking
            // 20ms smoothing standard
            const float gainSmoothAlpha = 0.999f; 
            gain = gainSmoothAlpha * gain + (1.0f - gainSmoothAlpha) * targetGain;

            // Apply to all channels
            for (int ch = 0; ch < numChannels; ++ch)
                block.setSample(ch, i, block.getSample(ch, i) * gain);
        }
    }
    
    // Support ContextReplacing
    template<typename ProcessContext>
    void process(const ProcessContext& context) {
        // Get non-const block from context
        auto& outBlock = const_cast<juce::dsp::AudioBlock<float>&>(context.getOutputBlock());
        process(outBlock);
    }

private:
    void updateCoefficients() {
        if (sampleRate > 0.0) {
            alphaAttack = std::exp(-1.0 / (sampleRate * attackTime));
            alphaRelease = std::exp(-1.0 / (sampleRate * releaseTime));
        }
    }

    double sampleRate = 44100.0;
    
    // State
    double envelope = 0.0f;
    float gain = 0.0f;
    
    // Coefficients
    double alphaAttack = 0.0f;
    double alphaRelease = 0.0f;
    
    // Parameters
    float attackTime = 0.005f; // 5ms default
    float releaseTime = 0.100f; // 100ms default
    
    float thresholdLinear = 0.0f;
    float floorGain = 0.0f;
};
