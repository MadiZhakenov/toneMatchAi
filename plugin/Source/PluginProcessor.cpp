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
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
    // Wire APVTS atomic parameter pointers into the DSP chain
    dspChain.setInputGainParameter   (apvts.getRawParameterValue("inputGain"));
    dspChain.setPreEqGainParameter   (apvts.getRawParameterValue("preEqGainDb"));
    dspChain.setPreEqFreqParameter   (apvts.getRawParameterValue("preEqFreqHz"));
    dspChain.setReverbWetParameter   (apvts.getRawParameterValue("reverbWet"));
    dspChain.setReverbRoomSizeParameter(apvts.getRawParameterValue("reverbRoomSize"));
    dspChain.setDelayTimeParameter   (apvts.getRawParameterValue("delayTimeMs"));
    dspChain.setDelayMixParameter    (apvts.getRawParameterValue("delayMix"));
    dspChain.setFinalEqGainParameter (apvts.getRawParameterValue("finalEqGainDb"));
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

    dspChain.prepare(spec);

    // Prepare DI capture buffer (store up to 10 seconds for matching)
    capturedDI.setSize(1, static_cast<int>(sampleRate * 10.0));
    capturedDI.clear();
}

void ToneMatchAudioProcessor::releaseResources()
{
    dspChain.reset();
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

    dspChain.process(buffer);
}

//==============================================================================
juce::AudioProcessorEditor* ToneMatchAudioProcessor::createEditor()
{
    return new ToneMatchAudioProcessorEditor(*this);
}

//==============================================================================
void ToneMatchAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = PresetManager::stateToVar(apvts, dspChain);
    auto json  = juce::JSON::toString(state, false);
    destData.append(json.toRawUTF8(), json.getNumBytesAsUTF8());
}

void ToneMatchAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    auto json   = juce::String::fromUTF8(static_cast<const char*>(data), sizeInBytes);
    auto parsed = juce::JSON::parse(json);

    if (parsed.isObject())
        PresetManager::varToState(parsed, apvts, dspChain);
}

//==============================================================================
void ToneMatchAudioProcessor::triggerMatch(const juce::File& refFile)
{
    if (pythonBridge.isRunning())
        return;

    // Save the captured DI buffer to a temp .wav file
    juce::File diTemp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                            .getChildFile("tonematch_di.wav");

    // Write captured mono DI to file
    if (auto writer = std::unique_ptr<juce::AudioFormatWriter>(
            juce::WavAudioFormat().createWriterFor(
                new juce::FileOutputStream(diTemp),
                getSampleRate(), 1, 16, {}, 0)))
    {
        writer->writeFromAudioSampleBuffer(capturedDI, 0, capturedDI.getNumSamples());
    }

    pythonBridge.startMatch(diTemp, refFile,
        [this](const MatchResult& result) { onMatchComplete(result); });
}

void ToneMatchAudioProcessor::onMatchComplete(const MatchResult& result)
{
    // This runs on the message thread
    if (! result.success)
    {
        DBG("Match failed: " + result.errorMessage);
        return;
    }

    // Apply DSP parameters via APVTS
    auto setParam = [this](const juce::String& id, float value)
    {
        if (auto* p = apvts.getParameter(id))
            p->setValueNotifyingHost(p->convertTo0to1(value));
    };

    setParam("inputGain",     result.inputGainDb);
    setParam("preEqGainDb",   result.preEqGainDb);
    setParam("preEqFreqHz",   result.preEqFreqHz);
    setParam("reverbWet",     result.reverbWet);
    setParam("reverbRoomSize",result.reverbRoomSize);
    setParam("delayTimeMs",   result.delayTimeMs);
    setParam("delayMix",      result.delayMix);
    setParam("finalEqGainDb", result.finalEqGainDb);

    // Load rig models
    if (result.fxNamPath.isNotEmpty())
        dspChain.loadPedalModel(juce::File(result.fxNamPath));

    if (result.ampNamPath.isNotEmpty())
        dspChain.loadAmpModel(juce::File(result.ampNamPath));

    if (result.irPath.isNotEmpty())
        dspChain.loadIR(juce::File(result.irPath));

    DBG("Match complete! Loss = " + juce::String(result.loss, 6));
}

//==============================================================================
// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new ToneMatchAudioProcessor();
}


