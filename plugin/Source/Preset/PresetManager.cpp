/*
  ==============================================================================
    PresetManager.cpp
    Save / load plugin state as .json preset files on disk.
  ==============================================================================
*/

#include "PresetManager.h"
#include "DSP/DSPChain.h"

//==============================================================================
PresetManager::PresetManager()  = default;
PresetManager::~PresetManager() = default;

//==============================================================================
bool PresetManager::savePreset(const juce::File& file,
                               const juce::AudioProcessorValueTreeState& apvts,
                               const DSPChain& chain) const
{
    auto data = stateToVar(apvts, chain);
    auto jsonText = juce::JSON::toString(data, true);   // pretty-print
    return file.replaceWithText(jsonText);
}

bool PresetManager::loadPreset(const juce::File& file,
                               juce::AudioProcessorValueTreeState& apvts,
                               DSPChain& chain) const
{
    if (! file.existsAsFile())
        return false;

    auto jsonText = file.loadFileAsString();
    auto data     = juce::JSON::parse(jsonText);

    if (! data.isObject())
        return false;

    varToState(data, apvts, chain);
    return true;
}

//==============================================================================
juce::var PresetManager::stateToVar(const juce::AudioProcessorValueTreeState& apvts,
                                    const DSPChain& chain)
{
    auto* obj = new juce::DynamicObject();

    // Preset name
    obj->setProperty("name", "User Preset");

    // Rig info
    auto* rig = new juce::DynamicObject();
    rig->setProperty("fx_nam",  chain.getPedalModelName());
    rig->setProperty("amp_nam", chain.getAmpModelName());
    rig->setProperty("ir",      chain.getIRName());
    // Paths are not stored from APVTS — they live in the chain.
    // For simplicity store names only; the user re-browses if paths differ.
    rig->setProperty("fx_nam_path",  "");
    rig->setProperty("amp_nam_path", "");
    rig->setProperty("ir_path",      "");
    obj->setProperty("rig", juce::var(rig));

    // Parameters
    auto* params = new juce::DynamicObject();

    auto readParam = [&](const juce::String& id) -> double
    {
        if (auto* p = apvts.getRawParameterValue(id))
            return static_cast<double>(p->load());
        return 0.0;
    };

    params->setProperty("input_gain_db",    readParam("inputGain"));
    params->setProperty("pre_eq_gain_db",   readParam("preEqGainDb"));
    params->setProperty("pre_eq_freq_hz",   readParam("preEqFreqHz"));
    params->setProperty("reverb_wet",       readParam("reverbWet"));
    params->setProperty("reverb_room_size", readParam("reverbRoomSize"));
    params->setProperty("delay_time_ms",    readParam("delayTimeMs"));
    params->setProperty("delay_mix",        readParam("delayMix"));
    params->setProperty("final_eq_gain_db", readParam("finalEqGainDb"));
    obj->setProperty("params", juce::var(params));

    obj->setProperty("loss", 0.0);

    return juce::var(obj);
}

void PresetManager::varToState(const juce::var& data,
                               juce::AudioProcessorValueTreeState& apvts,
                               DSPChain& chain)
{
    // ── Restore parameters ───────────────────────────────────────────────────
    if (auto* params = data.getProperty("params", {}).getDynamicObject())
    {
        auto setParam = [&](const juce::String& apvtsId, const juce::Identifier& jsonKey)
        {
            if (auto* p = apvts.getParameter(apvtsId))
            {
                float val = static_cast<float>((double) params->getProperty(jsonKey));
                p->setValueNotifyingHost(p->convertTo0to1(val));
            }
        };

        setParam("inputGain",     "input_gain_db");
        setParam("preEqGainDb",   "pre_eq_gain_db");
        setParam("preEqFreqHz",   "pre_eq_freq_hz");
        setParam("reverbWet",     "reverb_wet");
        setParam("reverbRoomSize","reverb_room_size");
        setParam("delayTimeMs",   "delay_time_ms");
        setParam("delayMix",      "delay_mix");
        setParam("finalEqGainDb", "final_eq_gain_db");
    }

    // ── Restore rig models ───────────────────────────────────────────────────
    if (auto* rig = data.getProperty("rig", {}).getDynamicObject())
    {
        juce::String fxPath  = rig->getProperty("fx_nam_path").toString();
        juce::String ampPath = rig->getProperty("amp_nam_path").toString();
        juce::String irPath  = rig->getProperty("ir_path").toString();

        if (fxPath.isNotEmpty())
            chain.loadPedalModel(juce::File(fxPath));

        if (ampPath.isNotEmpty())
            chain.loadAmpModel(juce::File(ampPath));

        if (irPath.isNotEmpty())
            chain.loadIR(juce::File(irPath));
    }
}

//==============================================================================
juce::File PresetManager::getDefaultPresetDirectory()
{
    auto dir = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory)
                   .getChildFile("ToneMatchAI")
                   .getChildFile("Presets");

    if (! dir.isDirectory())
        dir.createDirectory();

    return dir;
}


