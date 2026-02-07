/*
  ==============================================================================
    PresetManager.cpp
    Save / load plugin state as .json preset files on disk.
  ==============================================================================
*/

#include "PresetManager.h"
#include "PluginProcessor.h"

//==============================================================================
PresetManager::PresetManager()  = default;
PresetManager::~PresetManager() = default;

//==============================================================================
bool PresetManager::savePreset(const juce::File& file,
                               const juce::AudioProcessorValueTreeState& apvts,
                               const ToneMatchAudioProcessor& processor) const
{
    auto data = stateToVar(apvts, processor);
    auto jsonText = juce::JSON::toString(data, true);   // pretty-print
    return file.replaceWithText(jsonText);
}

bool PresetManager::loadPreset(const juce::File& file,
                               juce::AudioProcessorValueTreeState& apvts,
                               ToneMatchAudioProcessor& processor) const
{
    if (! file.existsAsFile())
        return false;

    auto jsonText = file.loadFileAsString();
    auto data     = juce::JSON::parse(jsonText);

    if (! data.isObject())
        return false;

    varToState(data, apvts, processor);
    return true;
}

//==============================================================================
juce::var PresetManager::stateToVar(const juce::AudioProcessorValueTreeState& apvts,
                                    const ToneMatchAudioProcessor& processor)
{
    auto* obj = new juce::DynamicObject();

    // Preset name
    obj->setProperty("name", "User Preset");

    // Rig info
    auto* rig = new juce::DynamicObject();
    rig->setProperty("fx_nam",  processor.getPedalModelName());
    rig->setProperty("amp_nam", processor.getAmpModelName());
    rig->setProperty("ir",      processor.getIRName());
    // Store full paths for proper preset loading
    rig->setProperty("fx_nam_path",  processor.getCurrentFxPath());
    rig->setProperty("amp_nam_path", processor.getCurrentAmpPath());
    rig->setProperty("ir_path",      processor.getCurrentIrPath());
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
                               ToneMatchAudioProcessor& processor)
{
    // Build RigParameters from JSON data
    RigParameters params;

    // ── Read rig paths ───────────────────────────────────────────────────────
    if (auto* rig = data.getProperty("rig", {}).getDynamicObject())
    {
        params.fx_path  = rig->getProperty("fx_nam_path").toString();
        params.amp_path = rig->getProperty("amp_nam_path").toString();
        params.ir_path  = rig->getProperty("ir_path").toString();
    }

    // ── Read parameters ─────────────────────────────────────────────────────
    if (auto* paramsObj = data.getProperty("params", {}).getDynamicObject())
    {
        auto readFloat = [&](const juce::Identifier& key, float defaultValue) -> float
        {
            auto val = paramsObj->getProperty(key);
            if (val.isDouble() || val.isInt())
                return static_cast<float>((double) val);
            return defaultValue;
        };

        params.input_gain_db      = readFloat("input_gain_db", 0.0f);
        params.pre_eq_gain_db     = readFloat("pre_eq_gain_db", 0.0f);
        params.pre_eq_freq_hz     = readFloat("pre_eq_freq_hz", 800.0f);
        params.reverb_wet         = readFloat("reverb_wet", 0.0f);
        params.reverb_room_size   = readFloat("reverb_room_size", 0.5f);
        params.delay_time_ms       = readFloat("delay_time_ms", 100.0f);
        params.delay_mix           = readFloat("delay_mix", 0.0f);
        params.final_eq_gain_db   = readFloat("final_eq_gain_db", 0.0f);
    }

    // ── Apply complete rig using applyNewRig ─────────────────────────────────
    // This will load models and update all parameters via APVTS
    processor.applyNewRig(params);
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


