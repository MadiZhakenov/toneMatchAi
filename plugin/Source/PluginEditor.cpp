/*
  ==============================================================================
    PluginEditor.cpp
    ToneMatch AI — main plugin GUI.
  ==============================================================================
*/

#include "PluginEditor.h"

//==============================================================================
ToneMatchAudioProcessorEditor::ToneMatchAudioProcessorEditor(
        ToneMatchAudioProcessor& p)
    : AudioProcessorEditor(&p),
      processorRef(p)
{
    setSize(720, 520);

    setupTopBar();
    setupSlots();
    setupKnobs();

    // Refresh slot names periodically
    startTimerHz(4);
}

ToneMatchAudioProcessorEditor::~ToneMatchAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void ToneMatchAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Dark background gradient
    g.fillAll(juce::Colour(0xFF1E1E1E));

    // Section dividers
    g.setColour(juce::Colour(0xFF333333));
    g.drawHorizontalLine(70, 0.0f, static_cast<float>(getWidth()));
    g.drawHorizontalLine(230, 0.0f, static_cast<float>(getWidth()));

    // Section labels
    g.setColour(juce::Colour(0xFF888888));
    g.setFont(juce::FontOptions(11.0f));
    g.drawText("SIGNAL CHAIN", 10, 74, 200, 16, juce::Justification::centredLeft);
    g.drawText("POST-FX CONTROLS", 10, 234, 200, 16, juce::Justification::centredLeft);
}

void ToneMatchAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();

    // ── Top bar (70 px) ──────────────────────────────────────────────────────
    {
        auto topBar = area.removeFromTop(70).reduced(10, 10);
        presetNameLabel.setBounds(topBar.removeFromLeft(200));
        matchButton.setBounds(topBar.removeFromRight(90));
        topBar.removeFromRight(8);
        loadButton.setBounds(topBar.removeFromRight(60));
        topBar.removeFromRight(4);
        saveButton.setBounds(topBar.removeFromRight(60));
    }

    // ── Signal chain slots (160 px) ─────────────────────────────────────────
    {
        auto slotsArea = area.removeFromTop(160).reduced(10, 10);
        const int slotW = slotsArea.getWidth() / 3;

        pedalSlot.setBounds(slotsArea.removeFromLeft(slotW).reduced(4));
        ampSlot.setBounds(slotsArea.removeFromLeft(slotW).reduced(4));
        cabSlot.setBounds(slotsArea.reduced(4));
    }

    // ── Post-FX knobs ────────────────────────────────────────────────────────
    {
        auto knobArea = area.reduced(10, 10);
        knobArea.removeFromTop(10);   // skip section label

        const int rowH = (knobArea.getHeight()) / 2;

        // Row 1: Gain + EQ
        {
            auto row = knobArea.removeFromTop(rowH);
            gainKnobs.setBounds(row.removeFromLeft(row.getWidth() / 2));
            eqKnobs.setBounds(row);
        }

        // Row 2: Delay + Reverb
        {
            auto row = knobArea;
            delayKnobs.setBounds(row.removeFromLeft(row.getWidth() / 2));
            reverbKnobs.setBounds(row);
        }
    }
}

//==============================================================================
void ToneMatchAudioProcessorEditor::timerCallback()
{
    refreshSlotNames();

    // Update match button state
    if (processorRef.isMatchRunning())
    {
        matchButton.setButtonText("Matching...");
        matchButton.setEnabled(false);
    }
    else
    {
        matchButton.setButtonText("Match");
        matchButton.setEnabled(true);
    }
}

//==============================================================================
void ToneMatchAudioProcessorEditor::setupTopBar()
{
    // Preset name
    presetNameLabel.setText("ToneMatch AI", juce::dontSendNotification);
    presetNameLabel.setFont(juce::FontOptions(22.0f, juce::Font::bold));
    presetNameLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(presetNameLabel);

    // Save button
    saveButton.onClick = [this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Save Preset", PresetManager::getDefaultPresetDirectory(), "*.json");

        chooser->launchAsync(juce::FileBrowserComponent::saveMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().getFullPathName().isNotEmpty())
                processorRef.getPresetManager().savePreset(
                    fc.getResult(), processorRef.getAPVTS(), processorRef.getDSPChain());
        });
    };
    addAndMakeVisible(saveButton);

    // Load button
    loadButton.onClick = [this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Load Preset", PresetManager::getDefaultPresetDirectory(), "*.json");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
                processorRef.getPresetManager().loadPreset(
                    fc.getResult(), processorRef.getAPVTS(), processorRef.getDSPChain());
        });
    };
    addAndMakeVisible(loadButton);

    // Match button
    matchButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xFFE8A838));
    matchButton.setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    matchButton.onClick = [this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select Reference Audio", juce::File(), "*.wav;*.mp3");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
                processorRef.triggerMatch(fc.getResult());
        });
    };
    addAndMakeVisible(matchButton);
}

void ToneMatchAudioProcessorEditor::setupSlots()
{
    // Pedal slot — browse .nam files
    pedalSlot.onBrowse([this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select Pedal NAM Model", juce::File(), "*.nam");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
                processorRef.getDSPChain().loadPedalModel(fc.getResult());
        });
    });
    addAndMakeVisible(pedalSlot);

    // Amp slot
    ampSlot.onBrowse([this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select Amp NAM Model", juce::File(), "*.nam");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
                processorRef.getDSPChain().loadAmpModel(fc.getResult());
        });
    });
    addAndMakeVisible(ampSlot);

    // Cab slot — browse IR .wav files
    cabSlot.onBrowse([this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select IR Cabinet (.wav)", juce::File(), "*.wav");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
                processorRef.getDSPChain().loadIR(fc.getResult());
        });
    });
    addAndMakeVisible(cabSlot);
}

void ToneMatchAudioProcessorEditor::setupKnobs()
{
    auto& apvts = processorRef.getAPVTS();

    // Gain knob
    gainKnobs.addKnob(apvts, "inputGain", "Input Gain");
    addAndMakeVisible(gainKnobs);

    // EQ knobs
    eqKnobs.addKnob(apvts, "preEqGainDb",  "Pre-EQ dB");
    eqKnobs.addKnob(apvts, "preEqFreqHz",  "Pre-EQ Hz");
    eqKnobs.addKnob(apvts, "finalEqGainDb","Final EQ");
    addAndMakeVisible(eqKnobs);

    // Delay knobs
    delayKnobs.addKnob(apvts, "delayTimeMs", "Time");
    delayKnobs.addKnob(apvts, "delayMix",    "Mix");
    addAndMakeVisible(delayKnobs);

    // Reverb knobs
    reverbKnobs.addKnob(apvts, "reverbWet",      "Wet");
    reverbKnobs.addKnob(apvts, "reverbRoomSize",  "Room");
    addAndMakeVisible(reverbKnobs);
}

void ToneMatchAudioProcessorEditor::refreshSlotNames()
{
    pedalSlot.setModelName(processorRef.getDSPChain().getPedalModelName());
    ampSlot.setModelName(processorRef.getDSPChain().getAmpModelName());
    cabSlot.setModelName(processorRef.getDSPChain().getIRName());
}


