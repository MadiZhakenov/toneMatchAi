/*
  ==============================================================================
    PluginEditor.cpp
    ToneMatch AI — modern minimalist UI implementation.
  ==============================================================================
*/

#include "PluginEditor.h"

//==============================================================================
ToneMatchAudioProcessorEditor::ToneMatchAudioProcessorEditor(
        ToneMatchAudioProcessor& p)
    : AudioProcessorEditor(&p),
      processorRef(p),
      progressBar(progressValue)
{
    setSize(700, 550);

    // Subscribe to progress state changes
    processorRef.getProgressState().addListener(this);

    setupSectionA();
    setupSectionB();
    setupSectionC();

    // Initial display update
    updateRigDisplay();
    updateProgressDisplay();

    // Start timer for periodic updates
    startTimerHz(30);
}

ToneMatchAudioProcessorEditor::~ToneMatchAudioProcessorEditor()
{
    stopTimer();
    processorRef.getProgressState().removeListener(this);
}

//==============================================================================
void ToneMatchAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Charcoal background
    g.fillAll(getBackgroundColour());

    // No borders, no dividers - clean minimal design
}

void ToneMatchAudioProcessorEditor::resized()
{
    auto area = getLocalBounds();

    // ── Section A: Main Control (top, ~120px) ────────────────────────────────
    {
        auto sectionA = area.removeFromTop(120).reduced(20, 10);
        
        const int buttonWidth = 220;
        const int buttonHeight = 60;
        
        // Status label at top (centered, only visible when matching)
        auto statusArea = sectionA.removeFromTop(30);
        statusLabel.setBounds(statusArea.withSizeKeepingCentre(buttonWidth, 30));
        
        sectionA.removeFromTop(10);
        
        // Button area (centered)
        auto buttonArea = sectionA.withSizeKeepingCentre(buttonWidth, buttonHeight);
        matchToneButton.setBounds(buttonArea);
        
        // Progress bar (below button, same width, centered)
        auto progressArea = sectionA.removeFromBottom(8);
        progressBar.setBounds(progressArea.withSizeKeepingCentre(buttonWidth, 8));
    }

    // ── Section B: Diagnostics (center, ~180px) ─────────────────────────────
    {
        auto sectionB = area.removeFromTop(180).reduced(20, 10);
        
        // Rig name label (top)
        rigNameLabel.setBounds(sectionB.removeFromTop(35));
        sectionB.removeFromTop(8);
        
        // Cabinet label
        cabinetLabel.setBounds(sectionB.removeFromTop(35));
        sectionB.removeFromTop(15);
        
        // Tone Shape slider (large, horizontal)
        toneShapeLabel.setBounds(sectionB.removeFromTop(18));
        sectionB.removeFromTop(5);
        toneShapeSlider.setBounds(sectionB.removeFromTop(40));
    }

    // ── Section C: Details & Presets (bottom, remaining) ─────────────────────
    {
        auto sectionC = area.reduced(20, 10);
        
        // Reserve space for preset buttons at bottom
        auto buttonRow = sectionC.removeFromBottom(35);
        saveButton.setBounds(buttonRow.removeFromRight(70).reduced(3));
        buttonRow.removeFromRight(10);
        loadButton.setBounds(buttonRow.removeFromRight(70).reduced(3));
        
        sectionC.removeFromBottom(8);
        
        // Delay and Reverb sliders (horizontal layout)
        const int sliderHeight = 70;
        const int totalWidth = sectionC.getWidth();
        const int spacing = 8;
        const int sliderWidth = (totalWidth - (spacing * 3)) / 4;  // 4 sliders with 3 spacings
        
        // Delay Time
        auto delayTimeArea = sectionC.removeFromLeft(sliderWidth);
        delayTimeLabel.setBounds(delayTimeArea.removeFromTop(18));
        delayTimeSlider.setBounds(delayTimeArea.withHeight(sliderHeight));
        sectionC.removeFromLeft(spacing);
        
        // Delay Mix
        auto delayMixArea = sectionC.removeFromLeft(sliderWidth);
        delayMixLabel.setBounds(delayMixArea.removeFromTop(18));
        delayMixSlider.setBounds(delayMixArea.withHeight(sliderHeight));
        sectionC.removeFromLeft(spacing);
        
        // Reverb Size
        auto reverbSizeArea = sectionC.removeFromLeft(sliderWidth);
        reverbSizeLabel.setBounds(reverbSizeArea.removeFromTop(18));
        reverbSizeSlider.setBounds(reverbSizeArea.withHeight(sliderHeight));
        sectionC.removeFromLeft(spacing);
        
        // Reverb Wet
        auto reverbWetArea = sectionC.removeFromLeft(sliderWidth);
        reverbWetLabel.setBounds(reverbWetArea.removeFromTop(18));
        reverbWetSlider.setBounds(reverbWetArea.withHeight(sliderHeight));
    }
}

//==============================================================================
void ToneMatchAudioProcessorEditor::timerCallback()
{
    updateRigDisplay();
    updateProgressDisplay();
}

void ToneMatchAudioProcessorEditor::valueTreePropertyChanged(
    juce::ValueTree& tree, const juce::Identifier& property)
{
    if (tree == processorRef.getProgressState())
    {
        if (property == juce::Identifier("progressStage") ||
            property == juce::Identifier("statusText") ||
            property == juce::Identifier("progress"))
        {
            // Debug logging for testing
            juce::String propName = property.toString();
            int stage = tree.getProperty("progressStage", 0);
            juce::String status = tree.getProperty("statusText", "Unknown");
            double progress = tree.getProperty("progress", 0.0);
            
            DBG("[Editor] ValueTree changed: " + propName + 
                " -> Stage: " + juce::String(stage) + 
                ", Status: " + status + 
                ", Progress: " + juce::String(progress * 100.0, 1) + "%");
            
            updateProgressDisplay();
        }
    }
}

//==============================================================================
void ToneMatchAudioProcessorEditor::setupSectionA()
{
    // Match Tone button
    matchToneButton.setColour(juce::TextButton::buttonColourId, getAccentColour());
    matchToneButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    matchToneButton.setColour(juce::TextButton::buttonOnColourId, getAccentColour().darker(0.2f));
    matchToneButton.setColour(juce::TextButton::textColourOnId, getTextColour());
    matchToneButton.setButtonText("MATCH TONE");
    
    matchToneButton.onClick = [this]()
    {
        DBG("[Editor] MATCH TONE button clicked");
        
        // First select reference file
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select Reference Audio", juce::File(), "*.wav;*.mp3");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
            {
                DBG("[Editor] Reference file selected: " + fc.getResult().getFullPathName());
                
                // Show progress immediately
                matchToneButton.setVisible(false);
                progressBar.setVisible(true);
                statusLabel.setVisible(true);
                statusLabel.setText("Starting...", juce::dontSendNotification);
                progressValue = 0.1;
                repaint();
                
                // Trigger the match process
                processorRef.triggerMatch(fc.getResult());
            }
            else
            {
                DBG("[Editor] File selection cancelled");
            }
        });
    };
    addAndMakeVisible(matchToneButton);
    matchToneButton.setVisible(true);

    // Progress bar
    progressBar.setColour(juce::ProgressBar::foregroundColourId, getAccentColour());
    progressBar.setColour(juce::ProgressBar::backgroundColourId, getSecondaryColour());
    progressBar.setPercentageDisplay(false);
    addAndMakeVisible(progressBar);
    progressBar.setVisible(false);

    // Status label
    statusLabel.setFont(juce::FontOptions(14.0f));
    statusLabel.setColour(juce::Label::textColourId, getTextColour());
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setText("Ready", juce::dontSendNotification);
    addAndMakeVisible(statusLabel);
    statusLabel.setVisible(false);
}

void ToneMatchAudioProcessorEditor::setupSectionB()
{
    auto& apvts = processorRef.getAPVTS();

    // Rig name label
    rigNameLabel.setFont(juce::FontOptions(16.0f));
    rigNameLabel.setColour(juce::Label::textColourId, getTextColour());
    rigNameLabel.setJustificationType(juce::Justification::centredLeft);
    rigNameLabel.setText("Rig Name: None", juce::dontSendNotification);
    addAndMakeVisible(rigNameLabel);

    // Cabinet label
    cabinetLabel.setFont(juce::FontOptions(16.0f));
    cabinetLabel.setColour(juce::Label::textColourId, getTextColour());
    cabinetLabel.setJustificationType(juce::Justification::centredLeft);
    cabinetLabel.setText("Cabinet: None", juce::dontSendNotification);
    addAndMakeVisible(cabinetLabel);

    // Tone Shape slider (horizontal, large)
    toneShapeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    toneShapeSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    toneShapeSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    toneShapeSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    toneShapeSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    toneShapeSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    toneShapeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    toneShapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "finalEqGainDb", toneShapeSlider);
    addAndMakeVisible(toneShapeSlider);

    toneShapeLabel.setText("TONE SHAPE", juce::dontSendNotification);
    toneShapeLabel.setFont(juce::FontOptions(12.0f, juce::Font::bold));
    toneShapeLabel.setColour(juce::Label::textColourId, getTextColour());
    toneShapeLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(toneShapeLabel);
}

void ToneMatchAudioProcessorEditor::setupSectionC()
{
    auto& apvts = processorRef.getAPVTS();

    // Delay Time slider
    delayTimeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    delayTimeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 16);
    delayTimeSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    delayTimeSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    delayTimeSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    delayTimeSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    delayTimeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    delayTimeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "delayTimeMs", delayTimeSlider);
    addAndMakeVisible(delayTimeSlider);

    delayTimeLabel.setText("DELAY TIME", juce::dontSendNotification);
    delayTimeLabel.setFont(juce::FontOptions(11.0f));
    delayTimeLabel.setColour(juce::Label::textColourId, getTextColour());
    delayTimeLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(delayTimeLabel);

    // Delay Mix slider
    delayMixSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    delayMixSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 16);
    delayMixSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    delayMixSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    delayMixSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    delayMixSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    delayMixSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    delayMixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "delayMix", delayMixSlider);
    addAndMakeVisible(delayMixSlider);

    delayMixLabel.setText("DELAY MIX", juce::dontSendNotification);
    delayMixLabel.setFont(juce::FontOptions(11.0f));
    delayMixLabel.setColour(juce::Label::textColourId, getTextColour());
    delayMixLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(delayMixLabel);

    // Reverb Size slider
    reverbSizeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    reverbSizeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 16);
    reverbSizeSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    reverbSizeSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    reverbSizeSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    reverbSizeSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    reverbSizeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    reverbSizeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "reverbRoomSize", reverbSizeSlider);
    addAndMakeVisible(reverbSizeSlider);

    reverbSizeLabel.setText("REVERB SIZE", juce::dontSendNotification);
    reverbSizeLabel.setFont(juce::FontOptions(11.0f));
    reverbSizeLabel.setColour(juce::Label::textColourId, getTextColour());
    reverbSizeLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(reverbSizeLabel);

    // Reverb Wet slider
    reverbWetSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    reverbWetSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 16);
    reverbWetSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    reverbWetSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    reverbWetSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    reverbWetSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    reverbWetSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    reverbWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "reverbWet", reverbWetSlider);
    addAndMakeVisible(reverbWetSlider);

    reverbWetLabel.setText("REVERB WET", juce::dontSendNotification);
    reverbWetLabel.setFont(juce::FontOptions(11.0f));
    reverbWetLabel.setColour(juce::Label::textColourId, getTextColour());
    reverbWetLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(reverbWetLabel);

    // Save button
    saveButton.setButtonText("💾 SAVE");
    saveButton.setColour(juce::TextButton::buttonColourId, getSecondaryColour());
    saveButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    saveButton.onClick = [this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Save Preset", PresetManager::getDefaultPresetDirectory(), "*.json");

        chooser->launchAsync(juce::FileBrowserComponent::saveMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().getFullPathName().isNotEmpty())
                processorRef.getPresetManager().savePreset(
                    fc.getResult(), processorRef.getAPVTS(), processorRef);
        });
    };
    addAndMakeVisible(saveButton);

    // Load button
    loadButton.setButtonText("📁 LOAD");
    loadButton.setColour(juce::TextButton::buttonColourId, getSecondaryColour());
    loadButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    loadButton.onClick = [this]()
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Load Preset", PresetManager::getDefaultPresetDirectory(), "*.json");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
            {
                if (processorRef.loadPresetToProcessor(fc.getResult()))
                {
                    updateRigDisplay();
                }
            }
        });
    };
    addAndMakeVisible(loadButton);
}

void ToneMatchAudioProcessorEditor::updateProgressDisplay()
{
    auto& progressState = processorRef.getProgressState();
    int stage = progressState.getProperty("progressStage", 0);
    juce::String statusText = progressState.getProperty("statusText", "Ready");
    double progress = progressState.getProperty("progress", 0.0);

    bool isMatching = (stage > 0 && stage < 3);
    bool isDone = (stage == 3);

    // Show/hide button vs progress bar
    matchToneButton.setVisible(!isMatching);
    matchToneButton.setEnabled(!isMatching);
    progressBar.setVisible(isMatching);
    
    // Show status label only when matching or done (NOT on errors)
    statusLabel.setVisible(isMatching || isDone);

    if (isMatching)
    {
        progressValue = progress;
        // Show simple status messages only
        if (statusText == "Grid Search..." || statusText == "Optimizing..." || statusText == "Starting...")
            statusLabel.setText(statusText, juce::dontSendNotification);
        else
            statusLabel.setText("Processing...", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, getTextColour());
        repaint();
    }
    else if (isDone)
    {
        progressValue = 1.0;
        statusLabel.setText("Done!", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        matchToneButton.setEnabled(true);
    }
    else
    {
        // Idle or error - just show button, hide status
        progressValue = 0.0;
        statusLabel.setVisible(false);
        matchToneButton.setEnabled(true);
    }
}

void ToneMatchAudioProcessorEditor::updateRigDisplay()
{
    juce::String ampName = processorRef.getLastAmpName();
    juce::String cabName = processorRef.getLastCabName();
    juce::String pedalName = processorRef.getPedalModelName();

    // Build rig name string
    juce::String rigText = "Rig Name: ";
    if (pedalName.isNotEmpty() && ampName.isNotEmpty())
    {
        rigText += pedalName + " -> " + ampName;
    }
    else if (ampName.isNotEmpty())
    {
        rigText += ampName;
    }
    else
    {
        rigText += "None";
    }
    rigNameLabel.setText(rigText, juce::dontSendNotification);

    // Cabinet
    juce::String cabText = "Cabinet: ";
    if (cabName.isNotEmpty())
    {
        cabText += cabName;
    }
    else
    {
        cabText += "None";
    }
    cabinetLabel.setText(cabText, juce::dontSendNotification);
}
