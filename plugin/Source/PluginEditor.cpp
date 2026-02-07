/*
  ==============================================================================
    PluginEditor.cpp
    ToneMatch AI — modern minimalist UI implementation.
  ==============================================================================
*/

#include "PluginEditor.h"
#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
ToneMatchAudioProcessorEditor::ToneMatchAudioProcessorEditor(
        ToneMatchAudioProcessor& p)
    : AudioProcessorEditor(&p),
      processorRef(p),
      progressBar(progressValue)
{
    setSize(700, 690);

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
        const int buttonHeight = 45;
        
        // Status label at top (centered, only visible when matching)
        auto statusArea = sectionA.removeFromTop(25);
        statusLabel.setBounds(statusArea.withSizeKeepingCentre(buttonWidth, 25));
        
        sectionA.removeFromTop(5);
        
        // Buttons (stacked)
        auto buttonsArea = sectionA.removeFromTop(95);
        recordButton.setBounds(buttonsArea.removeFromTop(buttonHeight).withSizeKeepingCentre(buttonWidth, buttonHeight));
        buttonsArea.removeFromTop(5);
        matchToneButton.setBounds(buttonsArea.removeFromTop(buttonHeight).withSizeKeepingCentre(buttonWidth, buttonHeight));
        
        // Progress bar (below button, same width, centered)
        auto progressArea = sectionA.removeFromBottom(8);
        progressBar.setBounds(progressArea.withSizeKeepingCentre(buttonWidth, 8));
    }

    // ── Section B: Diagnostics (center, expanded for Expert Tweaks) ──────────
    {
        auto sectionB = area.removeFromTop(320).reduced(20, 10);
        
        // Rig name label (top)
        rigNameLabel.setBounds(sectionB.removeFromTop(25));
        
        // Lock buttons (side by side under rig name)
        auto lockRow = sectionB.removeFromTop(25);
        aiLockButton.setBounds(lockRow.removeFromLeft(100).reduced(2));
        lockRow.removeFromLeft(10);
        cabLockButton.setBounds(lockRow.removeFromLeft(100).reduced(2));
        sectionB.removeFromTop(8);
        
        // Cabinet label
        cabinetLabel.setBounds(sectionB.removeFromTop(25));
        sectionB.removeFromTop(10);
        
        // EXPERT TWEAKS section
        expertTweaksLabel.setBounds(sectionB.removeFromTop(20));
        sectionB.removeFromTop(5);
        
        // TONE SHAPE
        toneShapeLabel.setBounds(sectionB.removeFromTop(16));
        sectionB.removeFromTop(2);
        toneShapeSlider.setBounds(sectionB.removeFromTop(30));
        sectionB.removeFromTop(5);
        
        // GAIN
        gainLabel.setBounds(sectionB.removeFromTop(16));
        sectionB.removeFromTop(2);
        gainSlider.setBounds(sectionB.removeFromTop(30));
        sectionB.removeFromTop(5);
        
        // HPF and LPF (side by side)
        auto filterRow = sectionB.removeFromTop(30);
        auto hpfArea = filterRow.removeFromLeft(filterRow.getWidth() / 2 - 5);
        hpfLabel.setBounds(hpfArea.removeFromTop(16));
        hpfArea.removeFromTop(2);
        hpfSlider.setBounds(hpfArea);
        
        filterRow.removeFromLeft(10);
        
        auto lpfArea = filterRow;
        lpfLabel.setBounds(lpfArea.removeFromTop(16));
        lpfArea.removeFromTop(2);
        lpfSlider.setBounds(lpfArea);
    }

    // ── Section C: Details & Presets (bottom, remaining) ─────────────────────
    {
        auto sectionC = area.reduced(20, 10);
        
        // Reserve space for preset buttons at bottom
        auto buttonRow = sectionC.removeFromBottom(35);
        saveButton.setBounds(buttonRow.removeFromRight(70).reduced(3));
        buttonRow.removeFromRight(10);
        loadButton.setBounds(buttonRow.removeFromRight(70).reduced(3));
        buttonRow.removeFromRight(10);
        namFileButton.setBounds(buttonRow.removeFromRight(70).reduced(3));
        
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
    // Auto-stop recording if buffer is full
    static bool wasRecording = false;
    bool isRecording = processorRef.isCapturingDI();
    
    if (isRecording)
    {
        wasRecording = true;
        int samples = processorRef.getCapturedDISamples();
        int total = processorRef.getDIBufferSize();
        
        if (samples >= total)
        {
            processorRef.stopCapturingDI();
            recordButton.setButtonText("RECORD DI (30s)");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkred);
            
            // Show success message
            double seconds = (double)samples / processorRef.getSampleRate();
            statusLabel.setVisible(true);
            statusLabel.setText("Recorded: " + juce::String(seconds, 1) + "s", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            
            // Hide message after 3 seconds
            juce::Timer::callAfterDelay(3000, [this]() {
                if (!processorRef.isCapturingDI() && processorRef.getProgressStage() == 0)
                {
                    statusLabel.setVisible(false);
                }
            });
        }
        else
        {
            double seconds = (double)samples / processorRef.getSampleRate();
            recordButton.setButtonText("RECORDING... (" + juce::String(seconds, 1) + "s)");
        }
    }
    else if (wasRecording && !isRecording)
    {
        // Recording just stopped (manually)
        wasRecording = false;
        int samples = processorRef.getCapturedDISamples();
        double seconds = (double)samples / processorRef.getSampleRate();
        
        recordButton.setButtonText("RECORD DI (30s)");
        recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkred);
        
        // Show success message
        if (samples > 0)
        {
            statusLabel.setVisible(true);
            statusLabel.setText("Recorded: " + juce::String(seconds, 1) + "s", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
            
            // Hide message after 3 seconds
            juce::Timer::callAfterDelay(3000, [this]() {
                if (!processorRef.isCapturingDI() && processorRef.getProgressStage() == 0)
                {
                    statusLabel.setVisible(false);
                }
            });
        }
        else
        {
            statusLabel.setVisible(true);
            statusLabel.setText("No audio recorded", juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
            
            // Hide message after 2 seconds
            juce::Timer::callAfterDelay(2000, [this]() {
                if (!processorRef.isCapturingDI() && processorRef.getProgressStage() == 0)
                {
                    statusLabel.setVisible(false);
                }
            });
        }
    }

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
    // Record DI button
    recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkred);
    recordButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    recordButton.setButtonText("RECORD DI (30s)");
    recordButton.onClick = [this]()
    {
        if (processorRef.isCapturingDI())
        {
            // Stop recording - feedback will be shown in timerCallback
            processorRef.stopCapturingDI();
        }
        else
        {
            // Start recording
            processorRef.startCapturingDI();
            recordButton.setButtonText("STOP RECORDING");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
            
            // Hide any previous status messages
            if (processorRef.getProgressStage() == 0)
            {
                statusLabel.setVisible(false);
            }
        }
    };
    addAndMakeVisible(recordButton);

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
    
    // Make sure status label is always on top for visibility
    statusLabel.setAlwaysOnTop(true);
}

void ToneMatchAudioProcessorEditor::setupSectionB()
{
    auto& apvts = processorRef.getAPVTS();

    // Rig name label
    rigNameLabel.setFont(juce::FontOptions(16.0f));
    rigNameLabel.setColour(juce::Label::textColourId, getTextColour());
    rigNameLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    rigNameLabel.setJustificationType(juce::Justification::centredLeft);
    rigNameLabel.setText("Rig Name: None", juce::dontSendNotification);
    addAndMakeVisible(rigNameLabel);

    // AI LOCK button
    aiLockButton.setButtonText("AI LOCK");
    aiLockButton.setColour(juce::ToggleButton::textColourId, getTextColour());
    aiLockButton.setColour(juce::ToggleButton::tickColourId, getAccentColour());
    aiLockButton.setColour(juce::ToggleButton::tickDisabledColourId, juce::Colour(0xFF666666));
    aiLockButton.onClick = [this]() { processorRef.syncLockStates(); };
    aiLockAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "aiLock", aiLockButton);
    addAndMakeVisible(aiLockButton);

    // CAB LOCK button
    cabLockButton.setButtonText("CAB LOCK");
    cabLockButton.setColour(juce::ToggleButton::textColourId, getTextColour());
    cabLockButton.setColour(juce::ToggleButton::tickColourId, getAccentColour());
    cabLockButton.setColour(juce::ToggleButton::tickDisabledColourId, juce::Colour(0xFF666666));
    cabLockButton.onClick = [this]() { processorRef.syncLockStates(); };
    cabLockAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "cabLock", cabLockButton);
    addAndMakeVisible(cabLockButton);

    // Cabinet label
    cabinetLabel.setFont(juce::FontOptions(16.0f));
    cabinetLabel.setColour(juce::Label::textColourId, getTextColour());
    cabinetLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    cabinetLabel.setJustificationType(juce::Justification::centredLeft);
    cabinetLabel.setText("Cabinet: None", juce::dontSendNotification);
    addAndMakeVisible(cabinetLabel);

    // EXPERT TWEAKS section label
    expertTweaksLabel.setText("EXPERT TWEAKS", juce::dontSendNotification);
    expertTweaksLabel.setFont(juce::FontOptions(12.0f, juce::Font::bold));
    expertTweaksLabel.setColour(juce::Label::textColourId, getAccentColour());
    expertTweaksLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(expertTweaksLabel);

    // TONE SHAPE slider (Pre-EQ Mid Boost)
    toneShapeSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    toneShapeSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    toneShapeSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    toneShapeSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    toneShapeSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    toneShapeSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    toneShapeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    toneShapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "preEqGainDb", toneShapeSlider);
    addAndMakeVisible(toneShapeSlider);

    toneShapeLabel.setText("TONE SHAPE", juce::dontSendNotification);
    toneShapeLabel.setFont(juce::FontOptions(11.0f));
    toneShapeLabel.setColour(juce::Label::textColourId, getTextColour());
    toneShapeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    toneShapeLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(toneShapeLabel);

    // GAIN slider (Input Gain)
    gainSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    gainSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    gainSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    gainSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    gainSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    gainSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    gainSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    gainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "inputGain", gainSlider);
    addAndMakeVisible(gainSlider);

    gainLabel.setText("GAIN", juce::dontSendNotification);
    gainLabel.setFont(juce::FontOptions(11.0f));
    gainLabel.setColour(juce::Label::textColourId, getTextColour());
    gainLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    gainLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(gainLabel);

    // HPF slider
    hpfSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    hpfSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    hpfSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    hpfSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    hpfSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    hpfSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    hpfSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    hpfAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "hpfFreq", hpfSlider);
    addAndMakeVisible(hpfSlider);

    hpfLabel.setText("HPF", juce::dontSendNotification);
    hpfLabel.setFont(juce::FontOptions(11.0f));
    hpfLabel.setColour(juce::Label::textColourId, getTextColour());
    hpfLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    hpfLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(hpfLabel);

    // LPF slider
    lpfSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    lpfSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
    lpfSlider.setColour(juce::Slider::trackColourId, getAccentColour());
    lpfSlider.setColour(juce::Slider::thumbColourId, getAccentColour());
    lpfSlider.setColour(juce::Slider::textBoxTextColourId, getTextColour());
    lpfSlider.setColour(juce::Slider::textBoxBackgroundColourId, getBackgroundColour());
    lpfSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    lpfAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "lpfFreq", lpfSlider);
    addAndMakeVisible(lpfSlider);

    lpfLabel.setText("LPF", juce::dontSendNotification);
    lpfLabel.setFont(juce::FontOptions(11.0f));
    lpfLabel.setColour(juce::Label::textColourId, getTextColour());
    lpfLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    lpfLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(lpfLabel);
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
    delayTimeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
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
    delayMixLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
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
    reverbSizeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
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
    reverbWetLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    reverbWetLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(reverbWetLabel);

    // Save button
    saveButton.setButtonText("SAVE");
    saveButton.setColour(juce::TextButton::buttonColourId, getSecondaryColour());
    saveButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    saveButton.setColour(juce::TextButton::textColourOnId, getTextColour());
    saveButton.setColour(juce::TextButton::buttonOnColourId, getSecondaryColour().brighter(0.1f));
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
    loadButton.setButtonText("LOAD");
    loadButton.setColour(juce::TextButton::buttonColourId, getSecondaryColour());
    loadButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    loadButton.setColour(juce::TextButton::textColourOnId, getTextColour());
    loadButton.setColour(juce::TextButton::buttonOnColourId, getSecondaryColour().brighter(0.1f));
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

    // NAM File button (for manual override)
    namFileButton.setButtonText("NAM");
    namFileButton.setColour(juce::TextButton::buttonColourId, getSecondaryColour());
    namFileButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    namFileButton.setColour(juce::TextButton::textColourOnId, getTextColour());
    namFileButton.setColour(juce::TextButton::buttonOnColourId, getSecondaryColour().brighter(0.1f));
    namFileButton.onClick = [this]()
    {
        // Check if AI lock is enabled
        if (aiLockButton.getToggleState())
        {
            juce::NativeMessageBox::showMessageBoxAsync(
                juce::MessageBoxIconType::WarningIcon,
                "AI Lock Enabled",
                "AI Lock is enabled. Disable it to manually change the NAM model.");
            return;
        }

        auto chooser = std::make_shared<juce::FileChooser>(
            "Select NAM Model", juce::File(), "*.nam");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
            {
                if (processorRef.loadAmpModel(fc.getResult()))
                {
                    updateRigDisplay();
                }
            }
        });
    };
    addAndMakeVisible(namFileButton);
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
    bool showControls = !isMatching;
    recordButton.setVisible(showControls);
    matchToneButton.setVisible(showControls);
    matchToneButton.setEnabled(showControls);
    progressBar.setVisible(isMatching);
    
    // Show status label when matching, done, or when showing recording feedback
    // (visibility is also controlled by timerCallback for recording feedback)
    if (isMatching || isDone)
    {
        statusLabel.setVisible(true);
    }

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
        
        // Show success message with rig info
        juce::String ampName = processorRef.getLastAmpName();
        juce::String cabName = processorRef.getLastCabName();
        
        if (ampName.isNotEmpty() || cabName.isNotEmpty())
        {
            juce::String doneText = "Done! ";
            if (ampName.isNotEmpty())
                doneText += ampName;
            if (ampName.isNotEmpty() && cabName.isNotEmpty())
                doneText += " + ";
            if (cabName.isNotEmpty())
                doneText += cabName;
            statusLabel.setText(doneText, juce::dontSendNotification);
        }
        else
        {
            statusLabel.setText("Done!", juce::dontSendNotification);
        }
        
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        statusLabel.setVisible(true);
        matchToneButton.setEnabled(true);
        
        // Keep message visible for 5 seconds, then hide
        juce::Timer::callAfterDelay(5000, [this]() {
            if (processorRef.getProgressStage() == 3)
            {
                // Reset to idle state
                processorRef.setProgressStage(0, "Ready");
                statusLabel.setVisible(false);
            }
        });
    }
    else
    {
        // Idle or error
        progressValue = 0.0;
        
        // Show error message if there's an error status
        if (statusText.startsWith("Error:"))
        {
            statusLabel.setVisible(true);
            statusLabel.setText(statusText, juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
            
            // Hide error after 5 seconds
            juce::Timer::callAfterDelay(5000, [this]() {
                if (processorRef.getProgressStage() == 0)
                {
                    statusLabel.setVisible(false);
                }
            });
        }
        // Otherwise, don't change visibility (let timerCallback handle recording feedback)
        
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
