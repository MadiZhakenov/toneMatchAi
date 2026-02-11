/*
  ==============================================================================
    PluginEditor.cpp
    ToneMatch AI — Ultra-Premium Dark Modern UI with Tabs (2026)
    References: Neural DSP Archetype, Ableton Live 12, Minimalist Sci-Fi
  ==============================================================================
*/

#include "PluginEditor.h"
#include <juce_gui_basics/juce_gui_basics.h>

//==============================================================================
ToneMatchAudioProcessorEditor::ToneMatchAudioProcessorEditor(
    ToneMatchAudioProcessor& p)
: AudioProcessorEditor(&p),
  processorRef(p),
  progressBar(progressValue),
  tabbedComponent(juce::TabbedButtonBar::TabsAtTop)
{
setLookAndFeel(&modernLookAndFeel);

// INCREASED HEIGHT to 640px to ensure all knob values are visible
setSize(640, 640);

processorRef.getProgressState().addListener(this);

setupGlobalControls();

tonePanel = std::make_unique<ToneControlPanel>(processorRef, *this);
effectsPanel = std::make_unique<EffectsPanel>(processorRef, *this);
libraryPanel = std::make_unique<LibraryPanel>(processorRef, *this);

tabbedComponent.addTab("TONE", getSecondaryColour(), tonePanel.get(), false);
tabbedComponent.addTab("FX", getSecondaryColour(), effectsPanel.get(), false);
tabbedComponent.addTab("LIBRARY", getSecondaryColour(), libraryPanel.get(), false);

tabbedComponent.setColour(juce::TabbedComponent::outlineColourId, juce::Colour(0xFF1A1A1A));
tabbedComponent.setCurrentTabIndex(0);

addAndMakeVisible(tabbedComponent);
addAndMakeVisible(titleLabel);
addAndMakeVisible(recordButton);
addAndMakeVisible(matchToneButton);
addAndMakeVisible(progressBar);
addAndMakeVisible(statusLabel);

updateRigDisplay();
updateProgressDisplay();

startTimerHz(30);
}

ToneMatchAudioProcessorEditor::~ToneMatchAudioProcessorEditor()
{
    stopTimer();
    processorRef.getProgressState().removeListener(this);
    setLookAndFeel(nullptr);
}

//==============================================================================
void ToneMatchAudioProcessorEditor::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Ultra-premium radial gradient background with vignette
    juce::ColourGradient gradient(
        juce::Colour(0xFF1E1E1E), bounds.getCentreX(), bounds.getCentreY(),
        juce::Colour(0xFF0A0A0A), bounds.getX(), bounds.getY(), true);
    gradient.addColour(0.7, juce::Colour(0xFF151515));
    g.setGradientFill(gradient);
    g.fillAll();

    // Subtle noise texture (simulated with random dots)
    juce::Random random(12345);
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.015f));
    for (int i = 0; i < 800; ++i)
    {
        auto x = random.nextFloat() * bounds.getWidth();
        auto y = random.nextFloat() * bounds.getHeight();
        g.fillRect(x, y, 1.0f, 1.0f);
    }
}

void ToneMatchAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(15, 10);
    
    // Title
    titleLabel.setFont(FontManager::getInstance().getExtraBold(20.0f));
    titleLabel.setText("ToneMatch AI", juce::dontSendNotification);
    titleLabel.setColour(juce::Label::textColourId, getAccentColour().brighter(0.3f));
    titleLabel.setBounds(area.removeFromTop(35).removeFromLeft(180));
    
    area.removeFromTop(5);
    
    // Global controls row
    auto globalRow = area.removeFromTop(50);
    
    // Check if we're in loading state (matching)
    bool isMatching = processorRef.getProgressStage() > 0 && processorRef.getProgressStage() < 3;
    
    if (isMatching)
    {
        // Loading state: center progress bar and status label
        int centerWidth = 400;
        int centerX = (globalRow.getWidth() - centerWidth) / 2;
        auto centerArea = globalRow.withX(centerX).withWidth(centerWidth);
        
        // Status label on top
        statusLabel.setBounds(centerArea.removeFromTop(20).reduced(0, 2));
        
        // Progress bar below (thin, 4-6px height)
        progressBar.setBounds(centerArea.removeFromTop(5).reduced(0, 0));
    }
    else
    {
        // Normal state: show buttons
        recordButton.setBounds(globalRow.removeFromLeft(160).reduced(2));
        globalRow.removeFromLeft(10);
        
        matchToneButton.setBounds(globalRow.removeFromLeft(160).reduced(2));
        globalRow.removeFromLeft(10);
        
        forceHighGainButton.setBounds(globalRow.removeFromLeft(160).reduced(2));
        
        // Progress bar on the right (hidden in normal state)
        progressBar.setBounds(globalRow.removeFromRight(160).reduced(2, 10));
    }
    
    area.removeFromTop(8);
    
    // Tabbed component takes remaining space
    tabbedComponent.setBounds(area);
}

void ToneMatchAudioProcessorEditor::drawSectionFrame(juce::Graphics& g, juce::Rectangle<int> bounds, const juce::String& title)
{
    // Ultra-subtle frame
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.05f));
    g.drawRoundedRectangle(bounds.toFloat().reduced(2), 8.0f, 1.0f);

    // Laser-etched title (if provided)
    if (title.isNotEmpty())
    {
        g.setFont(FontManager::getInstance().getBold(11.0f));
        g.setColour(juce::Colour(0xFF00A8FF).withAlpha(0.6f));
        g.drawText(title, bounds.getX() + 15, bounds.getY() - 6, 150, 12,
                  juce::Justification::centredLeft);
        
        // Title background
        g.setColour(juce::Colour(0xFF0A0A0A));
        g.fillRect(bounds.getX() + 12, bounds.getY() - 7, title.length() * 7 + 8, 16);
        
        // Re-draw title on top
        g.setColour(juce::Colour(0xFF00A8FF).withAlpha(0.7f));
        g.drawText(title, bounds.getX() + 15, bounds.getY() - 6, 150, 12,
                  juce::Justification::centredLeft);
    }
}

//==============================================================================
void ToneMatchAudioProcessorEditor::setupGlobalControls()
{
    // Title
    titleLabel.setFont(FontManager::getInstance().getExtraBold(20.0f));
    titleLabel.setText("ToneMatch AI", juce::dontSendNotification);
    titleLabel.setColour(juce::Label::textColourId, getAccentColour().brighter(0.3f));
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);
    
    // Record DI button
    recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkred);
    recordButton.setColour(juce::TextButton::textColourOffId, getTextColour());
    recordButton.setButtonText("RECORD DI (30s)");
    recordButton.onClick = [this]()
    {
        if (processorRef.isCapturingDI())
        {
            processorRef.stopCapturingDI();
        }
        else
        {
            // Start recording (or redo if DI already recorded)
            processorRef.startCapturingDI();
            recordButton.setButtonText("RECORDING...");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
            
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
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select Reference Audio", juce::File(), "*.wav;*.mp3");

        chooser->launchAsync(juce::FileBrowserComponent::openMode, [this, chooser](const auto& fc)
        {
            if (fc.getResult().existsAsFile())
            {
                matchToneButton.setVisible(false);
                progressBar.setVisible(true);
                statusLabel.setVisible(true);
                statusLabel.setText("Starting...", juce::dontSendNotification);
                progressValue = 0.1;
                repaint();
                
                processorRef.triggerMatch(fc.getResult(), forceHighGainButton.getToggleState());
            }
        });
    };
    addAndMakeVisible(matchToneButton);

    // Force High Gain toggle button
    forceHighGainButton.setButtonText("FORCE HIGH GAIN");
    forceHighGainButton.setColour(juce::ToggleButton::textColourId, getTextColour());
    forceHighGainButton.setColour(juce::ToggleButton::tickColourId, getAccentColour());
    forceHighGainButton.setColour(juce::ToggleButton::tickDisabledColourId, juce::Colour(0xFF666666));
    addAndMakeVisible(forceHighGainButton);

    // Progress bar
    progressBar.setColour(juce::ProgressBar::foregroundColourId, getAccentColour());
    progressBar.setColour(juce::ProgressBar::backgroundColourId, getSecondaryColour());
    progressBar.setPercentageDisplay(false);
    progressBar.setVisible(false);
    addAndMakeVisible(progressBar);

    // Status label (Manrope SemiBold)
    statusLabel.setFont(FontManager::getInstance().getSemiBold(14.0f));
    statusLabel.setColour(juce::Label::textColourId, getTextColour());
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setVisible(false);
    statusLabel.setAlwaysOnTop(true);
    addAndMakeVisible(statusLabel);
}

//==============================================================================
// TONE CONTROL PANEL IMPLEMENTATION
//==============================================================================
ToneControlPanel::ToneControlPanel(
        ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor)
    : processorRef(processor), editorRef(editor)
{
    setupControls();
}

void ToneControlPanel::setupControls()
{
    auto& apvts = processorRef.getAPVTS();

    // Rig name label (Manrope Bold) - LARGER FONT
    rigNameLabel.setFont(FontManager::getInstance().getBold(18.0f));
    rigNameLabel.setColour(juce::Label::textColourId, editorRef.getTextColour());
    rigNameLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    rigNameLabel.setJustificationType(juce::Justification::centredLeft);
    rigNameLabel.setText("Rig Name: None", juce::dontSendNotification);
    addAndMakeVisible(rigNameLabel);

    // Cabinet label (Manrope Bold) - LARGER FONT
    cabinetLabel.setFont(FontManager::getInstance().getBold(18.0f));
    cabinetLabel.setColour(juce::Label::textColourId, editorRef.getTextColour());
    cabinetLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    cabinetLabel.setJustificationType(juce::Justification::centredLeft);
    cabinetLabel.setText("Cabinet: None", juce::dontSendNotification);
    addAndMakeVisible(cabinetLabel);

    // EXPERT TWEAKS section label (Manrope ExtraBold) - LARGER FONT
    expertTweaksLabel.setText("EXPERT TWEAKS", juce::dontSendNotification);
    expertTweaksLabel.setFont(FontManager::getInstance().getExtraBold(14.0f));
    expertTweaksLabel.setColour(juce::Label::textColourId, editorRef.getAccentColour());
    expertTweaksLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(expertTweaksLabel);

    // TONE SHAPE slider (Rotary Knob)
    toneShapeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    toneShapeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    toneShapeSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    toneShapeSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    toneShapeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    toneShapeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "preEqGainDb", toneShapeSlider);
    addAndMakeVisible(toneShapeSlider);

    toneShapeLabel.setText("TONE SHAPE", juce::dontSendNotification);
    toneShapeLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    toneShapeLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    toneShapeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    toneShapeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(toneShapeLabel);

    // GAIN slider (Rotary Knob)
    gainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    gainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    gainSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    gainSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    gainSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    gainAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "inputGain", gainSlider);
    addAndMakeVisible(gainSlider);

    gainLabel.setText("GAIN", juce::dontSendNotification);
    gainLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    gainLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    gainLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    gainLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(gainLabel);

    // INPUT TRIM slider (Rotary Knob)
    inputTrimSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    inputTrimSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    inputTrimSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    inputTrimSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    inputTrimSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    inputTrimAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "inputTrim", inputTrimSlider);
    addAndMakeVisible(inputTrimSlider);

    inputTrimLabel.setText("INPUT TRIM", juce::dontSendNotification);
    inputTrimLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    inputTrimLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    inputTrimLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    inputTrimLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(inputTrimLabel);

    // AUTO-COMPENSATION indicator
    autoCompensationLabel.setText("Auto-Boost: 0.0 dB", juce::dontSendNotification);
    autoCompensationLabel.setFont(FontManager::getInstance().getRegular(12.0f));
    autoCompensationLabel.setColour(juce::Label::textColourId, editorRef.getAccentColour().withAlpha(0.8f));
    autoCompensationLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    autoCompensationLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(autoCompensationLabel);

    // OVERDRIVE slider (Rotary Knob)
    overdriveSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    overdriveSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    overdriveSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    overdriveSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    overdriveSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    overdriveAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "overdrive", overdriveSlider);
    addAndMakeVisible(overdriveSlider);

    overdriveLabel.setText("OVERDRIVE", juce::dontSendNotification);
    overdriveLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    overdriveLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    overdriveLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    overdriveLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(overdriveLabel);

    // HPF slider (Rotary Knob)
    hpfSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    hpfSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    hpfSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    hpfSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    hpfSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    hpfAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "hpfFreq", hpfSlider);
    addAndMakeVisible(hpfSlider);

    hpfLabel.setText("HPF", juce::dontSendNotification);
    hpfLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    hpfLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    hpfLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    hpfLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(hpfLabel);

    // LPF slider (Rotary Knob)
    lpfSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    lpfSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    lpfSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    lpfSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    lpfSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    lpfAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "lpfFreq", lpfSlider);
    addAndMakeVisible(lpfSlider);

    lpfLabel.setText("LPF", juce::dontSendNotification);
    lpfLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    lpfLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    lpfLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    lpfLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(lpfLabel);

    // Reset Tone button
    resetToneButton.setButtonText("RESET");
    resetToneButton.setColour(juce::TextButton::buttonColourId, editorRef.getSecondaryColour());
    resetToneButton.setColour(juce::TextButton::textColourOffId, editorRef.getTextColour());
    resetToneButton.onClick = [this, &apvts]()
    {
        // Reset tone parameters to defaults
        apvts.getParameter("inputGain")->setValueNotifyingHost(apvts.getParameterRange("inputGain").convertTo0to1(0.0f));
        apvts.getParameter("inputTrim")->setValueNotifyingHost(apvts.getParameterRange("inputTrim").convertTo0to1(0.0f));
        apvts.getParameter("overdrive")->setValueNotifyingHost(apvts.getParameterRange("overdrive").convertTo0to1(0.0f));
        apvts.getParameter("preEqGainDb")->setValueNotifyingHost(apvts.getParameterRange("preEqGainDb").convertTo0to1(0.0f));
        apvts.getParameter("hpfFreq")->setValueNotifyingHost(apvts.getParameterRange("hpfFreq").convertTo0to1(70.0f));
        apvts.getParameter("lpfFreq")->setValueNotifyingHost(apvts.getParameterRange("lpfFreq").convertTo0to1(6000.0f));
    };
    addAndMakeVisible(resetToneButton);

    // Save button (preset management)
    saveButton.setButtonText("SAVE");
    saveButton.setColour(juce::TextButton::buttonColourId, editorRef.getSecondaryColour());
    saveButton.setColour(juce::TextButton::textColourOffId, editorRef.getTextColour());
    saveButton.setColour(juce::TextButton::textColourOnId, editorRef.getTextColour());
    saveButton.setColour(juce::TextButton::buttonOnColourId, editorRef.getSecondaryColour().brighter(0.1f));
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

    // Load button (preset management)
    loadButton.setButtonText("LOAD");
    loadButton.setColour(juce::TextButton::buttonColourId, editorRef.getSecondaryColour());
    loadButton.setColour(juce::TextButton::textColourOffId, editorRef.getTextColour());
    loadButton.setColour(juce::TextButton::textColourOnId, editorRef.getTextColour());
    loadButton.setColour(juce::TextButton::buttonOnColourId, editorRef.getSecondaryColour().brighter(0.1f));
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
                    // Update will be handled by the main editor
                }
            }
        });
    };
    addAndMakeVisible(loadButton);
}

void ToneControlPanel::resized()
{
    auto area = getLocalBounds().reduced(20, 15);
    
    // Rig name
    rigNameLabel.setBounds(area.removeFromTop(32));
    area.removeFromTop(8);
    
    // Cabinet
    cabinetLabel.setBounds(area.removeFromTop(32));
    area.removeFromTop(15);
    
    // Expert Tweaks header
    expertTweaksLabel.setBounds(area.removeFromTop(28));
    area.removeFromTop(10);
    
    // FIXED: Make knobs smaller to fit all 6 in the available height
    const int knobSize = 90; // Reduced from 110px to 90px
    const int knobSpacing = 15; // Reduced from 20px to 15px
    const int labelHeight = 20;
    
    int totalKnobsWidth = (knobSize * 3) + (knobSpacing * 2);
    int startX = getWidth()/2 - totalKnobsWidth/2;
    
    // Row 1: TONE SHAPE, GAIN, INPUT TRIM
    int currentY = area.getY();
    
    toneShapeSlider.setBounds(startX, currentY + labelHeight + 5, knobSize, knobSize);
    toneShapeLabel.setBounds(startX, currentY, knobSize, labelHeight);
    
    gainSlider.setBounds(startX + knobSize + knobSpacing, currentY + labelHeight + 5, knobSize, knobSize);
    gainLabel.setBounds(startX + knobSize + knobSpacing, currentY, knobSize, labelHeight);
    
    inputTrimSlider.setBounds(startX + (knobSize + knobSpacing) * 2, currentY + labelHeight + 5, knobSize, knobSize);
    inputTrimLabel.setBounds(startX + (knobSize + knobSpacing) * 2, currentY, knobSize, labelHeight);
    
    area.removeFromTop(knobSize + labelHeight + 10); // Reduced spacing
    
    // Auto-Compensation
    autoCompensationLabel.setBounds(startX, area.getY(), totalKnobsWidth, 22);
    area.removeFromTop(25); // Reduced spacing
    
    // Row 2: OVERDRIVE, HPF, LPF
    currentY = area.getY();
    
    overdriveSlider.setBounds(startX, currentY + labelHeight + 5, knobSize, knobSize);
    overdriveLabel.setBounds(startX, currentY, knobSize, labelHeight);
    
    hpfSlider.setBounds(startX + knobSize + knobSpacing, currentY + labelHeight + 5, knobSize, knobSize);
    hpfLabel.setBounds(startX + knobSize + knobSpacing, currentY, knobSize, labelHeight);
    
    lpfSlider.setBounds(startX + (knobSize + knobSpacing) * 2, currentY + labelHeight + 5, knobSize, knobSize);
    lpfLabel.setBounds(startX + (knobSize + knobSpacing) * 2, currentY, knobSize, labelHeight);
    
    // Reset button in top right corner
    resetToneButton.setBounds(getWidth() - 80, 15, 70, 25);
    
    // Preset buttons (LOAD/SAVE) in bottom-left corner of Expert Tweaks section
    auto bottomArea = area;
    int buttonWidth = 70;
    int buttonHeight = 30;
    int buttonSpacing = 10;
    int buttonsY = getHeight() - buttonHeight - 15; // 15px from bottom
    int buttonsX = 20; // 20px from left
    
    loadButton.setBounds(buttonsX, buttonsY, buttonWidth, buttonHeight);
    saveButton.setBounds(buttonsX + buttonWidth + buttonSpacing, buttonsY, buttonWidth, buttonHeight);
}


void ToneControlPanel::paint(juce::Graphics& g)
{
    // Subtle frame
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.03f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(2), 8.0f, 1.0f);
}

void ToneControlPanel::updateRigDisplay()
{
    juce::String ampName = processorRef.getLastAmpName();
    juce::String cabName = processorRef.getLastCabName();
    juce::String pedalName = processorRef.getPedalModelName();
    
    // Check if amp is bypassed (currentAmpPath is empty and no model loaded)
    bool isBypassed = processorRef.getCurrentAmpPath().isEmpty() && processorRef.getAmpModelName().isEmpty();

    juce::String rigText = "Rig Name: ";
    if (isBypassed)
    {
        rigText += "Direct Input (No Amp)";
    }
    else if (pedalName.isNotEmpty() && ampName.isNotEmpty())
    {
        rigText += pedalName + " → " + ampName;
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

void ToneControlPanel::updateAutoCompensation(float autoCompDb)
{
    if (std::abs(autoCompDb) < 0.1f)
    {
        autoCompensationLabel.setText("Auto-Boost: 0.0 dB", juce::dontSendNotification);
    }
    else
    {
        autoCompensationLabel.setText(juce::String("Auto-Boost: ") + 
            (autoCompDb >= 0.0f ? "+" : "") + 
            juce::String(autoCompDb, 1) + " dB", juce::dontSendNotification);
    }
}

//==============================================================================
// EFFECTS & NOISE GATE PANEL IMPLEMENTATION
//==============================================================================
EffectsPanel::EffectsPanel(
        ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor)
    : processorRef(processor), editorRef(editor)
{
    setupControls();
}

void EffectsPanel::setupControls()
{
    auto& apvts = processorRef.getAPVTS();

    // Effects header
    effectsLabel.setText("EFFECTS", juce::dontSendNotification);
    effectsLabel.setFont(FontManager::getInstance().getExtraBold(14.0f));
    effectsLabel.setColour(juce::Label::textColourId, editorRef.getAccentColour());
    effectsLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(effectsLabel);

    // Delay Time slider (Rotary Knob)
    delayTimeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    delayTimeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    delayTimeSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    delayTimeSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    delayTimeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    delayTimeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "delayTimeMs", delayTimeSlider);
    addAndMakeVisible(delayTimeSlider);

    delayTimeLabel.setText("DELAY TIME", juce::dontSendNotification);
    delayTimeLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    delayTimeLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    delayTimeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    delayTimeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(delayTimeLabel);

    // Delay Mix slider (Rotary Knob)
    delayMixSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    delayMixSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    delayMixSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    delayMixSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    delayMixSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    delayMixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "delayMix", delayMixSlider);
    addAndMakeVisible(delayMixSlider);

    delayMixLabel.setText("DELAY MIX", juce::dontSendNotification);
    delayMixLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    delayMixLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    delayMixLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    delayMixLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(delayMixLabel);

    // Reverb Size slider (Rotary Knob)
    reverbSizeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    reverbSizeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    reverbSizeSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    reverbSizeSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    reverbSizeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    reverbSizeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "reverbRoomSize", reverbSizeSlider);
    addAndMakeVisible(reverbSizeSlider);

    reverbSizeLabel.setText("REVERB SIZE", juce::dontSendNotification);
    reverbSizeLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    reverbSizeLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    reverbSizeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    reverbSizeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(reverbSizeLabel);

    // Reverb Wet slider (Rotary Knob)
    reverbWetSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    reverbWetSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    reverbWetSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    reverbWetSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    reverbWetSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    reverbWetAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "reverbWet", reverbWetSlider);
    addAndMakeVisible(reverbWetSlider);

    reverbWetLabel.setText("REVERB WET", juce::dontSendNotification);
    reverbWetLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    reverbWetLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    reverbWetLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    reverbWetLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(reverbWetLabel);

    // NOISE GATE section
    noiseGateLabel.setText("NOISE GATE", juce::dontSendNotification);
    noiseGateLabel.setFont(FontManager::getInstance().getExtraBold(14.0f));
    noiseGateLabel.setColour(juce::Label::textColourId, editorRef.getAccentColour());
    noiseGateLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(noiseGateLabel);

    // Noise Gate Enabled button
    noiseGateEnabledButton.setButtonText("ENABLED");
    noiseGateEnabledButton.setColour(juce::ToggleButton::textColourId, editorRef.getTextColour());
    noiseGateEnabledButton.setColour(juce::ToggleButton::tickColourId, editorRef.getAccentColour());
    noiseGateEnabledButton.setColour(juce::ToggleButton::tickDisabledColourId, juce::Colour(0xFF666666));
    noiseGateEnabledAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        apvts, "noiseGateEnabled", noiseGateEnabledButton);
    addAndMakeVisible(noiseGateEnabledButton);

    // Noise Gate Threshold slider (Rotary Knob)
    noiseGateThresholdSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    noiseGateThresholdSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    noiseGateThresholdSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    noiseGateThresholdSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    noiseGateThresholdSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    noiseGateThresholdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noiseGateThreshold", noiseGateThresholdSlider);
    addAndMakeVisible(noiseGateThresholdSlider);

    noiseGateThresholdLabel.setText("THRESHOLD", juce::dontSendNotification);
    noiseGateThresholdLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    noiseGateThresholdLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    noiseGateThresholdLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    noiseGateThresholdLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseGateThresholdLabel);

    // Noise Gate Attack slider (Rotary Knob)
    noiseGateAttackSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    noiseGateAttackSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    noiseGateAttackSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    noiseGateAttackSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    noiseGateAttackSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    noiseGateAttackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noiseGateAttack", noiseGateAttackSlider);
    addAndMakeVisible(noiseGateAttackSlider);

    noiseGateAttackLabel.setText("ATTACK", juce::dontSendNotification);
    noiseGateAttackLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    noiseGateAttackLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    noiseGateAttackLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    noiseGateAttackLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseGateAttackLabel);

    // Noise Gate Release slider (Rotary Knob)
    noiseGateReleaseSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    noiseGateReleaseSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    noiseGateReleaseSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    noiseGateReleaseSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    noiseGateReleaseSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    noiseGateReleaseAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noiseGateRelease", noiseGateReleaseSlider);
    addAndMakeVisible(noiseGateReleaseSlider);

    noiseGateReleaseLabel.setText("RELEASE", juce::dontSendNotification);
    noiseGateReleaseLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    noiseGateReleaseLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    noiseGateReleaseLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    noiseGateReleaseLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseGateReleaseLabel);

    // Noise Gate Range slider (Rotary Knob)
    noiseGateRangeSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    noiseGateRangeSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 22);
    noiseGateRangeSlider.setColour(juce::Slider::textBoxTextColourId, editorRef.getTextColour());
    noiseGateRangeSlider.setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    noiseGateRangeSlider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    noiseGateRangeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noiseGateRange", noiseGateRangeSlider);
    addAndMakeVisible(noiseGateRangeSlider);

    noiseGateRangeLabel.setText("RANGE", juce::dontSendNotification);
    noiseGateRangeLabel.setFont(FontManager::getInstance().getSemiBold(12.0f));
    noiseGateRangeLabel.setColour(juce::Label::textColourId, editorRef.getTextColour().withAlpha(0.8f));
    noiseGateRangeLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    noiseGateRangeLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(noiseGateRangeLabel);

    // Reset FX button
    resetFxButton.setButtonText("RESET");
    resetFxButton.setColour(juce::TextButton::buttonColourId, editorRef.getSecondaryColour());
    resetFxButton.setColour(juce::TextButton::textColourOffId, editorRef.getTextColour());
    resetFxButton.onClick = [this, &apvts]()
    {
        // Reset FX parameters to defaults
        apvts.getParameter("reverbWet")->setValueNotifyingHost(apvts.getParameterRange("reverbWet").convertTo0to1(0.0f));
        apvts.getParameter("reverbRoomSize")->setValueNotifyingHost(apvts.getParameterRange("reverbRoomSize").convertTo0to1(0.5f));
        apvts.getParameter("delayTimeMs")->setValueNotifyingHost(apvts.getParameterRange("delayTimeMs").convertTo0to1(100.0f));
        apvts.getParameter("delayMix")->setValueNotifyingHost(apvts.getParameterRange("delayMix").convertTo0to1(0.0f));
    };
    addAndMakeVisible(resetFxButton);
}

void EffectsPanel::resized()
{
    auto area = getLocalBounds().reduced(20, 15);
    
    // Split into left (Effects) and right (Noise Gate)
    auto leftPanel = area.removeFromLeft(area.getWidth() / 2);
    auto rightPanel = area;
    
    // ===== LEFT PANEL - EFFECTS =====
    leftPanel.removeFromRight(15);
    
    // Effects header
    effectsLabel.setBounds(leftPanel.removeFromTop(28));
    leftPanel.removeFromTop(10);
    
    // FIXED: Add an invisible spacer to match the height of the ENABLED button on the right
    // This creates a 30px tall invisible spacer to align the knob rows
    auto spacerArea = leftPanel.removeFromTop(30);
    // Completely invisible - just for spacing
    
    // Effects knobs - 2x2 grid
    const int fxKnobSize = 100;
    const int fxSpacing = 20;
    const int labelHeight = 22;
    
    int totalFxGridWidth = (fxKnobSize * 2) + fxSpacing;
    int fxStartX = leftPanel.getX() + (leftPanel.getWidth() - totalFxGridWidth) / 2;
    
    // Row 1: Delay Time and Delay Mix
    int currentY = leftPanel.getY();
    
    delayTimeSlider.setBounds(fxStartX, currentY + labelHeight + 5, fxKnobSize, fxKnobSize);
    delayTimeLabel.setBounds(fxStartX, currentY, fxKnobSize, labelHeight);
    
    delayMixSlider.setBounds(fxStartX + fxKnobSize + fxSpacing, currentY + labelHeight + 5, fxKnobSize, fxKnobSize);
    delayMixLabel.setBounds(fxStartX + fxKnobSize + fxSpacing, currentY, fxKnobSize, labelHeight);
    
    leftPanel.removeFromTop(fxKnobSize + labelHeight + 15);
    
    // Row 2: Reverb Size and Reverb Wet
    currentY = leftPanel.getY();
    
    reverbSizeSlider.setBounds(fxStartX, currentY + labelHeight + 5, fxKnobSize, fxKnobSize);
    reverbSizeLabel.setBounds(fxStartX, currentY, fxKnobSize, labelHeight);
    
    reverbWetSlider.setBounds(fxStartX + fxKnobSize + fxSpacing, currentY + labelHeight + 5, fxKnobSize, fxKnobSize);
    reverbWetLabel.setBounds(fxStartX + fxKnobSize + fxSpacing, currentY, fxKnobSize, labelHeight);
    
    leftPanel.removeFromTop(fxKnobSize + labelHeight + 20);
    
    // Preset buttons at bottom (removed - moved to TONE tab)
    
    // ===== RIGHT PANEL - NOISE GATE =====
    rightPanel.removeFromLeft(20);
    
    // Noise Gate header
    noiseGateLabel.setBounds(rightPanel.removeFromTop(28));
    rightPanel.removeFromTop(5);
    
    // Enabled button
    noiseGateEnabledButton.setBounds(rightPanel.removeFromTop(30).removeFromLeft(90).withX(rightPanel.getX()));
    rightPanel.removeFromTop(15); // This creates the space that we match on the left
    
    // Noise Gate knobs - 2x2 grid
    const int ngKnobSize = 100;
    const int ngSpacing = 20;
    
    int totalNgGridWidth = (ngKnobSize * 2) + ngSpacing;
    int ngStartX = rightPanel.getX() + (rightPanel.getWidth() - totalNgGridWidth) / 2;
    
    // Row 1: Threshold and Attack
    currentY = rightPanel.getY();
    
    noiseGateThresholdSlider.setBounds(ngStartX, currentY + labelHeight + 5, ngKnobSize, ngKnobSize);
    noiseGateThresholdLabel.setBounds(ngStartX, currentY, ngKnobSize, labelHeight);
    
    noiseGateAttackSlider.setBounds(ngStartX + ngKnobSize + ngSpacing, currentY + labelHeight + 5, ngKnobSize, ngKnobSize);
    noiseGateAttackLabel.setBounds(ngStartX + ngKnobSize + ngSpacing, currentY, ngKnobSize, labelHeight);
    
    rightPanel.removeFromTop(ngKnobSize + labelHeight + 15);
    
    // Row 2: Release and Range
    currentY = rightPanel.getY();
    
    noiseGateReleaseSlider.setBounds(ngStartX, currentY + labelHeight + 5, ngKnobSize, ngKnobSize);
    noiseGateReleaseLabel.setBounds(ngStartX, currentY, ngKnobSize, labelHeight);
    
    noiseGateRangeSlider.setBounds(ngStartX + ngKnobSize + ngSpacing, currentY + labelHeight + 5, ngKnobSize, ngKnobSize);
    noiseGateRangeLabel.setBounds(ngStartX + ngKnobSize + ngSpacing, currentY, ngKnobSize, labelHeight);
    
    // Reset FX button in top right corner
    resetFxButton.setBounds(getWidth() - 80, 15, 70, 25);
}

void EffectsPanel::paint(juce::Graphics& g)
{
    // Subtle frame
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.03f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(2), 8.0f, 1.0f);
    
    // Draw subtle divider between Effects and Noise Gate
    auto dividerX = getWidth() / 2;
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.05f));
    g.drawVerticalLine(dividerX, 20, getHeight() - 20);
}

//==============================================================================
// LIBRARY PANEL IMPLEMENTATION
//==============================================================================
LibraryPanel::LibraryPanel(
        ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor)
    : processorRef(processor), editorRef(editor), listBoxModel(*this)
{
    setupControls();
}

void LibraryPanel::setupControls()
{
    // Scan models on first creation
    processorRef.scanModels();
    
    // Setup ListBox
    modelListBox.setModel(&listBoxModel);
    modelListBox.setRowHeight(30);
    modelListBox.setColour(juce::ListBox::backgroundColourId, editorRef.getBackgroundColour());
    modelListBox.setColour(juce::ListBox::outlineColourId, juce::Colour(0xFF1A1A1A));
    addAndMakeVisible(modelListBox);
}

void LibraryPanel::resized()
{
    auto area = getLocalBounds().reduced(20, 15);
    modelListBox.setBounds(area);
}

void LibraryPanel::paint(juce::Graphics& g)
{
    // Subtle frame
    g.setColour(juce::Colour(0xFFFFFFFF).withAlpha(0.03f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(2), 8.0f, 1.0f);
}

//==============================================================================
// MODEL LIST BOX MODEL IMPLEMENTATION
//==============================================================================
int LibraryPanel::ModelListBoxModel::getNumRows()
{
    // +1 for the bypass option at index 0
    return static_cast<int>(panelRef.processorRef.getAvailableModels().size()) + 1;
}

void LibraryPanel::ModelListBoxModel::paintListBoxItem(
    int rowNumber, juce::Graphics& g, int width, int height, bool rowIsSelected)
{
    auto bounds = juce::Rectangle<int>(0, 0, width, height);
    
    // Background
    if (rowIsSelected)
    {
        g.setColour(panelRef.editorRef.getAccentColour().withAlpha(0.3f));
        g.fillRect(bounds);
        
        // Highlight border
        g.setColour(panelRef.editorRef.getAccentColour());
        g.drawRect(bounds, 1);
    }
    else
    {
        g.setColour(panelRef.editorRef.getBackgroundColour());
        g.fillRect(bounds);
    }
    
    // Text
    juce::String text;
    if (rowNumber == 0)
    {
        text = "NO AMP (BYPASS)";
    }
    else
    {
        const auto& models = panelRef.processorRef.getAvailableModels();
        int modelIndex = rowNumber - 1;
        if (modelIndex >= 0 && modelIndex < static_cast<int>(models.size()))
        {
            text = models[modelIndex].displayName;
        }
    }
    
    g.setColour(rowIsSelected ? 
                panelRef.editorRef.getAccentColour() : 
                panelRef.editorRef.getTextColour());
    g.setFont(FontManager::getInstance().getSemiBold(13.0f));
    g.drawText(text, bounds.reduced(10, 0), juce::Justification::centredLeft);
}

void LibraryPanel::ModelListBoxModel::listBoxItemClicked(int row, const juce::MouseEvent&)
{
    // Check AI Lock
    if (panelRef.editorRef.getAILockButton().getToggleState())
    {
        juce::NativeMessageBox::showMessageBoxAsync(
            juce::MessageBoxIconType::WarningIcon,
            "AI Lock Enabled",
            "AI Lock is enabled. Disable it to manually change the NAM model.");
        return;
    }
    
    if (row == 0)
    {
        // Bypass option
        panelRef.processorRef.bypassAmpModel();
    }
    else
    {
        // Load model
        const auto& models = panelRef.processorRef.getAvailableModels();
        int modelIndex = row - 1;
        if (modelIndex >= 0 && modelIndex < static_cast<int>(models.size()))
        {
            if (panelRef.processorRef.loadAmpModel(models[modelIndex].file))
            {
                // Update will be handled by the main editor's updateRigDisplay()
            }
        }
    }
    
    // Update rig display on main tab
    panelRef.editorRef.updateRigDisplay();
}

//==============================================================================
// TIMER AND UPDATE METHODS
//==============================================================================
void ToneMatchAudioProcessorEditor::timerCallback()
{
    // Auto-stop recording if buffer is full
    static bool wasRecording = false;
    bool isRecording = processorRef.isCapturingDI();
    int samples = processorRef.getCapturedDISamples();
    
    if (isRecording)
    {
        wasRecording = true;
        int total = processorRef.getDIBufferSize();
        
        if (samples >= total)
        {
            processorRef.stopCapturingDI();
            // Will be handled by the else branch below
        }
        else
        {
            // Recording in progress - show red button
            double seconds = (double)samples / processorRef.getSampleRate();
            recordButton.setButtonText("RECORDING... (" + juce::String(seconds, 1) + "s)");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
        }
    }
    else
    {
        // Not recording - determine button state based on captured samples
        if (wasRecording && !isRecording)
        {
            // Recording just stopped (manually or auto-stop)
            wasRecording = false;
            double seconds = (double)samples / processorRef.getSampleRate();
            
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
        
        // Update button state: Green if DI recorded, dark red if empty
        if (samples > 0)
        {
            // DI is recorded - show green button
            recordButton.setButtonText("DI RECORDED (CLICK TO REDO)");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::forestgreen);
        }
        else
        {
            // Buffer is empty - show dark red button
            recordButton.setButtonText("RECORD DI (30s)");
            recordButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkred);
        }
    }

    // Update Auto-Compensation indicator (via tonePanel)
    if (tonePanel != nullptr)
    {
        float autoCompDb = processorRef.getAutoCompensationDb();
        tonePanel->updateAutoCompensation(autoCompDb);
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
            updateProgressDisplay();
        }
    }
}

void ToneMatchAudioProcessorEditor::updateProgressDisplay()
{
    auto& progressState = processorRef.getProgressState();
    int stage = progressState.getProperty("progressStage", 0);
    juce::String statusText = progressState.getProperty("statusText", "");
    double progress = progressState.getProperty("progress", 0.0);

    bool isMatching = (stage > 0 && stage < 3);
    bool isDone = (stage == 3);

    bool showControls = !isMatching;
    recordButton.setVisible(showControls);
    matchToneButton.setVisible(showControls);
    matchToneButton.setEnabled(showControls);
    forceHighGainButton.setVisible(showControls);
    progressBar.setVisible(isMatching);
    
    if (isMatching || isDone)
    {
        statusLabel.setVisible(true);
    }
    
    // Update layout when loading state changes
    if (isMatching)
    {
        resized(); // Recalculate layout to center progress bar and status
    }

    if (isMatching)
    {
        progressValue = progress;
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
        
        juce::Timer::callAfterDelay(5000, [this]() {
            if (processorRef.getProgressStage() == 3)
            {
                processorRef.setProgressStage(0, "");
                statusLabel.setVisible(false);
            }
        });
    }
    else
    {
        progressValue = 0.0;
        
        if (statusText.startsWith("Error:"))
        {
            statusLabel.setVisible(true);
            statusLabel.setText(statusText, juce::dontSendNotification);
            statusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
            
            juce::Timer::callAfterDelay(5000, [this]() {
                if (processorRef.getProgressStage() == 0)
                {
                    statusLabel.setVisible(false);
                }
            });
        }
        
        matchToneButton.setEnabled(true);
    }
}

void ToneMatchAudioProcessorEditor::updateRigDisplay()
{
    if (tonePanel != nullptr)
        tonePanel->updateRigDisplay();
}