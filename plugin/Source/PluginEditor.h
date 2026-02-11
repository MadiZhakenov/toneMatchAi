/*
  ==============================================================================
    PluginEditor.h
    ToneMatch AI — Ultra-Premium Dark Modern Interface with Tabs (2026)
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginProcessor.h"
#include "FontManager.h"

//==============================================================================
// ULTRA-PREMIUM MODERN LOOK AND FEEL WITH MANROPE FONT
//==============================================================================
class ModernLookAndFeel : public juce::LookAndFeel_V4
{
public:
    ModernLookAndFeel()
    {
        setColour(juce::Slider::thumbColourId, juce::Colour(0xFF00A8FF));
        setColour(juce::Slider::trackColourId, juce::Colour(0xFF00A8FF));
        setColour(juce::Slider::backgroundColourId, juce::Colour(0xFF2A2A2A));
        
        // Tab colours
        setColour(juce::TabbedComponent::outlineColourId, juce::Colour(0xFF1A1A1A));
        setColour(juce::TabbedComponent::backgroundColourId, juce::Colour(0xFF0A0A0A));
        setColour(juce::TabbedButtonBar::tabOutlineColourId, juce::Colour(0xFF1A1A1A));
        setColour(juce::TabbedButtonBar::tabTextColourId, juce::Colour(0xFF888888));
        setColour(juce::TabbedButtonBar::frontOutlineColourId, juce::Colour(0xFF00A8FF));

        // Ensure Manrope fonts are loaded
        FontManager::getInstance().initialise();
    }

    //==========================================================================
    // TAB BUTTONS - Modern minimal style
    //==========================================================================
    void drawTabButton(juce::TabBarButton& button, juce::Graphics& g, 
                       bool isMouseOver, bool isMouseDown) override
    {
        auto bounds = button.getLocalBounds().toFloat();
        
        // Background
        if (button.getToggleState())
        {
            g.setColour(juce::Colour(0xFF151515));
            g.fillRect(bounds);
            
            // Active indicator
            g.setColour(juce::Colour(0xFF00A8FF));
            g.fillRect(bounds.removeFromBottom(2));
        }
        else
        {
            g.setColour(juce::Colour(0xFF0A0A0A));
            g.fillRect(bounds);
        }
        
        // Text with Manrope
        g.setColour(button.getToggleState() ? 
                   juce::Colour(0xFF00A8FF) : 
                   juce::Colour(0xFF888888));
        g.setFont(FontManager::getInstance().getSemiBold(12.0f));
        g.drawText(button.getButtonText(), bounds, juce::Justification::centred);
    }

    //==========================================================================
    // ROTARY KNOBS - Matte dark metal with glowing arc
    //==========================================================================
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider& slider) override
    {
        auto bounds = juce::Rectangle<int>(x, y, width, height).toFloat().reduced(10);
        auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
        auto toAngle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
        auto lineW = juce::jmin(8.0f, radius * 0.5f);
        auto arcRadius = radius - lineW * 0.5f;

        // Outer shadow ring for depth
        juce::Path shadowPath;
        shadowPath.addCentredArc(bounds.getCentreX(), bounds.getCentreY(),
                                 arcRadius + 2, arcRadius + 2,
                                 0.0f, 0.0f, juce::MathConstants<float>::twoPi, true);
        g.setColour(juce::Colours::black.withAlpha(0.3f));
        g.strokePath(shadowPath, juce::PathStrokeType(lineW + 4));

        // Background arc - dark metal
        juce::Path backgroundArc;
        backgroundArc.addCentredArc(bounds.getCentreX(), bounds.getCentreY(),
                                    arcRadius, arcRadius,
                                    0.0f, rotaryStartAngle, rotaryEndAngle, true);
        g.setColour(juce::Colour(0xFF1A1A1A));
        g.strokePath(backgroundArc, juce::PathStrokeType(lineW, juce::PathStrokeType::curved));

        // Value arc - electric blue with glow
        if (sliderPos > 0.0f)
        {
            juce::Path valueArc;
            valueArc.addCentredArc(bounds.getCentreX(), bounds.getCentreY(),
                                   arcRadius, arcRadius,
                                   0.0f, rotaryStartAngle, toAngle, true);
            
            // Glow effect
            g.setColour(juce::Colour(0xFF00A8FF).withAlpha(0.3f));
            g.strokePath(valueArc, juce::PathStrokeType(lineW + 6, juce::PathStrokeType::curved));
            
            // Main arc
            g.setColour(juce::Colour(0xFF00A8FF));
            g.strokePath(valueArc, juce::PathStrokeType(lineW, juce::PathStrokeType::curved));
        }

        // Center knob - brushed metal effect
        auto centre = bounds.getCentre();
        auto knobRadius = radius * 0.4f;
        
        // Knob shadow
        g.setColour(juce::Colours::black.withAlpha(0.5f));
        g.fillEllipse(centre.x - knobRadius + 1, centre.y - knobRadius + 1, 
                      knobRadius * 2, knobRadius * 2);
        
        // Knob gradient - matte metal
        juce::ColourGradient knobGradient(
            juce::Colour(0xFF3A3A3A), centre.x, centre.y - knobRadius,
            juce::Colour(0xFF1A1A1A), centre.x, centre.y + knobRadius, false);
        g.setGradientFill(knobGradient);
        g.fillEllipse(centre.x - knobRadius, centre.y - knobRadius, 
                      knobRadius * 2, knobRadius * 2);

        // Indicator line
        juce::Path p;
        auto pointerLength = knobRadius * 0.7f;
        auto pointerThickness = 2.5f;
        p.addRectangle(-pointerThickness * 0.5f, -knobRadius, pointerThickness, pointerLength);
        p.applyTransform(juce::AffineTransform::rotation(toAngle).translated(centre.x, centre.y));
        g.setColour(juce::Colour(0xFF00A8FF));
        g.fillPath(p);
        
        // Don't draw value text here - the slider already has its own text box configured
        // Drawing it here would cause duplicate values
    }

    //==========================================================================
    // LINEAR SLIDERS - Futuristic progress bars
    //==========================================================================
    void drawLinearSlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float minSliderPos, float maxSliderPos,
                          juce::Slider::SliderStyle style, juce::Slider& slider) override
    {
        auto trackWidth = juce::jmin(6.0f, (float)height * 0.25f);
        
        juce::Point<float> startPoint(x + width * 0.0f, y + height * 0.5f);
        juce::Point<float> endPoint(x + width * 1.0f, y + height * 0.5f);

        // Background track with subtle inner shadow
        juce::Path backgroundTrack;
        backgroundTrack.startNewSubPath(startPoint);
        backgroundTrack.lineTo(endPoint);
        
        g.setColour(juce::Colour(0xFF0A0A0A));
        g.strokePath(backgroundTrack, juce::PathStrokeType(trackWidth + 2, juce::PathStrokeType::curved));
        
        g.setColour(juce::Colour(0xFF1A1A1A));
        g.strokePath(backgroundTrack, juce::PathStrokeType(trackWidth, juce::PathStrokeType::curved));

        // Value track with glow
        if (sliderPos > minSliderPos)
        {
            juce::Point<float> valueEnd(sliderPos, startPoint.y);
            juce::Path valueTrack;
            valueTrack.startNewSubPath(startPoint);
            valueTrack.lineTo(valueEnd);
            
            // Glow
            g.setColour(juce::Colour(0xFF00A8FF).withAlpha(0.3f));
            g.strokePath(valueTrack, juce::PathStrokeType(trackWidth + 4, juce::PathStrokeType::curved));
            
            // Main track
            g.setColour(juce::Colour(0xFF00A8FF));
            g.strokePath(valueTrack, juce::PathStrokeType(trackWidth, juce::PathStrokeType::curved));
        }

        // Thumb - minimal design
        auto thumbWidth = trackWidth * 2.5f;
        g.setColour(juce::Colours::black.withAlpha(0.5f));
        g.fillRoundedRectangle(sliderPos - thumbWidth * 0.5f + 1, 
                               y + height * 0.5f - thumbWidth * 0.5f + 1,
                               thumbWidth, thumbWidth, thumbWidth * 0.5f);
        
        g.setColour(juce::Colour(0xFF00A8FF));
        g.fillRoundedRectangle(sliderPos - thumbWidth * 0.5f, 
                               y + height * 0.5f - thumbWidth * 0.5f,
                               thumbWidth, thumbWidth, thumbWidth * 0.5f);
    }

    //==========================================================================
    // BUTTONS - Flat with subtle hover gradient (Manrope text)
    //==========================================================================
    void drawButtonBackground(juce::Graphics& g, juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override
    {
        auto bounds = button.getLocalBounds().toFloat().reduced(0.5f, 0.5f);
        auto cornerSize = 6.0f;

        // Shadow
        g.setColour(juce::Colours::black.withAlpha(0.3f));
        g.fillRoundedRectangle(bounds.translated(0, 1), cornerSize);

        // Base color
        g.setColour(backgroundColour);
        
        // Hover/press gradient
        if (shouldDrawButtonAsDown)
        {
            g.setColour(backgroundColour.darker(0.3f));
        }
        else if (shouldDrawButtonAsHighlighted)
        {
            juce::ColourGradient gradient(
                backgroundColour.brighter(0.1f), bounds.getCentreX(), bounds.getY(),
                backgroundColour, bounds.getCentreX(), bounds.getBottom(), false);
            g.setGradientFill(gradient);
        }

        g.fillRoundedRectangle(bounds, cornerSize);

        // Subtle highlight on top edge
        if (! shouldDrawButtonAsDown)
        {
            g.setColour(juce::Colours::white.withAlpha(0.05f));
            g.drawRoundedRectangle(bounds, cornerSize, 1.0f);
        }
    }

    void drawButtonText(juce::Graphics& g, juce::TextButton& button,
                        bool /*shouldDrawButtonAsHighlighted*/,
                        bool /*shouldDrawButtonAsDown*/) override
    {
        auto bounds = button.getLocalBounds().toFloat();
        auto colour = button.findColour(button.getToggleState()
                                            ? juce::TextButton::textColourOnId
                                            : juce::TextButton::textColourOffId);

        g.setColour(colour);
        g.setFont(FontManager::getInstance().getSemiBold(12.0f));
        g.drawText(button.getButtonText(), bounds, juce::Justification::centred);
    }

    //==========================================================================
    // TOGGLE BUTTONS - With Manrope font
    //==========================================================================
    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                          bool /*shouldDrawButtonAsHighlighted*/,
                          bool /*shouldDrawButtonAsDown*/) override
    {
        auto bounds  = button.getLocalBounds().toFloat();
        auto tickBox = bounds.removeFromLeft(bounds.getHeight()).reduced(4);

        // Background
        g.setColour(juce::Colour(0xFF1A1A1A));
        g.fillRoundedRectangle(tickBox, 4.0f);

        // Border
        g.setColour(button.getToggleState() ? juce::Colour(0xFF00A8FF)
                                            : juce::Colour(0xFF333333));
        g.drawRoundedRectangle(tickBox, 4.0f, 1.0f);

        // Tick
        if (button.getToggleState())
        {
            auto tick = tickBox.reduced(3);
            g.setColour(button.findColour(juce::ToggleButton::tickColourId));
            g.fillRoundedRectangle(tick, 2.0f);
        }

        // Text with Manrope
        g.setColour(button.findColour(juce::ToggleButton::textColourId));
        g.setFont(FontManager::getInstance().getMedium(11.0f));
        g.drawText(button.getButtonText(), bounds, juce::Justification::centredLeft);
    }

    //==========================================================================
    // LABELS / POPUPS - With Manrope font
    //==========================================================================
    juce::Font getTextButtonFont(juce::TextButton&, int) override
    {
        return FontManager::getInstance().getSemiBold(12.0f);
    }

    juce::Font getLabelFont(juce::Label& label) override
    {
        auto& fm = FontManager::getInstance();
        auto text = label.getText();

        if (text.contains("EXPERT") || text.contains("EFFECTS") || text.contains("NOISE GATE"))
            return fm.getExtraBold(14.0f);

        if (text.contains("Rig Name:") || text.contains("Cabinet:"))
            return fm.getBold(18.0f);

        if (text.contains("Auto-Boost:"))
            return fm.getRegular(12.0f);

        if (text.contains("ToneMatch AI"))
            return fm.getExtraBold(20.0f);

        return fm.getSemiBold(12.0f);
    }

    juce::Font getPopupMenuFont() override
    {
        return FontManager::getInstance().getRegular(14.0f);
    }

    void drawLabel(juce::Graphics& g, juce::Label& label) override
    {
        auto bounds = label.getLocalBounds().toFloat();

        if (label.isBeingEdited())
        {
            g.setColour(juce::Colour(0xFF0A0A0A));
            g.fillRect(bounds);
            g.setColour(juce::Colour(0xFF00A8FF));
            g.drawRect(bounds, 1.0f);
        }

        g.setColour(label.findColour(juce::Label::textColourId));
        g.setFont(getLabelFont(label));
        g.drawText(label.getText(), bounds, label.getJustificationType());
    }
};

//==============================================================================
// Forward declarations for panel classes
//==============================================================================
class ToneControlPanel;
class EffectsPanel;
class LibraryPanel;

//==============================================================================
class ToneMatchAudioProcessorEditor : public juce::AudioProcessorEditor,
                                      private juce::Timer,
                                      private juce::ValueTree::Listener
{
public:
    explicit ToneMatchAudioProcessorEditor(ToneMatchAudioProcessor&);
    ~ToneMatchAudioProcessorEditor() override;

    //==========================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

    // Public accessors for child panels
    ToneMatchAudioProcessor& getProcessor() { return processorRef; }
    juce::ToggleButton& getAILockButton() { return aiLockButton; }
    
    // Public methods for child panels
    void updateRigDisplay();
    
    // Color scheme accessors
    static juce::Colour getBackgroundColour() { return juce::Colour(0xFF0A0A0A); }
    static juce::Colour getTextColour() { return juce::Colour(0xFFE0E0E0); }
    static juce::Colour getAccentColour() { return juce::Colour(0xFF00A8FF); }
    static juce::Colour getSecondaryColour() { return juce::Colour(0xFF1A1A1A); }

private:
    void timerCallback() override;
    
    // ValueTree::Listener
    void valueTreePropertyChanged(juce::ValueTree& tree, const juce::Identifier& property) override;
    void valueTreeChildAdded(juce::ValueTree&, juce::ValueTree&) override {}
    void valueTreeChildRemoved(juce::ValueTree&, juce::ValueTree&, int) override {}
    void valueTreeChildOrderChanged(juce::ValueTree&, int, int) override {}
    void valueTreeParentChanged(juce::ValueTree&) override {}

    //==========================================================================
    ToneMatchAudioProcessor& processorRef;
    ModernLookAndFeel modernLookAndFeel;

    // ── Tabbed Interface ────────────────────────────────────────────────────
    juce::TabbedComponent tabbedComponent;
    
    // Tab Panels
    std::unique_ptr<ToneControlPanel> tonePanel;
    std::unique_ptr<EffectsPanel> effectsPanel;
    std::unique_ptr<LibraryPanel> libraryPanel;

    // ── Global Controls (Always Visible) ────────────────────────────────────
    juce::Label titleLabel;
    juce::TextButton matchToneButton;
    juce::TextButton recordButton;
    double progressValue { 0.0 };
    juce::ProgressBar progressBar;
    juce::Label statusLabel;

    // ── Global Control Attachments (Moved from Section A) ───────────────────
    // These remain here because they're used by the global controls
    juce::ToggleButton aiLockButton;  // Needed for NAM button lock check

    //==========================================================================
    void setupGlobalControls();
    void updateProgressDisplay();
    void drawSectionFrame(juce::Graphics& g, juce::Rectangle<int> bounds, const juce::String& title);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatchAudioProcessorEditor)
};

//==============================================================================
// TONE CONTROL PANEL - Tab 1
//==============================================================================
class ToneControlPanel : public juce::Component
{
public:
    ToneControlPanel(ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor);
    ~ToneControlPanel() override = default;

    void resized() override;
    void paint(juce::Graphics& g) override;
    void updateRigDisplay();
    void updateAutoCompensation(float autoCompDb);

private:
    ToneMatchAudioProcessor& processorRef;
    ToneMatchAudioProcessorEditor& editorRef;

    // Rig Info
    juce::Label rigNameLabel;
    juce::Label cabinetLabel;
    
    // Expert Tweaks
    juce::Label expertTweaksLabel;
    juce::Slider toneShapeSlider;
    juce::Label toneShapeLabel;
    juce::Slider gainSlider;
    juce::Label gainLabel;
    juce::Slider inputTrimSlider;
    juce::Label inputTrimLabel;
    juce::Label autoCompensationLabel;
    juce::Slider overdriveSlider;
    juce::Label overdriveLabel;
    juce::Slider hpfSlider;
    juce::Label hpfLabel;
    juce::Slider lpfSlider;
    juce::Label lpfLabel;

    // APVTS Attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> toneShapeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> inputTrimAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> overdriveAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hpfAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> lpfAttachment;

    void setupControls();
};

//==============================================================================
// LIBRARY PANEL - Tab 3
//==============================================================================
class LibraryPanel : public juce::Component
{
public:
    LibraryPanel(ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor);
    ~LibraryPanel() override = default;

    void resized() override;
    void paint(juce::Graphics& g) override;

private:
    ToneMatchAudioProcessor& processorRef;
    ToneMatchAudioProcessorEditor& editorRef;

    // Model List Box
    juce::ListBox modelListBox;
    
    // Custom ListBoxModel for rendering model entries
    class ModelListBoxModel : public juce::ListBoxModel
    {
    public:
        ModelListBoxModel(LibraryPanel& panel) : panelRef(panel) {}
        
        int getNumRows() override;
        void paintListBoxItem(int rowNumber, juce::Graphics& g, 
                              int width, int height, bool rowIsSelected) override;
        void listBoxItemClicked(int row, const juce::MouseEvent&) override;
        
    private:
        LibraryPanel& panelRef;
    };
    
    ModelListBoxModel listBoxModel;

    void setupControls();
};

//==============================================================================
// EFFECTS & NOISE GATE PANEL - Tab 2
//==============================================================================
class EffectsPanel : public juce::Component
{
public:
    EffectsPanel(ToneMatchAudioProcessor& processor, ToneMatchAudioProcessorEditor& editor);
    ~EffectsPanel() override = default;

    void resized() override;
    void paint(juce::Graphics& g) override;

private:
    ToneMatchAudioProcessor& processorRef;
    ToneMatchAudioProcessorEditor& editorRef;

    // Effects header
    juce::Label effectsLabel;
    
    // Delay controls
    juce::Slider delayTimeSlider;
    juce::Slider delayMixSlider;
    juce::Label delayTimeLabel;
    juce::Label delayMixLabel;
    
    // Reverb controls
    juce::Slider reverbSizeSlider;
    juce::Slider reverbWetSlider;
    juce::Label reverbSizeLabel;
    juce::Label reverbWetLabel;
    
    // Noise Gate controls
    juce::Label noiseGateLabel;
    juce::ToggleButton noiseGateEnabledButton;
    juce::Slider noiseGateThresholdSlider;
    juce::Slider noiseGateAttackSlider;
    juce::Slider noiseGateReleaseSlider;
    juce::Slider noiseGateRangeSlider;
    juce::Label noiseGateThresholdLabel;
    juce::Label noiseGateAttackLabel;
    juce::Label noiseGateReleaseLabel;
    juce::Label noiseGateRangeLabel;
    
    // Preset buttons
    juce::TextButton saveButton;
    juce::TextButton loadButton;
    juce::TextButton namFileButton;

    // APVTS Attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> delayTimeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> delayMixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> reverbSizeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> reverbWetAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> noiseGateEnabledAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseGateThresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseGateAttackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseGateReleaseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseGateRangeAttachment;

    void setupControls();
};