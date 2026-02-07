/*
  ==============================================================================
    PluginEditor.h
    ToneMatch AI — modern minimalist UI.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginProcessor.h"

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

    // ── Section A: Main Control ──────────────────────────────────────────────
    juce::TextButton matchToneButton;
    juce::TextButton recordButton;
    double progressValue { 0.0 };
    juce::ProgressBar progressBar;
    juce::Label statusLabel;

    // ── Section B: Diagnostics ───────────────────────────────────────────────
    juce::Label rigNameLabel;
    juce::ToggleButton aiLockButton;
    juce::ToggleButton cabLockButton;
    juce::Label cabinetLabel;
    
    // Expert Tweaks section
    juce::Label expertTweaksLabel;
    juce::Slider toneShapeSlider;
    juce::Label toneShapeLabel;
    juce::Slider gainSlider;
    juce::Label gainLabel;
    juce::Slider hpfSlider;
    juce::Label hpfLabel;
    juce::Slider lpfSlider;
    juce::Label lpfLabel;

    // ── Section C: Details & Presets ─────────────────────────────────────────
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
    
    // Preset buttons
    juce::TextButton saveButton;
    juce::TextButton loadButton;
    juce::TextButton namFileButton;

    // APVTS attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> toneShapeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> gainAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hpfAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> lpfAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> aiLockAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> cabLockAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> delayTimeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> delayMixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> reverbSizeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> reverbWetAttachment;

    //==========================================================================
    void setupSectionA();
    void setupSectionB();
    void setupSectionC();
    void updateProgressDisplay();
    void updateRigDisplay();

    // Color scheme
    static juce::Colour getBackgroundColour() { return juce::Colour(0xFF1A1A1A); }  // Charcoal
    static juce::Colour getTextColour() { return juce::Colour(0xFFFFFFFF); }        // White
    static juce::Colour getAccentColour() { return juce::Colour(0xFF00A8FF); }      // Electric Blue
    static juce::Colour getSecondaryColour() { return juce::Colour(0xFF2A2A2A); }   // Dark Gray

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatchAudioProcessorEditor)
};
