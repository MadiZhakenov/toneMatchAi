/*
  ==============================================================================
    PluginEditor.h
    ToneMatch AI — main plugin GUI.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginProcessor.h"
#include "UI/SlotComponent.h"
#include "UI/KnobStrip.h"

//==============================================================================
class ToneMatchAudioProcessorEditor : public juce::AudioProcessorEditor,
                                      private juce::Timer
{
public:
    explicit ToneMatchAudioProcessorEditor(ToneMatchAudioProcessor&);
    ~ToneMatchAudioProcessorEditor() override;

    //==========================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    //==========================================================================
    ToneMatchAudioProcessor& processorRef;

    // ── Top Bar ──────────────────────────────────────────────────────────────
    juce::Label      presetNameLabel;
    juce::TextButton saveButton   { "Save" };
    juce::TextButton loadButton   { "Load" };
    juce::TextButton matchButton  { "Match" };

    // ── Signal Chain Slots ───────────────────────────────────────────────────
    SlotComponent pedalSlot  { "PEDAL",  "*.nam" };
    SlotComponent ampSlot    { "AMP",    "*.nam" };
    SlotComponent cabSlot    { "CAB",    "*.wav" };

    // ── Post-FX Knobs ────────────────────────────────────────────────────────
    KnobStrip gainKnobs;
    KnobStrip delayKnobs;
    KnobStrip reverbKnobs;
    KnobStrip eqKnobs;

    //==========================================================================
    void setupSlots();
    void setupKnobs();
    void setupTopBar();
    void refreshSlotNames();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ToneMatchAudioProcessorEditor)
};


