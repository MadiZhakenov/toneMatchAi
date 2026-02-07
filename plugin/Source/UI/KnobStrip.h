/*
  ==============================================================================
    KnobStrip.h
    A horizontal row of labelled rotary knobs, each attached to an APVTS parameter.
  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

//==============================================================================
/**
 * Owns N rotary sliders in a row.  Each slider is attached to an APVTS
 * parameter via SliderAttachment so changes propagate automatically.
 */
class KnobStrip : public juce::Component
{
public:
    KnobStrip() = default;
    ~KnobStrip() override = default;

    //==========================================================================
    /** Add a knob bound to an APVTS parameter.
     *  @param apvts        The value tree state.
     *  @param paramId      The parameter ID string.
     *  @param label        Display label beneath the knob.
     */
    void addKnob(juce::AudioProcessorValueTreeState& apvts,
                 const juce::String& paramId,
                 const juce::String& label);

    //==========================================================================
    void resized() override;

private:
    struct KnobEntry
    {
        std::unique_ptr<juce::Slider> slider;
        std::unique_ptr<juce::Label>  label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;
    };

    std::vector<KnobEntry> knobs;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(KnobStrip)
};


