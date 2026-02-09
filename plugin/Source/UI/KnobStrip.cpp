/*
  ==============================================================================
    KnobStrip.cpp
    A horizontal row of labelled rotary knobs attached to APVTS parameters.
  ==============================================================================
*/

#include "KnobStrip.h"

//==============================================================================
void KnobStrip::addKnob(juce::AudioProcessorValueTreeState& apvts,
                         const juce::String& paramId,
                         const juce::String& labelText)
{
    auto entry = KnobEntry();

    // Slider (rotary)
    entry.slider = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag,
                                                   juce::Slider::TextBoxBelow);
    entry.slider->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 16);
    entry.slider->setColour(juce::Slider::rotarySliderFillColourId,
                            juce::Colour(0xFFE8A838));   // amber accent
    addAndMakeVisible(*entry.slider);

    // Label
    entry.label = std::make_unique<juce::Label>();
    entry.label->setText(labelText, juce::dontSendNotification);
    entry.label->setFont(juce::FontOptions(11.0f));
    entry.label->setJustificationType(juce::Justification::centred);
    entry.label->setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(*entry.label);

    // APVTS attachment
    entry.attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, paramId, *entry.slider);

    knobs.push_back(std::move(entry));
    resized();
}

//==============================================================================
void KnobStrip::resized()
{
    if (knobs.empty())
        return;

    auto area = getLocalBounds();
    const int knobWidth = area.getWidth() / static_cast<int>(knobs.size());

    for (auto& k : knobs)
    {
        auto col = area.removeFromLeft(knobWidth);
        k.label->setBounds(col.removeFromBottom(16));
        k.slider->setBounds(col.reduced(4));
    }
}













