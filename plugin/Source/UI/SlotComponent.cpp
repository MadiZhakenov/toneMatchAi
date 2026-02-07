/*
  ==============================================================================
    SlotComponent.cpp
    Visual representation of one slot in the signal chain (Pedal, Amp, Cab).
  ==============================================================================
*/

#include "SlotComponent.h"

//==============================================================================
SlotComponent::SlotComponent(const juce::String& title,
                             const juce::String& /*fileExtensionFilter*/)
{
    // Title
    titleLabel.setText(title, juce::dontSendNotification);
    titleLabel.setFont(juce::FontOptions(16.0f, juce::Font::bold));
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(titleLabel);

    // Model name
    modelNameLabel.setText("(empty)", juce::dontSendNotification);
    modelNameLabel.setFont(juce::FontOptions(13.0f));
    modelNameLabel.setJustificationType(juce::Justification::centred);
    modelNameLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(modelNameLabel);

    // Browse button
    browseButton.onClick = [this]()
    {
        if (browseCallback)
            browseCallback();
    };
    addAndMakeVisible(browseButton);
}

SlotComponent::~SlotComponent() = default;

//==============================================================================
void SlotComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(2.0f);

    // Rounded card background
    g.setColour(juce::Colour(0xFF2A2A2A));
    g.fillRoundedRectangle(bounds, 8.0f);

    // Border
    g.setColour(juce::Colour(0xFF555555));
    g.drawRoundedRectangle(bounds, 8.0f, 1.0f);
}

void SlotComponent::resized()
{
    auto area = getLocalBounds().reduced(8);

    titleLabel.setBounds(area.removeFromTop(24));
    area.removeFromTop(4);

    modelNameLabel.setBounds(area.removeFromTop(20));
    area.removeFromTop(6);

    browseButton.setBounds(area.removeFromTop(24).reduced(16, 0));
}

//==============================================================================
void SlotComponent::setModelName(const juce::String& name)
{
    modelNameLabel.setText(name.isNotEmpty() ? name : "(empty)",
                           juce::dontSendNotification);
}




