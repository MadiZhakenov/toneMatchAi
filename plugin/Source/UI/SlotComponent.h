/*
  ==============================================================================
    SlotComponent.h
    Visual representation of one slot in the signal chain (Pedal, Amp, Cab).
    Shows model name, a Browse button, and optional parameter knobs.
  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <functional>

//==============================================================================
/**
 * A rectangular "card" that displays:
 *   [Title]
 *   [Model Name Label]
 *   [Browse... button]
 */
class SlotComponent : public juce::Component
{
public:
    using BrowseCallback = std::function<void()>;

    SlotComponent(const juce::String& title,
                  const juce::String& fileExtensionFilter);
    ~SlotComponent() override;

    //==========================================================================
    void paint(juce::Graphics& g) override;
    void resized() override;

    //==========================================================================
    /** Update the displayed model name. */
    void setModelName(const juce::String& name);

    /** Register a callback for the Browse button. */
    void onBrowse(BrowseCallback cb) { browseCallback = std::move(cb); }

private:
    juce::Label       titleLabel;
    juce::Label       modelNameLabel;
    juce::TextButton  browseButton { "Browse..." };

    BrowseCallback    browseCallback;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SlotComponent)
};



















