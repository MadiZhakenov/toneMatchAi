/*
  ==============================================================================
    test_progress_updates.cpp
    Simple test to verify ValueTree progress updates work correctly.
    
    This simulates the progress updates that would happen during tone matching.
    Compile and run this to test the ValueTree mechanism independently.
  ==============================================================================
*/

#include <JuceHeader.h>
#include <iostream>
#include <thread>
#include <chrono>

//==============================================================================
class ProgressTest : public juce::ValueTree::Listener
{
public:
    ProgressTest()
        : progressState("ProgressState")
    {
        // Initialize like PluginProcessor does
        progressState.setProperty("progressStage", 0, nullptr);
        progressState.setProperty("statusText", "Ready", nullptr);
        progressState.setProperty("progress", 0.0, nullptr);
        
        // Subscribe to changes
        progressState.addListener(this);
    }
    
    ~ProgressTest()
    {
        progressState.removeListener(this);
    }
    
    void setProgressStage(int stage, const juce::String& statusText)
    {
        progressState.setProperty("progressStage", stage, nullptr);
        if (statusText.isNotEmpty())
            progressState.setProperty("statusText", statusText, nullptr);
        
        double progress = 0.0;
        if (stage == 1) progress = 0.3;
        else if (stage == 2) progress = 0.7;
        else if (stage == 3) progress = 1.0;
        
        progressState.setProperty("progress", progress, nullptr);
        
        std::cout << "[Processor] Updated: Stage=" << stage 
                  << ", Status=" << statusText.toStdString()
                  << ", Progress=" << (progress * 100.0) << "%" << std::endl;
    }
    
    // ValueTree::Listener
    void valueTreePropertyChanged(juce::ValueTree& tree, const juce::Identifier& property) override
    {
        if (tree == progressState)
        {
            int stage = tree.getProperty("progressStage", 0);
            juce::String status = tree.getProperty("statusText", "Unknown");
            double progress = tree.getProperty("progress", 0.0);
            
            std::cout << "[Listener] Property changed: " << property.toString().toStdString()
                      << " -> Stage=" << stage
                      << ", Status=" << status.toStdString()
                      << ", Progress=" << (progress * 100.0) << "%" << std::endl;
        }
    }
    
    void valueTreeChildAdded(juce::ValueTree&, juce::ValueTree&) override {}
    void valueTreeChildRemoved(juce::ValueTree&, juce::ValueTree&, int) override {}
    void valueTreeChildOrderChanged(juce::ValueTree&, int, int) override {}
    void valueTreeParentChanged(juce::ValueTree&) override {}
    
private:
    juce::ValueTree progressState;
};

//==============================================================================
int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "ValueTree Progress Update Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    ProgressTest test;
    
    std::cout << "Simulating tone matching process..." << std::endl;
    std::cout << std::endl;
    
    // Simulate the matching process
    std::cout << "1. Starting match..." << std::endl;
    test.setProgressStage(1, "Grid Search...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << std::endl;
    std::cout << "2. Optimizing parameters..." << std::endl;
    test.setProgressStage(2, "Optimizing...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << std::endl;
    std::cout << "3. Match complete!" << std::endl;
    test.setProgressStage(3, "Done");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    std::cout << std::endl;
    std::cout << "4. Resetting to idle..." << std::endl;
    test.setProgressStage(0, "Ready");
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Test complete! Check that all updates" << std::endl;
    std::cout << "were received by the listener." << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

