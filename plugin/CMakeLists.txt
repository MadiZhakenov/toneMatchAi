cmake_minimum_required(VERSION 3.20)

project(ToneMatchAI VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Windows-specific settings
if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
    add_compile_definitions(NAM_CORE_AVAILABLE=1)
    # Disable VST2 compatibility to avoid missing header issues
    add_compile_definitions(JUCE_VST3_CAN_REPLACE_VST2=0)
endif()

# Add JUCE using FetchContent
if(NOT TARGET juce::juce_audio_processors)
    include(FetchContent)
    FetchContent_Declare(
        juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.0
        GIT_SHALLOW TRUE
    )
    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(juce)
endif()

# Set VST3 SDK path if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../vst3sdk")
    set(VST3_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../vst3sdk" CACHE PATH "Path to VST3 SDK")
    message(STATUS "Using VST3 SDK at: ${VST3_SDK_PATH}")
elseif(DEFINED ENV{VST3_SDK})
    set(VST3_SDK_PATH "$ENV{VST3_SDK}" CACHE PATH "Path to VST3 SDK")
    message(STATUS "Using VST3 SDK from environment: ${VST3_SDK_PATH}")
endif()

# Add NeuralAmpModelerCore
add_subdirectory(ThirdParty/NeuralAmpModelerCore)

# Create NAM library target if it doesn't exist
if(NOT TARGET NAM)
    # Create a static library for NAM
    file(GLOB_RECURSE NAM_SOURCES
        "ThirdParty/NeuralAmpModelerCore/NAM/*.cpp"
        "ThirdParty/NeuralAmpModelerCore/NAM/*.h"
    )
    
    add_library(NAM STATIC ${NAM_SOURCES})
    target_include_directories(NAM PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/NeuralAmpModelerCore"
        "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/NeuralAmpModelerCore/Dependencies/eigen"
        "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/NeuralAmpModelerCore/Dependencies/nlohmann"
    )
    
    if(WIN32)
        target_compile_definitions(NAM PUBLIC NAM_CORE_AVAILABLE=1)
    endif()
endif()

# Define plugin
juce_add_plugin(ToneMatchAI
    COMPANY_NAME "ToneMatch"
    PLUGIN_MANUFACTURER_CODE TmAI
    PLUGIN_CODE TmAI
    FORMATS VST3
    PRODUCT_NAME "ToneMatch AI"
    DESCRIPTION "AI-Powered Tone Matching VST Plugin"
    VERSION 1.0.0
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER "ToneMatch"
)

# Source files
target_sources(ToneMatchAI PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/Bridge/PythonBridge.cpp
    Source/Bridge/PythonBridge.h
    Source/DSP/NAMProcessor.cpp
    Source/DSP/NAMProcessor.h
    Source/DSP/DSPChain.cpp
    Source/DSP/DSPChain.h
    Source/Preset/PresetManager.cpp
    Source/Preset/PresetManager.h
    Source/UI/SlotComponent.cpp
    Source/UI/SlotComponent.h
    Source/UI/KnobStrip.cpp
    Source/UI/KnobStrip.h
)

# Link libraries
target_link_libraries(ToneMatchAI PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_audio_formats
    juce::juce_dsp
    juce::juce_gui_basics
    juce::juce_gui_extra
    juce::juce_core
    juce::juce_events
    NAM
)

# Include directories
target_include_directories(ToneMatchAI PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/NeuralAmpModelerCore"
    "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/NeuralAmpModelerCore/Dependencies/eigen"
)

# Resources will be loaded from file system at runtime
# No need to embed in binary

# Set output directory for VST3
# Note: VST3 installation to system folder requires admin rights
# Use install scripts (install_plugin.ps1) for system-wide installation

