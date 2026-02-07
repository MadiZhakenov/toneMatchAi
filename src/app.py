"""
Streamlit web interface for ToneMatch AI.
Allows users to upload DI and Reference tracks, run optimization, and get matched tone.
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Tuple, Dict

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from src.core.io import load_audio_file, save_audio_file, AudioTrack
from src.core.optimizer import ToneOptimizer
from src.core.analysis import analyze_track
from src.core.matching import create_match_filter


# Page configuration
st.set_page_config(
    page_title="ToneMatch AI",
    page_icon="🎸",
    layout="wide"
)

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("output", exist_ok=True)


def save_uploaded_file(uploaded_file, directory: str = "temp") -> str:
    """Save uploaded file to temporary directory with unique name.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        directory: Directory to save file (default: "temp")
    
    Returns:
        Path to saved file
    """
    # Generate unique filename
    file_extension = Path(uploaded_file.name).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(directory, unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def process_tone_match(ref_file_path: str, di_file_path: str, progress_container=None) -> Tuple[AudioTrack, Dict]:
    """Process tone matching pipeline using Universal Optimization.
    
    Uses the "Golden Standard" method: Fast Grid Search + Post-FX optimization.
    
    Args:
        ref_file_path: Path to reference audio file
        di_file_path: Path to DI audio file
        progress_container: Streamlit container for progress updates (optional)
    
    Returns:
        Tuple of (Final processed AudioTrack, optimization results dict)
    """
    # Load audio files
    ref_track = load_audio_file(ref_file_path)
    di_track = load_audio_file(di_file_path)
    
    # Universal optimization: Fast Grid Search + Deep Post-FX (with "Sighted" Optimizer)
    optimizer = ToneOptimizer(test_duration_sec=5.0, max_iterations=50)
    
    # Update progress: Stage 1
    if progress_container:
        progress_container.info("**Этап 1: Анализ** — Идет подбор лучшего усилителя и кабинета...")
    
    opt_results = optimizer.optimize_universal(di_track, ref_track)
    
    # Update progress: Stage 2 (happens inside optimize_universal, but we can update UI)
    if progress_container:
        progress_container.info("**Этап 2: Финальная настройка** — Оптимизируем реверберацию и дилей...")
    
    # Get final track and discovered rig
    final_track = opt_results['final_track']
    discovered_rig = opt_results['discovered_rig']
    
    return final_track, opt_results


def plot_spectral_comparison_streamlit(ref_features, matched_features):
    """Create spectral comparison plot for Streamlit.
    
    Args:
        ref_features: ToneFeatures from reference track
        matched_features: ToneFeatures from matched result
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both spectra
    ax.semilogx(ref_features.frequencies, ref_features.spectrum, 
                label='Reference', linewidth=2, alpha=0.8, color='blue')
    ax.semilogx(matched_features.frequencies, matched_features.spectrum, 
                label='Matched Result', linewidth=2, alpha=0.8, color='orange')
    
    # Labels and formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Spectral Comparison: Reference vs Matched Result', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable frequency range
    ax.set_xlim(20, ref_features.frequencies[-1])
    
    plt.tight_layout()
    return fig


def plot_eq_comparison_streamlit(ref_features, matched_features):
    """Create EQ comparison plot showing frequency response difference in dB.
    
    Shows the EQ difference between reference and matched result, highlighting
    where adjustments are needed.
    
    Args:
        ref_features: ToneFeatures from reference track
        matched_features: ToneFeatures from matched result
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert spectra to dB scale
    epsilon = 1e-10
    ref_spectrum_db = 20 * np.log10(ref_features.spectrum + epsilon)
    matched_spectrum_db = 20 * np.log10(matched_features.spectrum + epsilon)
    
    # Normalize both to 0 dB peak for fair comparison
    ref_peak_db = np.max(ref_spectrum_db)
    matched_peak_db = np.max(matched_spectrum_db)
    ref_spectrum_db_normalized = ref_spectrum_db - ref_peak_db
    matched_spectrum_db_normalized = matched_spectrum_db - matched_peak_db
    
    # Calculate EQ difference (what needs to be adjusted)
    eq_difference = matched_spectrum_db_normalized - ref_spectrum_db_normalized
    
    # Interpolate to common frequency axis if needed
    freq_min = max(ref_features.frequencies[0], matched_features.frequencies[0])
    freq_max = min(ref_features.frequencies[-1], matched_features.frequencies[-1])
    
    # Use matched frequencies as base (or ref if they're the same length)
    if len(ref_features.frequencies) == len(matched_features.frequencies):
        freq_axis = ref_features.frequencies
        eq_diff = eq_difference
    else:
        # Interpolate to matched frequencies
        freq_axis = matched_features.frequencies
        interp_func = interp1d(
            ref_features.frequencies, 
            ref_spectrum_db_normalized,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        ref_interp = interp_func(freq_axis)
        eq_diff = matched_spectrum_db_normalized - ref_interp
    
    # Filter to reasonable frequency range (20 Hz to 20 kHz)
    valid_mask = (freq_axis >= 20) & (freq_axis <= 20000)
    freq_axis = freq_axis[valid_mask]
    eq_diff = eq_diff[valid_mask]
    
    # Plot EQ difference
    # Positive values = matched has more energy (needs to be cut)
    # Negative values = matched has less energy (needs to be boosted)
    colors = np.where(eq_diff > 0, 'red', 'green')
    ax.fill_between(freq_axis, 0, eq_diff, alpha=0.3, color='gray', label='EQ Difference')
    ax.semilogx(freq_axis, eq_diff, linewidth=2.5, color='darkblue', label='EQ Difference (dB)')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add frequency bands annotations
    band_freqs = [80, 250, 500, 1000, 2000, 4000, 8000]
    band_names = ['Low', 'Low-Mid', 'Mid', 'Upper-Mid', 'Presence', 'High', 'Brilliance']
    for freq, name in zip(band_freqs, band_names):
        if freq_min <= freq <= freq_max:
            ax.axvline(x=freq, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)
            # Find closest index
            idx = np.argmin(np.abs(freq_axis - freq))
            if idx < len(eq_diff):
                value = eq_diff[idx]
                ax.text(freq, value + (2 if value >= 0 else -2), name, 
                       ha='center', fontsize=8, alpha=0.7, rotation=90)
    
    # Labels and formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('EQ Difference (dB)', fontsize=12, fontweight='bold')
    ax.set_title('EQ Analysis: Difference Between Matched Result and Reference', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(20, 20000)
    
    # Set y-axis limits with some padding
    y_max = max(np.abs(eq_diff)) * 1.2 if len(eq_diff) > 0 else 10
    y_max = max(y_max, 5)  # At least 5 dB range
    ax.set_ylim(-y_max, y_max)
    
    # Add text annotation with summary
    max_diff = np.max(np.abs(eq_diff)) if len(eq_diff) > 0 else 0
    mean_diff = np.mean(np.abs(eq_diff)) if len(eq_diff) > 0 else 0
    summary_text = f'Max Difference: {max_diff:.1f} dB\nMean Difference: {mean_diff:.1f} dB'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


# Main UI
def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎸 ToneMatch AI")
    st.markdown("*Upload your DI and a Reference track to copy the tone.*")
    
    st.divider()
    
    # File upload section
    st.subheader("📁 Upload Audio Files")
    col1, col2 = st.columns(2)
    
    with col1:
        ref_file = st.file_uploader(
            "Reference Track",
            type=['wav', 'mp3', 'flac'],
            help="Upload the reference track you want to match"
        )
        if ref_file is not None:
            st.success(f"✅ Loaded: {ref_file.name}")
    
    with col2:
        di_file = st.file_uploader(
            "Your DI Track",
            type=['wav', 'mp3', 'flac'],
            help="Upload your DI (Direct Input) track to process"
        )
        if di_file is not None:
            st.success(f"✅ Loaded: {di_file.name}")
    
    st.divider()
    
    # Process button
    if st.button("🔥 Match Tone!", type="primary", use_container_width=True):
        if ref_file is None or di_file is None:
            st.error("❌ Please upload both Reference and DI tracks!")
        else:
            try:
                # Create progress container
                progress_container = st.empty()
                
                with st.spinner("Processing... This may take a few minutes."):
                    # Save uploaded files
                    ref_path = save_uploaded_file(ref_file)
                    di_path = save_uploaded_file(di_file)
                    
                    # Process tone matching with progress updates
                    result_track, opt_results = process_tone_match(ref_path, di_path, progress_container)
                    
                    # Save result
                    output_path = "output/final_result.wav"
                    save_audio_file(result_track, output_path)
                    
                    # Store in session state for display
                    st.session_state['result_track'] = result_track
                    st.session_state['result_path'] = output_path
                    st.session_state['ref_path'] = ref_path
                    st.session_state['di_path'] = di_path
                    st.session_state['opt_results'] = opt_results
                    
                    # Clear progress container
                    progress_container.empty()
                    
                    st.success("✅ Processing complete!")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Results section
    if 'result_track' in st.session_state:
        st.divider()
        st.subheader("🎧 Result")
        
        # Display discovered rig information
        if 'opt_results' in st.session_state:
            opt_results = st.session_state['opt_results']
            discovered_rig = opt_results.get('discovered_rig', {})
            
            if discovered_rig:
                fx_name = discovered_rig.get('fx_nam_name', 'N/A')
                amp_name = discovered_rig.get('amp_nam_name', 'N/A')
                ir_name = discovered_rig.get('ir_name', 'N/A')
                
                st.success(f"**Обнаруженный риг:** [{fx_name}] → [{amp_name}] → [{ir_name}]")
                
                # Display error components if available
                post_fx_results = opt_results.get('post_fx_results', {})
                if 'initial_error_components' in post_fx_results and 'final_error_components' in post_fx_results:
                    initial = post_fx_results['initial_error_components']
                    final = post_fx_results['final_error_components']
                    
                    with st.expander("📊 Показать компоненты ошибки (Sighted Optimizer)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Harmonic Loss", 
                                    f"{final['harmonic_loss']:.4f}",
                                    f"{initial['harmonic_loss'] - final['harmonic_loss']:.4f}")
                        with col2:
                            st.metric("Envelope Loss",
                                    f"{final['envelope_loss']:.4f}",
                                    f"{initial['envelope_loss'] - final['envelope_loss']:.4f}")
                        with col3:
                            st.metric("Spectral Shape Loss",
                                    f"{final['spectral_shape_loss']:.4f}",
                                    f"{initial['spectral_shape_loss'] - final['spectral_shape_loss']:.4f}")
                        
                        st.metric("Brightness Loss",
                                f"{final['brightness_loss']:.4f}",
                                f"{initial['brightness_loss'] - final['brightness_loss']:.4f}")
                
                # Display Pre-EQ parameters if available
                post_fx_params = opt_results.get('post_fx_params', {})
                if post_fx_params:
                    pre_eq_gain = post_fx_params.get('pre_eq_gain_db', 0.0)
                    pre_eq_freq = post_fx_params.get('pre_eq_freq_hz', 800.0)
                    if abs(pre_eq_gain) > 0.01:  # Only show if significant
                        st.info(f"**Pre-EQ Shaping:** {pre_eq_gain:+.1f} dB @ {pre_eq_freq:.0f} Hz")
                
                # Display final loss
                final_loss = opt_results.get('final_loss', 0.0)
                st.metric("Final Loss", f"{final_loss:.6f}")
        
        # Audio players in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Reference Tone**")
            if os.path.exists(st.session_state['ref_path']):
                st.audio(st.session_state['ref_path'])
        
        with col2:
            st.write("**Your Matched Tone**")
            if os.path.exists(st.session_state['result_path']):
                st.audio(st.session_state['result_path'])
        
        # Download button
        if os.path.exists(st.session_state['result_path']):
            with open(st.session_state['result_path'], "rb") as f:
                st.download_button(
                    label="⬇️ Download Matched WAV",
                    data=f.read(),
                    file_name="matched_tone.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        
        # EQ Visualization - Always visible
        st.subheader("🎛️ EQ Analysis")
        try:
            # Load and analyze tracks
            ref_track = load_audio_file(st.session_state['ref_path'])
            ref_features = analyze_track(ref_track)
            matched_features = analyze_track(st.session_state['result_track'])
            
            # Create and display EQ comparison plot
            eq_fig = plot_eq_comparison_streamlit(ref_features, matched_features)
            st.pyplot(eq_fig)
            
            # Calculate and display key EQ metrics
            epsilon = 1e-10
            ref_spectrum_db = 20 * np.log10(ref_features.spectrum + epsilon)
            matched_spectrum_db = 20 * np.log10(matched_features.spectrum + epsilon)
            ref_peak_db = np.max(ref_spectrum_db)
            matched_peak_db = np.max(matched_spectrum_db)
            ref_spectrum_db_normalized = ref_spectrum_db - ref_peak_db
            matched_spectrum_db_normalized = matched_spectrum_db - matched_peak_db
            
            # Calculate EQ difference
            if len(ref_features.frequencies) == len(matched_features.frequencies):
                eq_difference = matched_spectrum_db_normalized - ref_spectrum_db_normalized
                freq_axis = ref_features.frequencies
            else:
                freq_axis = matched_features.frequencies
                interp_func = interp1d(
                    ref_features.frequencies, 
                    ref_spectrum_db_normalized,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                ref_interp = interp_func(freq_axis)
                eq_difference = matched_spectrum_db_normalized - ref_interp
            
            # Filter to audio range
            valid_mask = (freq_axis >= 20) & (freq_axis <= 20000)
            eq_difference = eq_difference[valid_mask]
            
            # Display key metrics
            max_diff = np.max(np.abs(eq_difference)) if len(eq_difference) > 0 else 0
            mean_diff = np.mean(np.abs(eq_difference)) if len(eq_difference) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max EQ Difference", f"{max_diff:.2f} dB", 
                         help="Maximum frequency response difference")
            with col2:
                st.metric("Mean EQ Difference", f"{mean_diff:.2f} dB",
                         help="Average frequency response difference")
            with col3:
                # Calculate RMS of difference
                rms_diff = np.sqrt(np.mean(eq_difference ** 2)) if len(eq_difference) > 0 else 0
                st.metric("RMS EQ Difference", f"{rms_diff:.2f} dB",
                         help="Root mean square of EQ difference")
            
            # Warning if difference is too large
            if max_diff > 6.0:
                st.warning(f"⚠️ Большая разница в EQ! Максимальная разница: {max_diff:.1f} dB. "
                          f"Результат значительно отличается от референса по частотной характеристике.")
            elif max_diff > 3.0:
                st.info(f"ℹ️ Заметная разница в EQ. Максимальная разница: {max_diff:.1f} dB. "
                       f"Рекомендуется дополнительная настройка.")
            else:
                st.success(f"✅ Хорошее совпадение EQ! Максимальная разница: {max_diff:.1f} dB.")
                
        except Exception as e:
            st.error(f"Error creating EQ visualization: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        # Spectral visualization (expanded)
        with st.expander("📊 Show Detailed Spectral Analysis", expanded=False):
            try:
                # Load and analyze tracks (if not already done)
                if 'ref_features' not in locals():
                    ref_track = load_audio_file(st.session_state['ref_path'])
                    ref_features = analyze_track(ref_track)
                    matched_features = analyze_track(st.session_state['result_track'])
                
                # Create and display plot
                fig = plot_spectral_comparison_streamlit(ref_features, matched_features)
                st.pyplot(fig)
                
                # Display metrics
                st.write("**Spectral Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reference Centroid", f"{ref_features.spectral_centroid:.1f} Hz")
                with col2:
                    st.metric("Matched Centroid", f"{matched_features.spectral_centroid:.1f} Hz")
                with col3:
                    diff = matched_features.spectral_centroid - ref_features.spectral_centroid
                    st.metric("Difference", f"{diff:+.1f} Hz")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")


if __name__ == "__main__":
    main()

