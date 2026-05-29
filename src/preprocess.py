"""
Step 2: Adaptive Audio Preprocessing

Converts and transforms audio to optimize for transcription.
Applies adaptive preprocessing based on metadata from Step 1.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pydub import AudioSegment, effects
import librosa
import soundfile as sf

from .utils import get_logger
from .analyze import AudioMetadata

logger = get_logger("preprocess")


def load_audio_pydub(file_path: Path) -> AudioSegment:
    """Load audio file using pydub."""
    try:
        audio = AudioSegment.from_file(str(file_path))
        logger.debug(f"Loaded audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels}ch")
        return audio
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise


def resample_to_target_rate(
    audio_data: np.ndarray, 
    orig_sr: int, 
    target_sr: int = 16000
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio_data
    
    logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
    resampled = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
    return resampled


def convert_to_mono(
    audio_data: np.ndarray, 
    has_stereo_separation: bool = False
) -> np.ndarray:
    """
    Convert to mono.
    
    If has_stereo_separation is False, averages both channels (safe default).
    If has_stereo_separation is True, logs a warning that speaker information
    will be lost — caller should use split_stereo_channels() instead.
    """
    if audio_data.ndim == 1:
        logger.debug("Already mono")
        return audio_data
    
    if has_stereo_separation:
        logger.warning(
            "Stereo separation detected (one speaker per channel). "
            "Averaging channels will mix speakers. "
            "Use split_stereo_channels() to preserve speaker identity."
        )
    
    logger.info(f"Converting from {audio_data.shape[0]} channels to mono")
    mono = np.mean(audio_data, axis=0)
    return mono


def split_stereo_channels(
    audio_data: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    stem: str
) -> List[Path]:
    """
    Split stereo audio into separate mono files per channel.
    
    Returns list of paths to channel files (ch0, ch1, ...).
    Each file is named {stem}_ch{N}.wav.
    """
    if audio_data.ndim == 1:
        logger.debug("Audio is already mono, nothing to split")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_paths = []
    
    for ch in range(audio_data.shape[0]):
        ch_data = audio_data[ch]
        ch_path = output_dir / f"{stem}_ch{ch}.wav"
        
        # Normalize to -1..1 range
        max_val = np.max(np.abs(ch_data))
        if max_val > 0:
            ch_data = ch_data / max_val
        
        sf.write(str(ch_path), ch_data, sample_rate, subtype='PCM_16')
        channel_paths.append(ch_path)
        logger.info(f"Channel {ch} saved: {ch_path}")
    
    return channel_paths


def apply_highpass_filter(
    audio_data: np.ndarray, 
    sample_rate: int, 
    cutoff_hz: int = 80
) -> np.ndarray:
    """
    Apply high-pass filter to remove low-frequency rumble.
    
    Safe for all audio types.
    """
    logger.info(f"Applying high-pass filter at {cutoff_hz}Hz")
    
    from scipy import signal
    
    # Design Butterworth high-pass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        logger.warning(f"Cutoff frequency {cutoff_hz}Hz too high, skipping filter")
        return audio_data
    
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    filtered = signal.filtfilt(b, a, audio_data)
    
    return filtered


def apply_lowpass_filter(
    audio_data: np.ndarray,
    sample_rate: int,
    cutoff_hz: int = 8000,
    bandwidth_type: str = "wideband"
) -> np.ndarray:
    """
    Apply low-pass filter only for narrowband (telephony) audio.
    
    For wideband or fullband, skips to preserve useful signal.
    """
    if bandwidth_type != "narrowband":
        logger.debug(f"Skipping low-pass filter for {bandwidth_type} audio")
        return audio_data
    
    logger.info(f"Applying low-pass filter at {cutoff_hz}Hz for narrowband telephony")
    
    from scipy import signal
    
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    
    if normalized_cutoff >= 1.0:
        logger.warning(f"Cutoff frequency {cutoff_hz}Hz too high, skipping filter")
        return audio_data
    
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, audio_data)
    
    return filtered


def normalize_loudness(
    audio_data: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0
) -> np.ndarray:
    """
    Normalize audio loudness to target LUFS using ITU-R BS.1770-4.
    """
    logger.info(f"Normalizing loudness to {target_lufs} LUFS")
    
    try:
        import pyloudnorm
        
        meter = pyloudnorm.Meter(sample_rate)
        current_loudness = meter.integrated_loudness(audio_data)
        
        if np.isinf(current_loudness) or current_loudness < -100:
            logger.warning("Audio appears silent or measurement failed, skipping loudness normalization")
            return audio_data
        
        loudness_diff = target_lufs - current_loudness
        gain = 10 ** (loudness_diff / 20)
        
        # Prevent clipping: limit gain so peak never exceeds 1.0
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            max_allowed_gain = 1.0 / peak
            if gain > max_allowed_gain:
                logger.warning(
                    f"Gain {gain:.2f} would clip (peak {peak:.2f} → {peak * gain:.2f}). "
                    f"Capping gain to {max_allowed_gain:.2f}"
                )
                gain = max_allowed_gain
        
        logger.debug(f"Current loudness: {current_loudness:.1f} LUFS, applying gain: {gain:.2f}")
        normalized = audio_data * gain
        
        return normalized
    except Exception as e:
        logger.warning(f"Loudness normalization failed: {e}, skipping")
        return audio_data


def denoise_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    enable: bool = False
) -> np.ndarray:
    """
    Apply noise reduction (disabled by default due to artifact risk).
    
    Only enable explicitly for high background noise.
    """
    if not enable:
        logger.debug("Denoising disabled (default)")
        return audio_data
    
    logger.info(f"Applying denoising with strength {strength}")
    
    try:
        # Use spectral subtraction or other lightweight method
        # For now, use simple spectral gating
        S = librosa.stft(audio_data)
        S_mag = np.abs(S)
        
        # Estimate noise floor from quietest 5% of frames
        noise_floor = np.percentile(S_mag, 5, axis=1, keepdims=True)
        
        # Apply spectral gating with strength factor
        S_mag_reduced = np.maximum(S_mag - strength * noise_floor, 0)
        
        # Restore phase
        phase = np.angle(S)
        S_denoised = S_mag_reduced * np.exp(1j * phase)
        
        # Inverse STFT
        denoised = librosa.istft(S_denoised)
        
        logger.debug(f"Denoising complete, output shape: {denoised.shape}")
        return denoised
    except Exception as e:
        logger.warning(f"Denoising failed: {e}, using original audio")
        return audio_data


def preprocess_audio(
    file_path: Path,
    metadata: AudioMetadata,
    output_dir: Optional[Path] = None,
    config: Optional[dict] = None
) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio file for transcription.
    
    Args:
        file_path: Path to audio file
        metadata: AudioMetadata from Step 1
        output_dir: Optional directory to save preprocessed WAV file
        config: Configuration dict with preprocessing settings
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    logger.info(f"Preprocessing: {metadata.file_name}")
    
    if config is None:
        config = {}
    
    # Default config values
    target_sr = config.get("target_sample_rate", 16000)
    high_pass_hz = config.get("high_pass_filter_hz", 80)
    low_pass_hz = config.get("low_pass_filter_hz", 8000)
    target_lufs = config.get("loudness_target_lufs", -16)
    denoise_enabled = config.get("denoising", False)
    denoise_strength = config.get("denoising_strength", 0.5)
    
    # Load audio
    audio_data, sr = librosa.load(str(file_path), sr=metadata.sample_rate, mono=False)
    
    # Handle stereo separation: split channels if detected
    if audio_data.ndim == 2 and metadata.has_stereo_separation:
        logger.info(
            f"Stereo separation detected (correlation < 0.3). "
            f"Splitting {audio_data.shape[0]} channels into separate files."
        )
        if output_dir:
            channel_paths = split_stereo_channels(
                audio_data, sr, output_dir, Path(file_path).stem
            )
            logger.info(
                f"Channel files saved: {[p.name for p in channel_paths]}. "
                f"Each channel will be transcribed as a separate speaker."
            )
        # For downstream compatibility, return averaged mono with a warning
        # The pipeline should check metadata.has_stereo_separation and process
        # channel files separately. See ISSUES.md #5.
        audio_data = convert_to_mono(audio_data, has_stereo_separation=True)
    elif audio_data.ndim == 2:
        # Normal stereo: average to mono
        audio_data = convert_to_mono(audio_data, has_stereo_separation=False)
    
    # Resample to target
    if sr != target_sr:
        audio_data = resample_to_target_rate(audio_data, sr, target_sr)
        sr = target_sr
    
    # Apply filters
    audio_data = apply_highpass_filter(audio_data, sr, high_pass_hz)
    audio_data = apply_lowpass_filter(
        audio_data, sr, low_pass_hz, metadata.bandwidth_type
    )
    
    # Normalize loudness
    audio_data = normalize_loudness(audio_data, sr, target_lufs)
    
    # Optional denoising
    if denoise_enabled and metadata.loudness_lufs > -30:  # Only if loud enough
        audio_data = denoise_audio(audio_data, sr, denoise_strength, enable=True)
    
    logger.info(f"Preprocessing complete: {audio_data.shape[0]} samples at {sr}Hz")
    
    # Save preprocessed audio if output dir specified
    if output_dir:
        output_path = output_dir / f"{Path(file_path).stem}_preprocessed.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize to -1..1 range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        sf.write(str(output_path), audio_data, sr, subtype='PCM_16')
        logger.info(f"Preprocessed audio saved to {output_path}")
        return audio_data, sr
    
    return audio_data, sr
