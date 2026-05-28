"""
Step 1: Audio Analysis and Metadata Extraction

Analyzes audio files and extracts metadata for use in later pipeline stages.
Includes file properties, VAD detection, bandwidth detection, and language identification.
"""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .utils import get_logger, save_json

logger = get_logger("analyze")


@dataclass
class AudioMetadata:
    """Audio file metadata."""

    file_path: str
    file_name: str
    file_size_mb: float
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_rate_kbps: int
    codec: str
    
    # Analysis results
    bandwidth_type: str  # "narrowband" or "wideband"
    has_stereo_separation: bool  # True if real stereo with one speaker per channel
    language: str  # Detected language code (e.g., "no" for Norwegian)
    has_speech: bool  # Has speech content (VAD result)
    loudness_lufs: float  # Loudness in LUFS
    peak_db: float  # Peak level in dB
    dynamic_range_db: float  # Difference between peak and RMS


def get_ffprobe_info(file_path: Path) -> dict:
    """Get audio file information using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"ffprobe failed for {file_path}: {e}")
        raise


def detect_bandwidth(sample_rate: int, audio_data: np.ndarray) -> str:
    """
    Detect bandwidth type based on sample rate and frequency content.
    
    Narrowband: ≤8 kHz (telephony)
    Wideband: >8 kHz and ≤16 kHz (VoLTE, VoWiFi)
    Fullband: >16 kHz (modern codecs)
    """
    if sample_rate <= 8000:
        return "narrowband"
    elif sample_rate <= 16000:
        return "wideband"
    else:
        return "fullband"


def detect_stereo_separation(
    audio_data: np.ndarray, sample_rate: int, threshold: float = 0.3
) -> bool:
    """
    Detect if stereo audio has real separation (one speaker per channel).
    
    Compares correlation between channels:
    - High correlation (>0.8): likely same speaker/music
    - Low correlation (<0.3): likely different speakers
    """
    if audio_data.ndim != 2 or audio_data.shape[0] != 2:
        return False

    # Compute cross-correlation
    correlation = np.corrcoef(audio_data[0], audio_data[1])[0, 1]

    return correlation < threshold


def detect_language(file_path: Path) -> str:
    """
    Detect language using whisperx/faster-whisper built-in language detection.
    
    Falls back to Norwegian ('no') if detection fails or is inconclusive.
    This avoids requiring the separate `openai-whisper` package.
    """
    try:
        import whisperx

        # Load audio for language detection
        audio = whisperx.load_audio(str(file_path))
        # Detect using WhisperX's built-in decoder (faster than full transcription)
        model = whisperx.load_model("tiny", device="cpu", compute_type="int8")
        result = model.transcribe(audio, task="transcribe")
        detected = result.get("language", "no")
        logger.info(f"Detected language for {file_path.name}: {detected}")
        return detected
    except Exception as e:
        logger.warning(f"Language detection failed for {file_path}: {e}, defaulting to 'no'")
        return "no"


def detect_speech_vad(audio_data: np.ndarray, sample_rate: int) -> bool:
    """
    Detect presence of speech using Voice Activity Detection.
    
    Uses Silero VAD model for robust detection.
    """
    try:
        import torch

        # Load Silero VAD model
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_ts, save_speech, read_audio, VADIterator, collect_chunks) = utils

        # Convert to 16-bit PCM if needed
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Run VAD
        wav = torch.from_numpy(audio_data).float()
        speech_dict = get_speech_ts(wav, model, num_steps=500, sample_rate=sample_rate)

        has_speech = len(speech_dict) > 0
        logger.info(f"VAD detected speech: {has_speech}")
        return has_speech
    except Exception as e:
        logger.warning(f"VAD detection failed: {e}, assuming speech present")
        return True


def calculate_loudness_and_dynamics(
    audio_data: np.ndarray, sample_rate: int
) -> tuple[float, float, float]:
    """
    Calculate loudness (LUFS) and dynamic range.
    
    Returns: (loudness_lufs, peak_db, dynamic_range_db)
    """
    try:
        import pyloudnorm

        meter = pyloudnorm.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio_data)

        # If loudness is -inf, assume silent audio
        if np.isinf(loudness) or loudness < -100:
            loudness = -100.0

        # Calculate peak and RMS
        peak_amplitude = np.max(np.abs(audio_data))
        peak_db = 20 * np.log10(peak_amplitude + 1e-10)

        rms = np.sqrt(np.mean(audio_data**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        dynamic_range = peak_db - rms_db

        return loudness, peak_db, dynamic_range
    except Exception as e:
        logger.warning(f"Loudness calculation failed: {e}")
        return -100.0, 0.0, 0.0


def analyze_audio(file_path: Path, config: Optional[dict] = None) -> AudioMetadata:
    """
    Analyze audio file and extract metadata.
    
    Args:
        file_path: Path to audio file
        config: Configuration dict (unused for now, for future use)
        
    Returns:
        AudioMetadata object with all analysis results
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info(f"Analyzing audio file: {file_path}")

    # Get ffprobe info
    ffprobe_data = get_ffprobe_info(file_path)
    fmt = ffprobe_data.get("format", {})
    streams = ffprobe_data.get("streams", [])

    # Find audio stream
    audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)
    if not audio_stream:
        raise ValueError(f"No audio stream found in {file_path}")

    # Extract metadata
    sample_rate = int(audio_stream.get("sample_rate", 0))
    channels = int(audio_stream.get("channels", 1))
    bit_rate = int(fmt.get("bit_rate", 0)) // 1000  # Convert to kbps
    codec = audio_stream.get("codec_name", "unknown")
    duration = float(fmt.get("duration", 0))
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # Load audio with librosa
    logger.debug(f"Loading audio: sr={sample_rate}, channels={channels}")
    audio_data, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)

    # Run analysis steps
    bandwidth = detect_bandwidth(sr, audio_data)
    stereo_sep = detect_stereo_separation(
        librosa.load(str(file_path), sr=sr, mono=False)[0], sr
    )
    language = detect_language(file_path)
    has_speech = detect_speech_vad(audio_data, sr)
    loudness, peak_db, dyn_range = calculate_loudness_and_dynamics(audio_data, sr)

    metadata = AudioMetadata(
        file_path=str(file_path),
        file_name=file_path.name,
        file_size_mb=file_size_mb,
        duration_seconds=duration,
        sample_rate=sr,
        channels=channels,
        bit_rate_kbps=bit_rate,
        codec=codec,
        bandwidth_type=bandwidth,
        has_stereo_separation=stereo_sep,
        language=language,
        has_speech=has_speech,
        loudness_lufs=loudness,
        peak_db=peak_db,
        dynamic_range_db=dyn_range,
    )

    logger.info(f"Analysis complete: {metadata.file_name}")
    logger.debug(f"Metadata: {asdict(metadata)}")

    return metadata


def save_metadata(metadata: AudioMetadata, output_dir: Path) -> Path:
    """Save metadata as JSON file next to audio file."""
    output_path = output_dir / f"{Path(metadata.file_path).stem}_metadata.json"
    save_json(asdict(metadata), output_path)
    logger.info(f"Metadata saved to {output_path}")
    return output_path
