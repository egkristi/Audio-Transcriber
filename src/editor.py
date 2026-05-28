"""
Step 6: Manual Correction Editor

Placeholder for interactive editor for manual transcription review and correction.
Currently exports to SRT format for external editing (Subtitle Edit for Mac, etc.)

Future: Integrate web-based editor with waveform visualization (wavesurfer.js + React).
"""

from pathlib import Path
from typing import Optional

from .utils import get_logger

logger = get_logger("editor")


def export_for_manual_editing(
    srt_file_path: Path,
    audio_file_path: Path,
    config: Optional[dict] = None
) -> str:
    """
    Prepare transcription for manual editing.
    
    Currently exports SRT file for use with external editors.
    
    Args:
        srt_file_path: Path to generated SRT file
        audio_file_path: Path to original audio file
        config: Configuration dict
        
    Returns:
        Instructions for manual editing
    """
    logger.info(f"Preparing for manual editing: {srt_file_path}")
    
    if not srt_file_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_file_path}")
    
    instructions = f"""
=== Manual Transcription Editing Instructions ===

SRT File: {srt_file_path}
Audio File: {audio_file_path}

RECOMMENDED TOOLS:
- Mac: Subtitle Edit for Mac (free, open-source)
  - Supports Norwegian spell-checking
  - Integrated waveform and player
- Alternative: Any text editor + audio player of choice

EDITING WORKFLOW:
1. Open the SRT file in your preferred editor
2. Play the audio file alongside
3. Review flagged segments (marked in comparison report)
4. Correct transcription errors:
   - Fix proper nouns (names, places, companies)
   - Correct numbers and technical terms
   - Fix grammar/punctuation if using main model
5. Save the corrected SRT file
6. Optionally export to other formats (VTT, PDF, etc.)

TIPS:
- Listen to sections with low confidence scores first
- Use keyboard shortcuts to speed up playback/rewinding
- Check for speaker label accuracy
- Verify timestamps align with audio

NEXT STEPS:
- Export final transcript to desired format
- Store in archive/database for future reference
- Consider feedback loop: track corrections for model improvement

Future enhancement: Web-based inline editor with integrated player
"""
    
    logger.info(f"Exported SRT for manual editing at: {srt_file_path}")
    return instructions


def launch_editor(
    srt_file_path: Path,
    audio_file_path: Path,
    config: Optional[dict] = None
) -> None:
    """
    Launch external editor for manual review.
    
    Currently prints instructions; future version will open web UI.
    
    Args:
        srt_file_path: Path to SRT file
        audio_file_path: Path to audio file
        config: Configuration dict
    """
    logger.info("Launching manual editor...")
    
    instructions = export_for_manual_editing(srt_file_path, audio_file_path, config)
    print(instructions)
    
    # Future: Open web UI with:
    # - Waveform viewer (wavesurfer.js)
    # - Inline subtitle editor
    # - Speaker color coding
    # - Confidence visualization
    # - Integrated audio player
    logger.info("Note: Web-based editor is a future enhancement")
