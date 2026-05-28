"""
Database and Logging Module

Tracks transcription history, performance metrics, and enables analysis of corrections.
Uses SQLite for local storage and easy querying.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from .utils import get_logger, ensure_dir

logger = get_logger("database")


class TranscriptionDatabase:
    """SQLite database for tracking transcriptions and corrections."""
    
    def __init__(self, db_path: Path):
        """Initialize database."""
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    duration_seconds REAL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT
                )
            """)
            
            # Transcriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    segment_id INTEGER,
                    start_time REAL,
                    end_time REAL,
                    original_text TEXT,
                    corrected_text TEXT,
                    confidence REAL,
                    speaker TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    corrected_at TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL,
                    stage TEXT,
                    duration_seconds REAL,
                    memory_mb REAL,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                )
            """)
            
            # Corrections table for tracking manual changes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcription_id INTEGER NOT NULL,
                    original_text TEXT,
                    corrected_text TEXT,
                    change_type TEXT,
                    correction_source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (transcription_id) REFERENCES transcriptions(id)
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def create_job(
        self,
        file_path: Path,
        duration_seconds: Optional[float] = None
    ) -> int:
        """Create a new transcription job entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO jobs (file_path, file_name, duration_seconds, status)
                VALUES (?, ?, ?, ?)
                """,
                (str(file_path), file_path.name, duration_seconds, "started")
            )
            conn.commit()
            job_id = cursor.lastrowid
            logger.debug(f"Created job {job_id} for {file_path.name}")
            return job_id
    
    def update_job_status(
        self,
        job_id: int,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update job status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE id = ?
                """,
                (status, error_message, job_id)
            )
            conn.commit()
    
    def log_transcription(
        self,
        job_id: int,
        model_name: str,
        segment_id: int,
        start_time: float,
        end_time: float,
        text: str,
        confidence: float,
        speaker: Optional[str] = None
    ) -> int:
        """Log a transcription segment."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transcriptions
                (job_id, model_name, segment_id, start_time, end_time, 
                 original_text, confidence, speaker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, model_name, segment_id, start_time, end_time, text, confidence, speaker)
            )
            conn.commit()
            return cursor.lastrowid
    
    def log_metric(
        self,
        job_id: int,
        stage: str,
        duration_seconds: float,
        memory_mb: Optional[float] = None
    ):
        """Log performance metrics for a stage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (job_id, stage, duration_seconds, memory_mb)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, stage, duration_seconds, memory_mb)
            )
            conn.commit()
            logger.debug(f"Logged metric for {stage}: {duration_seconds:.2f}s")
    
    def record_correction(
        self,
        transcription_id: int,
        original_text: str,
        corrected_text: str,
        change_type: str,
        correction_source: str = "manual"
    ):
        """Record a manual correction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO corrections
                (transcription_id, original_text, corrected_text, change_type, correction_source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (transcription_id, original_text, corrected_text, change_type, correction_source)
            )
            cursor.execute(
                """
                UPDATE transcriptions
                SET corrected_text = ?, corrected_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (corrected_text, transcription_id)
            )
            conn.commit()
            logger.debug(f"Recorded correction: {original_text} -> {corrected_text}")
    
    def get_job_stats(self, job_id: int) -> Dict[str, Any]:
        """Get statistics for a completed job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Job info
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            job = cursor.fetchone()
            
            if not job:
                return {}
            
            # Transcription stats
            cursor.execute(
                "SELECT COUNT(*) FROM transcriptions WHERE job_id = ?",
                (job_id,)
            )
            total_segments = cursor.fetchone()[0]
            
            cursor.execute(
                "SELECT COUNT(*) FROM corrections WHERE transcription_id IN (SELECT id FROM transcriptions WHERE job_id = ?)",
                (job_id,)
            )
            total_corrections = cursor.fetchone()[0]
            
            # Performance metrics
            cursor.execute(
                "SELECT stage, duration_seconds FROM metrics WHERE job_id = ?",
                (job_id,)
            )
            metrics = cursor.fetchall()
            
            return {
                "job_id": job[0],
                "file_name": job[2],
                "duration_seconds": job[3],
                "status": job[5],
                "total_segments": total_segments,
                "total_corrections": total_corrections,
                "metrics": {m[0]: m[1] for m in metrics}
            }
    
    def get_corrections_summary(self, job_id: int) -> Dict[str, int]:
        """Get summary of corrections by type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT change_type, COUNT(*) 
                FROM corrections 
                WHERE transcription_id IN (SELECT id FROM transcriptions WHERE job_id = ?)
                GROUP BY change_type
                """,
                (job_id,)
            )
            return dict(cursor.fetchall())
    
    def export_job_results(
        self,
        job_id: int,
        output_path: Path,
        format: str = "json"
    ):
        """Export job results to file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all transcriptions
            cursor.execute(
                """
                SELECT segment_id, start_time, end_time, original_text, 
                       corrected_text, confidence, speaker
                FROM transcriptions
                WHERE job_id = ?
                ORDER BY start_time
                """,
                (job_id,)
            )
            transcriptions = cursor.fetchall()
            
            if format == "json":
                data = {
                    "job_id": job_id,
                    "exported_at": datetime.now().isoformat(),
                    "segments": [
                        {
                            "segment_id": t[0],
                            "start": t[1],
                            "end": t[2],
                            "text": t[4] or t[3],  # Use corrected if available
                            "confidence": t[5],
                            "speaker": t[6]
                        }
                        for t in transcriptions
                    ]
                }
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results exported to {output_path}")


def init_database(output_dir: Path) -> TranscriptionDatabase:
    """Initialize and return database instance."""
    db_path = output_dir / "transcriptions.db"
    return TranscriptionDatabase(db_path)
