"""
Player layout discovery for Pupil Labs export directories.

Provides utilities for discovering and organizing Pupil Labs export files
from various export formats and directory structures.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..logging_setup import get_logger

logger = get_logger(__name__)

# Common file patterns for Pupil exports
GAZE_PATTERNS = [
    "gaze.csv",
    "gaze_data.csv", 
    "gaze_positions.csv",
    "pupil_positions.csv",
    "gaze_positions.csv",
    "exported_data/gaze.csv",
    "exports/gaze.csv",
]

WORLD_VIDEO_PATTERNS = [
    "world.mp4",
    "world.avi",
    "world.mov",
    "world_viz.mp4",
    "world_camera.mp4",
    "world_camera.avi",
    "world_camera.mov",
    "exported_data/world.mp4",
    "exports/world.mp4",
]

METADATA_PATTERNS = [
    "info.csv",
    "info.json",
    "session_info.csv",
    "recording_info.csv",
    "metadata.json",
    "exported_data/info.csv",
    "exports/info.csv",
]

FIXATION_PATTERNS = [
    "fixations.csv",
    "fixation_data.csv",
    "fixations_ivt.csv",
    "exported_data/fixations.csv",
    "exports/fixations.csv",
]


class SessionFiles:
    """Container for discovered session files."""
    
    def __init__(
        self,
        session_dir: Path,
        gaze_file: Optional[Path] = None,
        world_video: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
        fixation_file: Optional[Path] = None,
        additional_files: Optional[List[Path]] = None,
    ):
        self.session_dir = session_dir
        self.gaze_file = gaze_file
        self.world_video = world_video
        self.metadata_file = metadata_file
        self.fixation_file = fixation_file
        self.additional_files = additional_files or []
    
    def is_complete(self) -> bool:
        """Check if session has required files."""
        return self.gaze_file is not None and self.world_video is not None
    
    def has_gaze_data(self) -> bool:
        """Check if session has gaze data."""
        return self.gaze_file is not None
    
    def has_video(self) -> bool:
        """Check if session has world video."""
        return self.world_video is not None
    
    def get_session_name(self) -> str:
        """Get session name from directory."""
        return self.session_dir.name
    
    def __repr__(self) -> str:
        return (
            f"SessionFiles(session_dir={self.session_dir}, "
            f"gaze_file={self.gaze_file}, "
            f"world_video={self.world_video}, "
            f"metadata_file={self.metadata_file})"
        )


def find_files_by_patterns(
    directory: Path, 
    patterns: List[str],
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching patterns in directory.
    
    Args:
        directory: Directory to search
        patterns: List of file patterns to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    found_files = []
    
    for pattern in patterns:
        if recursive:
            matches = list(directory.rglob(pattern))
        else:
            matches = list(directory.glob(pattern))
        
        found_files.extend(matches)
    
    return found_files


def discover_session_files(
    root_path: Union[str, Path],
    recursive: bool = True
) -> List[SessionFiles]:
    """
    Discover Pupil Labs session files in directory structure.
    
    Args:
        root_path: Root directory to search
        recursive: Whether to search recursively
        
    Returns:
        List of SessionFiles objects
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")
    
    if not root_path.is_dir():
        raise ValueError(f"Root path is not a directory: {root_path}")
    
    sessions = []
    
    if recursive:
        # Find all potential session directories
        potential_dirs = [root_path]
        
        # Look for subdirectories that might contain session data
        for item in root_path.rglob("*"):
            if item.is_dir():
                # Check if directory contains gaze or video files
                gaze_files = find_files_by_patterns(item, GAZE_PATTERNS, recursive=False)
                video_files = find_files_by_patterns(item, WORLD_VIDEO_PATTERNS, recursive=False)
                
                if gaze_files or video_files:
                    potential_dirs.append(item)
    else:
        potential_dirs = [root_path]
    
    # Process each potential session directory
    for session_dir in potential_dirs:
        session_files = _discover_single_session(session_dir)
        if session_files:
            sessions.append(session_files)
    
    logger.info(f"Discovered {len(sessions)} sessions in {root_path}")
    
    return sessions


def _discover_single_session(session_dir: Path) -> Optional[SessionFiles]:
    """
    Discover files for a single session directory.
    
    Args:
        session_dir: Directory to search
        
    Returns:
        SessionFiles object or None if no relevant files found
    """
    # Find gaze data file
    gaze_files = find_files_by_patterns(session_dir, GAZE_PATTERNS, recursive=False)
    gaze_file = gaze_files[0] if gaze_files else None
    
    # Find world video file
    video_files = find_files_by_patterns(session_dir, WORLD_VIDEO_PATTERNS, recursive=False)
    world_video = video_files[0] if video_files else None
    
    # Find metadata file
    metadata_files = find_files_by_patterns(session_dir, METADATA_PATTERNS, recursive=False)
    metadata_file = metadata_files[0] if metadata_files else None
    
    # Find fixation file
    fixation_files = find_files_by_patterns(session_dir, FIXATION_PATTERNS, recursive=False)
    fixation_file = fixation_files[0] if fixation_files else None
    
    # Find additional files (CSV, JSON, etc.)
    additional_files = []
    for ext in ["*.csv", "*.json", "*.txt"]:
        additional_files.extend(session_dir.glob(ext))
    
    # Remove files we've already categorized
    categorized_files = {gaze_file, world_video, metadata_file, fixation_file}
    additional_files = [f for f in additional_files if f not in categorized_files]
    
    # Only create session if we have at least gaze data or video
    if gaze_file or world_video:
        return SessionFiles(
            session_dir=session_dir,
            gaze_file=gaze_file,
            world_video=world_video,
            metadata_file=metadata_file,
            fixation_file=fixation_file,
            additional_files=additional_files,
        )
    
    return None


def get_session_info(session_files: SessionFiles) -> Dict[str, any]:
    """
    Get information about a session from its files.
    
    Args:
        session_files: SessionFiles object
        
    Returns:
        Dictionary with session information
    """
    info = {
        "session_name": session_files.get_session_name(),
        "session_dir": str(session_files.session_dir),
        "has_gaze_data": session_files.has_gaze_data(),
        "has_video": session_files.has_video(),
        "is_complete": session_files.is_complete(),
    }
    
    if session_files.gaze_file:
        info["gaze_file"] = str(session_files.gaze_file)
        info["gaze_file_size"] = session_files.gaze_file.stat().st_size
    
    if session_files.world_video:
        info["world_video"] = str(session_files.world_video)
        info["world_video_size"] = session_files.world_video.stat().st_size
    
    if session_files.metadata_file:
        info["metadata_file"] = str(session_files.metadata_file)
    
    if session_files.fixation_file:
        info["fixation_file"] = str(session_files.fixation_file)
    
    info["additional_files"] = [str(f) for f in session_files.additional_files]
    
    return info


def validate_session_files(session_files: SessionFiles) -> List[str]:
    """
    Validate session files and return list of issues.
    
    Args:
        session_files: SessionFiles object to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check if session directory exists
    if not session_files.session_dir.exists():
        issues.append(f"Session directory does not exist: {session_files.session_dir}")
        return issues
    
    # Check gaze file
    if session_files.gaze_file:
        if not session_files.gaze_file.exists():
            issues.append(f"Gaze file does not exist: {session_files.gaze_file}")
        elif session_files.gaze_file.stat().st_size == 0:
            issues.append(f"Gaze file is empty: {session_files.gaze_file}")
    else:
        issues.append("No gaze data file found")
    
    # Check world video
    if session_files.world_video:
        if not session_files.world_video.exists():
            issues.append(f"World video does not exist: {session_files.world_video}")
        elif session_files.world_video.stat().st_size == 0:
            issues.append(f"World video is empty: {session_files.world_video}")
    else:
        issues.append("No world video file found")
    
    # Check metadata file
    if session_files.metadata_file and not session_files.metadata_file.exists():
        issues.append(f"Metadata file does not exist: {session_files.metadata_file}")
    
    # Check fixation file
    if session_files.fixation_file and not session_files.fixation_file.exists():
        issues.append(f"Fixation file does not exist: {session_files.fixation_file}")
    
    return issues


def find_best_session(
    sessions: List[SessionFiles],
    prefer_complete: bool = True
) -> Optional[SessionFiles]:
    """
    Find the best session from a list of sessions.
    
    Args:
        sessions: List of SessionFiles objects
        prefer_complete: Whether to prefer complete sessions
        
    Returns:
        Best SessionFiles object or None if no sessions
    """
    if not sessions:
        return None
    
    if prefer_complete:
        # Filter to complete sessions first
        complete_sessions = [s for s in sessions if s.is_complete()]
        if complete_sessions:
            sessions = complete_sessions
    
    # Sort by session name (most recent first if using timestamp names)
    sessions.sort(key=lambda s: s.get_session_name(), reverse=True)
    
    return sessions[0]


def export_session_summary(
    sessions: List[SessionFiles],
    output_file: Optional[Path] = None
) -> str:
    """
    Export a summary of discovered sessions.
    
    Args:
        sessions: List of SessionFiles objects
        output_file: Optional file to write summary to
        
    Returns:
        Summary text
    """
    summary_lines = [
        f"Pupil Labs Session Discovery Summary",
        f"Found {len(sessions)} sessions",
        f"=" * 50,
        ""
    ]
    
    for i, session in enumerate(sessions, 1):
        info = get_session_info(session)
        issues = validate_session_files(session)
        
        summary_lines.extend([
            f"Session {i}: {info['session_name']}",
            f"  Directory: {info['session_dir']}",
            f"  Complete: {info['is_complete']}",
            f"  Gaze Data: {info['has_gaze_data']}",
            f"  Video: {info['has_video']}",
        ])
        
        if info.get('gaze_file'):
            summary_lines.append(f"  Gaze File: {info['gaze_file']}")
        
        if info.get('world_video'):
            summary_lines.append(f"  World Video: {info['world_video']}")
        
        if issues:
            summary_lines.append(f"  Issues: {', '.join(issues)}")
        
        summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(summary_text)
        logger.info(f"Session summary written to {output_file}")
    
    return summary_text
