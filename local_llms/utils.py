import os
import pickle
import shutil
import hashlib
import logging
import subprocess
import tempfile
import asyncio
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_HASH_ALGO = "sha256"
DEFAULT_COMPRESSION_THREADS = 4
DEFAULT_ZIP_CHUNK_SIZE = 128  # MB

class CompressionError(Exception):
    """Raised when compression operations fail."""
    pass

class ExtractionError(Exception):
    """Raised when extraction operations fail."""
    pass

class FileOperationError(Exception):
    """Raised when file operations fail."""
    pass

@lru_cache(maxsize=128)
def get_command_path(cmd_name: str) -> Optional[str]:
    """Get the path of a command from environment variables with caching."""
    return os.environ.get(f"{cmd_name.upper()}_COMMAND")

def validate_commands() -> None:
    """Validate that all required commands are available."""
    required_commands = ["CAT", "PIGZ", "TAR"]
    missing_commands = []
    
    for cmd in required_commands:
        if not get_command_path(cmd):
            missing_commands.append(cmd)
    
    if missing_commands:
        raise RuntimeError(f"Missing required commands: {', '.join(missing_commands)}")

def compress_folder(
    model_folder: str,
    zip_chunk_size: int = DEFAULT_ZIP_CHUNK_SIZE,
    threads: int = DEFAULT_COMPRESSION_THREADS
) -> str:
    """
    Compress a folder into split parts using tar, pigz, and split.
    
    Args:
        model_folder: Path to the folder to compress
        zip_chunk_size: Size of each split part in MB
        threads: Number of compression threads
        
    Returns:
        Path to the temporary directory containing the split parts
        
    Raises:
        CompressionError: If compression fails
    """
    temp_dir = tempfile.mkdtemp()
    try:
        output_prefix = os.path.join(temp_dir, os.path.basename(model_folder) + ".zip.part-")
        tar_cmd = get_command_path("TAR")
        pigz_cmd = get_command_path("PIGZ")
        
        if not (tar_cmd and pigz_cmd):
            raise CompressionError("Required compression commands not found")
            
        tar_command = (
            f"{tar_cmd} -cf - '{model_folder}' | "
            f"{pigz_cmd} --best -p {threads} | "
            f"split -b {zip_chunk_size}M - '{output_prefix}'"
        )
        
        result = subprocess.run(
            tar_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Compression completed successfully: {result.stdout}")
        return temp_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Compression failed: {e.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise CompressionError(f"Compression failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during compression: {str(e)}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise CompressionError(f"Unexpected error during compression: {str(e)}")

def extract_zip(paths: List[Path], target_dir: Optional[Path] = None) -> None:
    """
    Extract zip files using parallel decompression.
    
    Args:
        paths: List of paths to zip files
        target_dir: Optional target directory for extraction
        
    Raises:
        ExtractionError: If extraction fails
    """
    try:
        # Use provided target directory or current directory
        target_dir = target_dir or Path.cwd()
        target_dir = target_dir.absolute()
        
        logger.info(f"Extracting files to: {target_dir}")
        
        # Validate commands
        validate_commands()
        
        # Sort and prepare paths
        sorted_paths = sorted(paths, key=lambda p: str(p))
        paths_str = " ".join(f"'{p.absolute()}'" for p in sorted_paths)
        
        # Get command paths
        cat_cmd = get_command_path("CAT")
        pigz_cmd = get_command_path("PIGZ")
        tar_cmd = get_command_path("TAR")
        
        # Build and execute extraction command
        cpus = os.cpu_count() or 1
        extract_command = (
            f"{cat_cmd} {paths_str} | "
            f"{pigz_cmd} -p {cpus} -d | "
            f"{tar_cmd} -xf - -C '{target_dir}'"
        )
        
        result = subprocess.run(
            extract_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Extraction completed successfully: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e.stderr}")
        raise ExtractionError(f"Extraction failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during extraction: {str(e)}")
        raise ExtractionError(f"Unexpected error during extraction: {str(e)}")

def compute_file_hash(
    file_path: Union[str, Path],
    hash_algo: str = DEFAULT_HASH_ALGO,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """
    Compute the hash of a file using the specified algorithm.
    
    Args:
        file_path: Path to the file
        hash_algo: Hash algorithm to use
        chunk_size: Size of chunks to read
        
    Returns:
        Hex digest of the file hash
    """
    try:
        hash_func = getattr(hashlib, hash_algo)()
        file_path = Path(file_path)
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hash_func.update(chunk)
                
        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {str(e)}")
        raise FileOperationError(f"Error computing hash: {str(e)}")

async def async_move(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Asynchronously move a file or directory.
    
    Args:
        src: Source path
        dst: Destination path
        
    Raises:
        FileOperationError: If move operation fails
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.move, str(src), str(dst))
    except Exception as e:
        logger.error(f"Error moving {src} to {dst}: {str(e)}")
        raise FileOperationError(f"Error moving file: {str(e)}")

async def async_rmtree(path: Union[str, Path]) -> None:
    """
    Asynchronously remove a directory tree.
    
    Args:
        path: Path to the directory to remove
        
    Raises:
        FileOperationError: If removal fails
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.rmtree, str(path), True)
    except Exception as e:
        logger.error(f"Error removing directory {path}: {str(e)}")
        raise FileOperationError(f"Error removing directory: {str(e)}")

async def async_extract_zip(paths: List[Path], target_dir: Optional[Path] = None) -> None:
    """
    Asynchronously extract zip files.
    
    Args:
        paths: List of paths to zip files
        target_dir: Optional target directory for extraction
        
    Raises:
        ExtractionError: If extraction fails
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract_zip, paths, target_dir)
    except Exception as e:
        logger.error(f"Error extracting zip files: {str(e)}")
        raise ExtractionError(f"Error extracting zip files: {str(e)}")

def check_downloading() -> List[str]:
    """
    Check which files are currently being downloaded.
    
    Returns:
        List of file hashes that are being downloaded
    """
    try:
        tracking_path = os.environ["TRACKING_DOWNLOAD_HASHES"]
        if os.path.exists(tracking_path):
            with open(tracking_path, "rb") as f:
                return pickle.load(f)
        return []
    except Exception as e:
        logger.error(f"Error checking download status: {str(e)}")
        return []

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Size of the file in bytes
    """
    try:
        return Path(file_path).stat().st_size
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        raise FileOperationError(f"Error getting file size: {str(e)}")

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
