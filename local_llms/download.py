import os
import requests
import aiohttp
import asyncio
import pickle
from tqdm import tqdm
from pathlib import Path
from local_llms.utils import compute_file_hash, async_extract_zip, async_move, async_rmtree
import random
import threading
import time
from typing import List, Dict, Tuple, Optional, Union
import logging

# Constants
GATEWAY_URL = "https://gateway.lighthouse.storage/ipfs/"
DEFAULT_OUTPUT_DIR = Path.cwd() / "llms-storage"
SLEEP_TIME = 60
MAX_ATTEMPTS = 10
CHUNK_SIZE = 1024*1024
POSTFIX_MODEL_PATH = ".gguf"
MAX_FILE_SIZE = 600 * 1024 * 1024  # 600MB in bytes
FLUSH_FREQUENCY = 0.1  # Flush to disk ~10% of chunks
CHUNK_TIMEOUT = 300  # 300 seconds without data is a timeout

# Create logger with NullHandler by default
# This prevents duplicate logging and follows best practices
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Global download progress tracker
class DownloadProgressTracker:
    def __init__(self):
        self.total_bytes_downloaded = 0
        self.total_bytes_to_download = 0
        self.lock = threading.Lock()
        self.start_time = None
        self.last_update_time = None
        self.last_bytes_downloaded = 0
        self.download_speed = 0
        self._last_progress_print = 0
        self._progress_print_interval = 1.0
        self.prev_index = -1

    def initialize(self, total_bytes, filecoin_hash, num_of_files):
        with self.lock:
            self.total_bytes_to_download = total_bytes
            self.filecoin_hash = filecoin_hash
            self.num_of_files = num_of_files
            self.total_bytes_downloaded = 0
            self.start_time = time.time()
            self.last_update_time = self.start_time
            self.last_bytes_downloaded = 0
            self.download_speed = 0
            self._last_progress_print = self.start_time
            logger.info(f"Initialized tracker for {num_of_files} files, {total_bytes} bytes.")

    def update(self, bytes_downloaded):
        with self.lock:
            self.total_bytes_downloaded += bytes_downloaded
            current_time = time.time()
            if current_time - self.last_update_time >= 1.0:
                time_diff = current_time - self.last_update_time
                bytes_diff = self.total_bytes_downloaded - self.last_bytes_downloaded
                self.download_speed = bytes_diff / time_diff
                self.last_update_time = current_time
                self.last_bytes_downloaded = self.total_bytes_downloaded

    def get_progress(self):
        with self.lock:
            if self.total_bytes_to_download == 0:
                downloaded_gb = self.total_bytes_downloaded / (1024 * 1024 * 1024)
                speed_mbps = self.download_speed / (1024 * 1024)
                return 0, downloaded_gb, 0, speed_mbps
            progress_percent = (self.total_bytes_downloaded / self.total_bytes_to_download) * 100
            downloaded_gb = self.total_bytes_downloaded / (1024 * 1024 * 1024)
            total_gb = self.total_bytes_to_download / (1024 * 1024 * 1024)
            speed_mbps = self.download_speed / (1024 * 1024)
            return progress_percent, downloaded_gb, total_gb, speed_mbps

    def should_print_progress(self):
        current_time = time.time()
        if current_time - self._last_progress_print >= self._progress_print_interval:
            self._last_progress_print = current_time
            return True
        return False

    def print_progress(self):
        if not self.should_print_progress():
            return
        progress_percent, _, total_gb, _ = self.get_progress()
        if total_gb == 0:
            logger.error("Total size unknown")
        else:
            estimate_num_of_downloaded_files = int((progress_percent / 100) * self.num_of_files)
            if self.prev_index < estimate_num_of_downloaded_files:
                self.prev_index = estimate_num_of_downloaded_files
                print(f"\n[LAUNCHER_LOGGER] [MODEL_INSTALL] --step {estimate_num_of_downloaded_files}/{self.num_of_files} --hash {self.filecoin_hash} --percent {progress_percent:.1f}%")

# Global tracker instance
download_tracker = DownloadProgressTracker()

def check_downloaded_model(filecoin_hash: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    """
    Check if the model is already downloaded and optionally save metadata.
    
    Args:
        filecoin_hash: IPFS hash of the model metadata
        output_file: Optional path to save metadata JSON
    
    Returns:
        bool: Whether the model is already downloaded
    """    
    try:
        local_path = output_dir / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
        local_path = local_path.absolute()
        
        is_downloaded = local_path.exists()
            
        if is_downloaded:
            logger.info(f"Model already exists at: {local_path}")
            
        return is_downloaded
    except requests.RequestException as e:
        logger.error(f"Failed to fetch model metadata: {e}")
        return False

async def download_single_file_async(session: aiohttp.ClientSession, file_info: dict, folder_path: Path, max_attempts: int = MAX_ATTEMPTS) -> tuple:
    """
    Asynchronously download a single file and verify its SHA256 hash, with retries.

    Args:
        session (aiohttp.ClientSession): Reusable HTTP session.
        file_info (dict): Contains 'cid', 'file_hash', and 'file_name'.
        folder_path (Path): Directory to save the file.
        max_attempts (int): Number of retries on failure.

    Returns:
        tuple: (Path to file if successful, None) or (None, error message).
    """
    cid = file_info["cid"]
    expected_hash = file_info["file_hash"]
    file_name = file_info["file_name"]
    file_path = folder_path / file_name
    attempts = 0
    
    # Try to use a temp file for download to avoid corrupt files on failure
    temp_path = folder_path / f"{file_name}.tmp"
    
    # Check if file already exists with correct hash
    if file_path.exists():
        try:
            computed_hash = compute_file_hash(file_path)
            if computed_hash == expected_hash:
                download_tracker.update(file_path.stat().st_size)
                logger.info(f"File {cid} already exists with correct hash.")
                return file_path, None
            else:
                logger.info(f"File {cid} exists but hash mismatch. Will re-download from scratch.")
                file_path.unlink(missing_ok=True)
                # Also remove any temporary file to ensure a clean download
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error checking existing file {cid}: {e}")
            file_path.unlink(missing_ok=True)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    # Always start from position 0 - no resuming of partial downloads
    resume_position = 0
    if temp_path.exists():
        logger.info(f"Removing existing temporary file for {file_name} to ensure clean download.")
        temp_path.unlink(missing_ok=True)

    while attempts < max_attempts:
        try:
            url = f"{GATEWAY_URL}{cid}"
            
            # Set up headers for resume if needed
            headers = {}
            if resume_position > 0:
                headers['Range'] = f'bytes={resume_position}-'
            
            # Use a longer timeout for large files
            timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=120, sock_connect=60)
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status in (200, 206):
                    # Get total size accounting for resumed downloads
                    total_size = int(response.headers.get("content-length", 0))
                    if response.status == 206:
                        # For partial content, content-length is the remaining bytes
                        content_range = response.headers.get("content-range", "")
                        if content_range:
                            try:
                                # Format is usually "bytes start-end/total"
                                total_size = int(content_range.split("/")[1]) 
                            except (IndexError, ValueError):
                                # If parsing fails, use resume_position + content-length
                                total_size += resume_position
                    
                    # Check if file size exceeds the maximum allowed size
                    if total_size > MAX_FILE_SIZE:
                        error_msg = f"File {cid} exceeds maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.0f}MB (actual: {total_size / (1024 * 1024):.0f}MB)"
                        logger.error(error_msg)
                        # Delete any partial download
                        if temp_path.exists():
                            temp_path.unlink(missing_ok=True)
                        return None, error_msg
                    
                    # Track downloaded bytes to ensure we don't exceed the limit
                    downloaded_bytes = resume_position
                    
                    # Add timeout protection for each chunk
                    last_data_time = time.time()
                    
                    # Open file in append mode if resuming, otherwise in write mode
                    mode = "ab" if resume_position > 0 else "wb"
                    with temp_path.open(mode) as f:
                        # Downloading with per-chunk timeout protection
                        flush_interval = 10
                        chunk_count = 0
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            # Reset timeout timer when data is received
                            last_data_time = time.time()
                            
                            # Check if this chunk would exceed the maximum file size
                            chunk_size_bytes = len(chunk)
                            if downloaded_bytes + chunk_size_bytes > MAX_FILE_SIZE:
                                error_msg = f"File {cid} would exceed maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
                                logger.error(error_msg)
                                # Delete the partial download
                                if temp_path.exists():
                                    temp_path.unlink(missing_ok=True)
                                return None, error_msg
                            
                            # Write chunk and update progress
                            f.write(chunk)
                            
                            # Update global download tracker
                            download_tracker.update(chunk_size_bytes)
                            
                            # Update downloaded bytes counter
                            downloaded_bytes += chunk_size_bytes
                            
                            # Flush to disk periodically
                            chunk_count += 1
                            if chunk_count % flush_interval == 0:
                                f.flush()
                                os.fsync(f.fileno())
                            
                            # Check if download has been idle
                            current_time = time.time()
                            if current_time - last_data_time > CHUNK_TIMEOUT:
                                raise asyncio.TimeoutError(f"No data received for {CHUNK_TIMEOUT} seconds")

                    # Verify hash
                    computed_hash = compute_file_hash(temp_path)
                    if computed_hash == expected_hash:
                        # Rename temp file to final file only after successful verification
                        if file_path.exists():
                            file_path.unlink()
                        temp_path.rename(file_path)
                        logger.info(f"File {cid} downloaded and verified successfully.")
                        download_tracker.update(file_path.stat().st_size)
                        return file_path, None
                    else:
                        logger.error(f"Hash mismatch for {cid}. Expected {expected_hash}, got {computed_hash}.")
                        # Always delete temp file on hash mismatch and start over on next attempt
                        temp_path.unlink(missing_ok=True)
                        download_tracker.update(-temp_path.stat().st_size)
                        resume_position = 0
                else:
                    logger.error(f"Failed to download {cid}. Status: {response.status}")
                    # For certain status codes, we might want to fail faster
                    if response.status in (404, 403, 401):
                        wait_time = SLEEP_TIME
                    else:
                        wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {cid} - will resume from position {resume_position}")
            # On timeout, don't reset resume position - we'll keep what we have
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except aiohttp.ClientError as e:
            logger.error(f"Client error downloading {cid}: {e}")
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except Exception as e:
            logger.error(f"Exception downloading {cid}: {e}")
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)

        attempts += 1
        if attempts < max_attempts:
            logger.info(f"Retrying in {wait_time}s (Attempt {attempts + 1}/{max_attempts})")
            download_tracker.update(-temp_path.stat().st_size)
            temp_path.unlink(missing_ok=True)
            await asyncio.sleep(wait_time)
        else:
            logger.error(f"Failed to download {cid} after {max_attempts} attempts.")
            # On final failure, leave the temp file for potential future resume
            return None, f"Failed to download {cid} after {max_attempts} attempts."

async def download_files_from_lighthouse_async(data: dict) -> list:
    """
    Asynchronously download files concurrently using Filecoin CIDs and verify hashes.

    Args:
        data (dict): JSON data with 'folder_name', 'files', 'num_of_files', and 'filecoin_hash'.

    Returns:
        list: Paths of successfully downloaded files, or empty list if failed.
    """
    folder_name = data["folder_name"]
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    num_of_files = data["num_of_files"]
    filecoin_hash = data["filecoin_hash"]
    files = data["files"]
    
    # Calculate total size for progress indication
    total_files = len(files)
    
    # Calculate total bytes to download - use actual file sizes if available
    total_bytes = 0
    if "file_sizes" in data and isinstance(data["file_sizes"], dict):
        # Use actual file sizes if provided
        for file_info in files:
            cid = file_info["cid"]
            if cid in data["file_sizes"]:
                total_bytes += data["file_sizes"][cid]
            else:
                # Fallback to estimate if size not available
                total_bytes += 512 * 1024 * 1024  # 512 MB estimate
    else:
        # Fallback to estimate
        total_bytes = total_files * 512 * 1024 * 1024  # 512 MB per file estimate

    # Initialize the download tracker
    download_tracker.initialize(total_bytes, filecoin_hash, num_of_files)
    
    # Use semaphore to limit concurrent downloads
    minimum_workers = min(os.cpu_count() * 2, 8)
    max_concurrent_downloads = min(minimum_workers, num_of_files)
    logger.info(f"Max concurrent downloads set to {max_concurrent_downloads}")
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    
    # Wrapper for download with semaphore
    async def download_with_semaphore(session, file_info, folder_path):
        async with semaphore:
            return await download_single_file_async(session, file_info, folder_path)
    
    connector = aiohttp.TCPConnector(limit=max_concurrent_downloads, ssl=False)
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_connect=60, sock_read=120)
    
    logger.info(f"Downloading {total_files} files with max {max_concurrent_downloads} concurrent downloads")
    logger.info(f"Maximum file size limit: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB per file")
    
    # Start a task to periodically print the total progress
    async def print_total_progress():
        while True:
            download_tracker.print_progress()
            await asyncio.sleep(1)
    
    progress_task = asyncio.create_task(print_total_progress())
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks
        tasks = [
            download_with_semaphore(session, file_info, folder_path)
            for file_info in files
        ]
        
        # Track overall progress
        successful_downloads = []
        failed_downloads = []
        size_limit_failures = []
        
        # Use as_completed to process files as they complete
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            try:
                path, error = await future
                if path:
                    successful_downloads.append(path)
                else:
                    failed_downloads.append(error)
                    # Check if this was a size limit failure
                    if error and "exceeds maximum allowed size" in error:
                        size_limit_failures.append(error)
                    logger.error(f"\nDownload failed: {error}")
            except Exception as e:
                logger.error(f"\nUnexpected error in download task: {e}")
                failed_downloads.append(str(e))
        
        # Cancel the progress printing task
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        
        # Print final progress
        download_tracker.print_progress()
        logger.info("")  # New line after progress
        
        # Check if all downloads were successful
        if len(successful_downloads) == num_of_files:
            logger.info(f"All {num_of_files} files downloaded successfully.")
            return successful_downloads
        else:
            logger.info(f"Downloaded {len(successful_downloads)} out of {num_of_files} files.")
            if size_limit_failures:
                logger.info(f"Size limit failures ({len(size_limit_failures)}):")
                for i, error in enumerate(size_limit_failures[:5], 1):
                    logger.info(f"  {i}. {error}")
                if len(size_limit_failures) > 5:
                    logger.info(f"  ... and {len(size_limit_failures) - 5} more size limit failures")
            
            if failed_downloads:
                logger.error(f"Failed downloads ({len(failed_downloads)}):")
                for i, error in enumerate(failed_downloads[:5], 1):
                    logger.error(f"  {i}. {error}")
                if len(failed_downloads) > 5:
                    logger.error(f"  ... and {len(failed_downloads) - 5} more errors")
            return successful_downloads if successful_downloads else []

async def download_model_from_filecoin_async(filecoin_hash: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> str | None:
    """
    Asynchronously download a model from Filecoin using its IPFS hash.

    Args:
        filecoin_hash (str): IPFS hash of the model metadata.
        output_dir (Path): Directory to save the downloaded model.

    Returns:
        str | None: Path to the downloaded model if successful, None otherwise.
    """
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    local_path = output_dir / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
    local_path_str = str(local_path.absolute())

    # Check if model is already downloaded
    if check_downloaded_model(filecoin_hash, output_dir):
        logger.info(f"Using existing model at {local_path_str}")
        return local_path_str

    # Track downloads in progress
    tracking_path = os.environ["TRACKING_DOWNLOAD_HASHES"]
    downloading_files = []
    
    try:
        if os.path.exists(tracking_path):
            with open(tracking_path, "rb") as f:
                downloading_files = pickle.load(f)
    except Exception as e:
        logger.error(f"Error reading tracking file: {e}")
        downloading_files = []

    # Add current hash to tracking
    if filecoin_hash not in downloading_files:
        downloading_files.append(filecoin_hash)
        with open(tracking_path, "wb") as f:
            pickle.dump(downloading_files, f)

    # Define input link
    input_link = f"{GATEWAY_URL}{filecoin_hash}"
    
    # Setup more robust session parameters
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    connector = aiohttp.TCPConnector(limit=4, ssl=False)
    
    # Initialize variables outside the loop
    folder_path = None
    extracted_files = []
    
    try:
        # Use exponential backoff for retries
        for attempt in range(1, MAX_ATTEMPTS + 1):
            backoff = min(SLEEP_TIME * (2 ** (attempt - 1)), 300)
            
            try:
                logger.info(f"Downloading model metadata (attempt {attempt}/{MAX_ATTEMPTS})")
                
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    async with session.get(input_link) as response:
                        if response.status != 200:
                            logger.error(f"Failed to fetch metadata: HTTP {response.status}")
                            if attempt < MAX_ATTEMPTS:
                                logger.info(f"Retrying in {backoff} seconds")
                                await asyncio.sleep(backoff)
                                continue
                            else:
                                raise Exception(f"Failed to fetch metadata after {MAX_ATTEMPTS} attempts")
                        
                        # Parse metadata
                        data = await response.json()
                        data["filecoin_hash"] = filecoin_hash
                        folder_name = data["folder_name"]
                        folder_path = Path.cwd() / folder_name
                        
                        # Create folder if it doesn't exist
                        folder_path.mkdir(exist_ok=True, parents=True)
                
                # Download files
                paths = await download_files_from_lighthouse_async(data)
                if not paths:
                    logger.error("Failed to download model files")
                    if attempt < MAX_ATTEMPTS:
                        logger.info(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception("Failed to download model files after all attempts")
                
                # Check if any files failed due to size limits
                if len(paths) < data["num_of_files"]:
                    logger.error(f"Some files failed to download: {len(paths)}/{data['num_of_files']} files downloaded")
                    if attempt < MAX_ATTEMPTS:
                        logger.info(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Failed to download all files after all attempts: {len(paths)}/{data['num_of_files']} files downloaded")
                
                # Track extracted files for cleanup
                extracted_files = paths
                
                # Extract files
                try:
                    logger.info("Extracting downloaded files...")
                    await async_extract_zip(paths)
                except Exception as e:
                    logger.error(f"Failed to extract files: {e}")
                    if attempt < MAX_ATTEMPTS:
                        logger.info(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Failed to extract files after {MAX_ATTEMPTS} attempts: {e}")
                
                # Move files to final location
                try:
                    source_text_path = folder_path / folder_name
                    source_text_path = source_text_path.absolute()
                    logger.info(f"Moving model to {local_path_str}")
                    
                    if source_text_path.exists():
                        # Handle projector path for multimodal models
                        source_projector_path = folder_path / (folder_name + "-projector")
                        source_projector_path = source_projector_path.absolute()
                        
                        if source_projector_path.exists():
                            projector_dest = local_path_str + "-projector"
                            logger.info(f"Moving projector to {projector_dest}")
                            await async_move(str(source_projector_path), projector_dest)
                        
                        # Move model to final location
                        await async_move(str(source_text_path), local_path_str)
                        
                        # Clean up folder after successful move
                        if folder_path.exists():
                            logger.info(f"Cleaning up temporary folder {folder_path}")
                            await async_rmtree(str(folder_path))
                        
                        logger.info(f"Model download complete: {local_path_str}")
                        
                        # Print download summary
                        progress_percent, downloaded_gb, total_gb, speed_mbps = download_tracker.get_progress()
                        elapsed_time = time.time() - download_tracker.start_time
                        hours, remainder = divmod(elapsed_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        
                        logger.info("\nDownload Summary:")
                        logger.info(f"  Total Size: {total_gb:.2f} GB")
                        logger.info(f"  Download Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
                        logger.info(f"  Average Speed: {speed_mbps:.2f} MB/s")
                        
                        # Update tracking file to remove this hash
                        with open(tracking_path, "wb") as f:
                            pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
                        
                        return local_path_str
                    else:
                        logger.error(f"Model not found at {source_text_path}")
                        if attempt < MAX_ATTEMPTS:
                            logger.info(f"Retrying in {backoff} seconds")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise Exception(f"Model not found at {source_text_path} after all attempts")
                except Exception as e:
                    logger.error(f"Failed to move model: {e}")
                    if attempt < MAX_ATTEMPTS:
                        logger.info(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Failed to move model after {MAX_ATTEMPTS} attempts: {e}")
            
            except aiohttp.ClientError as e:
                logger.error(f"HTTP error on attempt {attempt}: {e}")
                if attempt < MAX_ATTEMPTS:
                    logger.info(f"Retrying in {backoff} seconds")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    raise Exception(f"HTTP error after {MAX_ATTEMPTS} attempts: {e}")
            except Exception as e:
                logger.error(f"Download attempt {attempt} failed: {e}")
                if attempt < MAX_ATTEMPTS:
                    logger.info(f"Retrying in {backoff} seconds")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    raise Exception(f"Download failed after {MAX_ATTEMPTS} attempts: {e}")
    
    except Exception as e:
        logger.error(f"Download failed: {e}")
    finally:
        # Always clean up tracking file
        try:
            with open(tracking_path, "wb") as f:
                pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
        except Exception as e:
            logger.error(f"Error updating tracking file: {e}")

    logger.error("All download attempts failed")
    return None