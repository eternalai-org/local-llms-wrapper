import os
import shutil
import time
import httpx
import requests
import time
import aiohttp
import asyncio
import pickle
from tqdm import tqdm
from pathlib import Path
from local_llms.utils import compute_file_hash, async_extract_zip, async_move, async_rmtree

# Constants
GATEWAY_URL = "https://gateway.lighthouse.storage/ipfs/"
DEFAULT_OUTPUT_DIR = Path.cwd() / "llms-storage"
SLEEP_TIME = 60
MAX_ATTEMPTS = 10
CHUNK_SIZE = 4096
POSTFIX_MODEL_PATH = ".gguf"

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
        
        # Check if model exists
        is_downloaded = local_path.exists()
            
        if is_downloaded:
            print(f"Model already exists at: {local_path}")
            
        return is_downloaded
        
    except requests.RequestException as e:
        print(f"Failed to fetch model metadata: {e}")
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
                print(f"File {cid} already exists with correct hash.")
                return file_path, None
            else:
                print(f"File {cid} exists but hash mismatch. Retrying...")
                file_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"Error checking existing file {cid}: {e}")
            file_path.unlink(missing_ok=True)

    # Clean up any temporary file from previous attempts
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)

    while attempts < max_attempts:
        try:
            url = f"{GATEWAY_URL}{cid}"
            
            # Use a larger chunk size for faster downloads
            chunk_size = 1024 * 1024  # 1MB chunks
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    total_size = int(response.headers.get("content-length", 0))
                    
                    with temp_path.open("wb") as f, tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {file_name}",
                        ncols=80
                    ) as progress:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            progress.update(len(chunk))

                    # Verify hash
                    computed_hash = compute_file_hash(temp_path)
                    if computed_hash == expected_hash:
                        # Rename temp file to final file only after successful verification
                        if file_path.exists():
                            file_path.unlink()
                        temp_path.rename(file_path)
                        print(f"File {cid} downloaded and verified successfully.")
                        return file_path, None
                    else:
                        print(f"Hash mismatch for {cid}. Expected {expected_hash}, got {computed_hash}.")
                        temp_path.unlink(missing_ok=True)
                else:
                    print(f"Failed to download {cid}. Status: {response.status}")
                    # For certain status codes, we might want to fail faster
                    if response.status in (404, 403, 401):
                        wait_time = SLEEP_TIME
                    else:
                        wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
            
        except asyncio.TimeoutError:
            print(f"Timeout downloading {cid}")
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except aiohttp.ClientError as e:
            print(f"Client error downloading {cid}: {e}")
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        except Exception as e:
            print(f"Exception downloading {cid}: {e}")
            wait_time = min(SLEEP_TIME * (2 ** attempts), 300)
        finally:
            # Clean up temp file on failure
            if temp_path.exists() and not file_path.exists():
                temp_path.unlink(missing_ok=True)

        attempts += 1
        if attempts < max_attempts:
            print(f"Retrying in {wait_time}s (Attempt {attempts + 1}/{max_attempts})")
            await asyncio.sleep(wait_time)
        else:
            print(f"Failed to download {cid} after {max_attempts} attempts.")
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
    
    # Use semaphore to limit concurrent downloads
    max_concurrent_downloads = min(os.cpu_count() * 2, 8)
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    
    # Wrapper for download with semaphore
    async def download_with_semaphore(session, file_info, folder_path):
        async with semaphore:
            return await download_single_file_async(session, file_info, folder_path)
    
    connector = aiohttp.TCPConnector(limit=max_concurrent_downloads, ssl=False)
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_connect=60, sock_read=60)
    
    print(f"Downloading {total_files} files with max {max_concurrent_downloads} concurrent downloads")
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks
        tasks = [
            download_with_semaphore(session, file_info, folder_path)
            for file_info in files
        ]
        
        # Track overall progress
        successful_downloads = []
        failed_downloads = []
        
        # Use as_completed to process files as they complete
        for i, future in enumerate(asyncio.as_completed(tasks), 1):
            try:
                path, error = await future
                if path:
                    successful_downloads.append(path)
                    print(f"[LAUNCHER_LOGGER] [MODEL_INSTALL] --step {len(successful_downloads)}/{num_of_files} --hash {filecoin_hash}")
                    print(f"Progress: {len(successful_downloads)}/{total_files} files downloaded")
                else:
                    failed_downloads.append(error)
                    print(f"Download failed: {error}")
            except Exception as e:
                print(f"Unexpected error in download task: {e}")
                failed_downloads.append(str(e))
        
        # Check if all downloads were successful
        if len(successful_downloads) == num_of_files:
            print(f"All {num_of_files} files downloaded successfully.")
            return successful_downloads
        else:
            print(f"Downloaded {len(successful_downloads)} out of {num_of_files} files.")
            if failed_downloads:
                print(f"Failed downloads ({len(failed_downloads)}):")
                for i, error in enumerate(failed_downloads[:5], 1):
                    print(f"  {i}. {error}")
                if len(failed_downloads) > 5:
                    print(f"  ... and {len(failed_downloads) - 5} more errors")
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
        print(f"Using existing model at {local_path_str}")
        return local_path_str

    # Track downloads in progress
    tracking_path = os.environ["TRACKING_DOWNLOAD_HASHES"]
    downloading_files = []
    
    try:
        if os.path.exists(tracking_path):
            with open(tracking_path, "rb") as f:
                downloading_files = pickle.load(f)
    except Exception as e:
        print(f"Error reading tracking file: {e}")
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
                print(f"Downloading model metadata (attempt {attempt}/{MAX_ATTEMPTS})")
                
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    async with session.get(input_link) as response:
                        if response.status != 200:
                            print(f"Failed to fetch metadata: HTTP {response.status}")
                            if attempt < MAX_ATTEMPTS:
                                print(f"Retrying in {backoff} seconds")
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
                    print("Failed to download model files")
                    if attempt < MAX_ATTEMPTS:
                        print(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception("Failed to download model files after all attempts")
                
                # Track extracted files for cleanup
                extracted_files = paths
                
                # Extract files
                try:
                    print("Extracting downloaded files...")
                    await async_extract_zip(paths)
                except Exception as e:
                    print(f"Failed to extract files: {e}")
                    if attempt < MAX_ATTEMPTS:
                        print(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Failed to extract files after {MAX_ATTEMPTS} attempts: {e}")
                
                # Move files to final location
                try:
                    source_text_path = folder_path / folder_name
                    source_text_path = source_text_path.absolute()
                    print(f"Moving model to {local_path_str}")
                    
                    if source_text_path.exists():
                        # Handle projector path for multimodal models
                        source_projector_path = folder_path / (folder_name + "-projector")
                        source_projector_path = source_projector_path.absolute()
                        
                        if source_projector_path.exists():
                            projector_dest = local_path_str + "-projector"
                            print(f"Moving projector to {projector_dest}")
                            await async_move(str(source_projector_path), projector_dest)
                        
                        # Move model to final location
                        await async_move(str(source_text_path), local_path_str)
                        
                        # Clean up folder after successful move
                        if folder_path.exists():
                            print(f"Cleaning up temporary folder {folder_path}")
                            await async_rmtree(str(folder_path))
                        
                        print(f"Model download complete: {local_path_str}")
                        
                        # Update tracking file to remove this hash
                        with open(tracking_path, "wb") as f:
                            pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
                        
                        return local_path_str
                    else:
                        print(f"Model not found at {source_text_path}")
                        if attempt < MAX_ATTEMPTS:
                            print(f"Retrying in {backoff} seconds")
                            await asyncio.sleep(backoff)
                            continue
                        else:
                            raise Exception(f"Model not found at {source_text_path} after all attempts")
                except Exception as e:
                    print(f"Failed to move model: {e}")
                    if attempt < MAX_ATTEMPTS:
                        print(f"Retrying in {backoff} seconds")
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        raise Exception(f"Failed to move model after {MAX_ATTEMPTS} attempts: {e}")
            
            except aiohttp.ClientError as e:
                print(f"HTTP error on attempt {attempt}: {e}")
                if attempt < MAX_ATTEMPTS:
                    print(f"Retrying in {backoff} seconds")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    raise Exception(f"HTTP error after {MAX_ATTEMPTS} attempts: {e}")
            except Exception as e:
                print(f"Download attempt {attempt} failed: {e}")
                if attempt < MAX_ATTEMPTS:
                    print(f"Retrying in {backoff} seconds")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    raise Exception(f"Download failed after {MAX_ATTEMPTS} attempts: {e}")
    
    except Exception as e:
        print(f"Download failed: {e}")
    finally:
        # Always clean up tracking file
        try:
            with open(tracking_path, "wb") as f:
                pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
        except Exception as e:
            print(f"Error updating tracking file: {e}")
        
        # Clean up any partial downloads
        if folder_path and folder_path.exists():
            try:
                print(f"Cleaning up folder {folder_path} after failure")
                await async_rmtree(str(folder_path))
            except Exception as e:
                print(f"Error cleaning up folder: {e}")
        
        # Remove any partially extracted files if download failed
        for path in extracted_files:
            if path.exists():
                try:
                    path.unlink(missing_ok=True)
                except Exception as e:
                    print(f"Error removing temporary file {path}: {e}")

    print("All download attempts failed")
    return None