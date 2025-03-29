import os
import shutil
import time
import httpx
import requests
import aiohttp
import asyncio
import pickle
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from local_llms.utils import compute_file_hash, async_extract_zip, async_move, async_rmtree

# Configure logging
logger = logging.getLogger(__name__)

# Constants
GATEWAY_URL = "https://gateway.lighthouse.storage/ipfs/"
DEFAULT_OUTPUT_DIR = Path.cwd() / "llms-storage"
SLEEP_TIME = 60
MAX_ATTEMPTS = 10
CHUNK_SIZE = 8192
POSTFIX_MODEL_PATH = ".gguf"
MAX_CONCURRENT_DOWNLOADS = 5
MAX_RETRY_BACKOFF = 300
HASH_VERIFICATION_THREADS = 4

class DownloadManager:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session: Optional[aiohttp.ClientSession] = None
        self._executor = ThreadPoolExecutor(max_workers=HASH_VERIFICATION_THREADS)

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        self._executor.shutdown(wait=True)

    async def download_with_progress(self, url: str, file_path: Path, desc: str) -> bool:
        """Download a file with progress bar and proper error handling."""
        try:
            async with self.semaphore:  # Limit concurrent downloads
                async with self.session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download {url}. Status: {response.status}")
                        return False

                    total_size = int(response.headers.get("content-length", 0))
                    with file_path.open("wb") as f, tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=desc,
                        ncols=80
                    ) as progress:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                            progress.update(len(chunk))
                    return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}", exc_info=True)
            return False

    async def verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file hash using a thread pool to avoid blocking."""
        try:
            computed_hash = await asyncio.get_event_loop().run_in_executor(
                self._executor, compute_file_hash, file_path
            )
            return computed_hash == expected_hash
        except Exception as e:
            logger.error(f"Error verifying hash for {file_path}: {e}", exc_info=True)
            return False

async def download_single_file_async(
    manager: DownloadManager,
    file_info: Dict[str, Any],
    folder_path: Path,
    max_attempts: int = MAX_ATTEMPTS
) -> Tuple[Optional[Path], Optional[str]]:
    """Asynchronously download a single file with improved error handling and retries."""
    cid = file_info["cid"]
    expected_hash = file_info["file_hash"]
    file_name = file_info["file_name"]
    file_path = folder_path / file_name

    try:
        if file_path.exists():
            if await manager.verify_hash(file_path, expected_hash):
                logger.info(f"File {cid} already exists with correct hash.")
                return file_path, None
            file_path.unlink()

        for attempt in range(max_attempts):
            try:
                url = f"{GATEWAY_URL}{cid}"
                if await manager.download_with_progress(url, file_path, f"Downloading {file_name}"):
                    if await manager.verify_hash(file_path, expected_hash):
                        logger.info(f"File {cid} downloaded and verified successfully.")
                        return file_path, None
                    file_path.unlink()
                    logger.warning(f"Hash mismatch for {cid}. Expected {expected_hash}")

            except Exception as e:
                logger.error(f"Exception downloading {cid}: {e}", exc_info=True)

            if attempt < max_attempts - 1:
                backoff = min(SLEEP_TIME * (2 ** attempt), MAX_RETRY_BACKOFF)
                logger.info(f"Retrying in {backoff}s (Attempt {attempt + 2}/{max_attempts})")
                await asyncio.sleep(backoff)

        return None, f"Failed to download {cid} after {max_attempts} attempts."
    except Exception as e:
        logger.error(f"Critical error downloading {cid}: {e}", exc_info=True)
        return None, str(e)

async def download_files_from_lighthouse_async(data: Dict[str, Any]) -> List[Path]:
    """Asynchronously download files with improved concurrency control."""
    folder_name = data["folder_name"]
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    num_of_files = data["num_of_files"]
    filecoin_hash = data["filecoin_hash"]

    async with DownloadManager() as manager:
        tasks = [
            download_single_file_async(manager, file_info, folder_path)
            for file_info in data["files"]
        ]
        
        successful_downloads = []
        for future in asyncio.as_completed(tasks):
            path, error = await future
            if path:
                successful_downloads.append(path)
                print(f"[LAUNCHER_LOGGER] [MODEL_INSTALL] --step {len(successful_downloads)}-{num_of_files} --hash {filecoin_hash}")
            else:
                logger.error(f"Download failed: {error}")

        if len(successful_downloads) == num_of_files:
            logger.info(f"All {num_of_files} files downloaded successfully.")
            return successful_downloads
        else:
            logger.warning(f"Downloaded {len(successful_downloads)} out of {num_of_files} files.")
            return []

def check_downloaded_model(filecoin_hash: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    """Check if the model is already downloaded and verified."""
    try:
        local_path = output_dir / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
        local_path = local_path.absolute()
        return local_path.exists()
    except Exception as e:
        logger.error(f"Error checking downloaded model: {e}", exc_info=True)
        return False

async def download_model_from_filecoin_async(
    filecoin_hash: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_retries: int = MAX_ATTEMPTS
) -> Optional[str]:
    """
    Asynchronously download a model from Filecoin with improved error handling and resource management.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    local_path = output_dir / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
    local_path_str = str(local_path.absolute())

    if check_downloaded_model(filecoin_hash, output_dir):
        logger.info(f"Using existing model at {local_path_str}")
        return local_path_str

    tracking_path = os.environ["TRACKING_DOWNLOAD_HASHES"]
    downloading_files = []
    if os.path.exists(tracking_path):
        with open(tracking_path, "rb") as f:
            downloading_files = pickle.load(f)

    if filecoin_hash not in downloading_files:
        downloading_files.append(filecoin_hash)
        with open(tracking_path, "wb") as f:
            pickle.dump(downloading_files, f)

    input_link = f"{GATEWAY_URL}{filecoin_hash}"
    temp_folder = None

    try:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading model metadata (attempt {attempt}/{max_retries})")
                async with aiohttp.ClientSession() as session:
                    async with session.get(input_link) as response:
                        response.raise_for_status()
                        data = await response.json()
                        data["filecoin_hash"] = filecoin_hash
                        
                        temp_folder = Path.cwd() / f"temp_{filecoin_hash}_{attempt}"
                        temp_folder.mkdir(exist_ok=True, parents=True)
                        
                        paths = await download_files_from_lighthouse_async(data)
                        if not paths:
                            logger.error("Failed to download model files")
                            continue

                        try:
                            await async_extract_zip(paths)
                        except Exception as e:
                            logger.error(f"Failed to extract files: {e}", exc_info=True)
                            continue

                        source_text_path = temp_folder / data["folder_name"]
                        source_text_path = source_text_path.absolute()
                        
                        if not source_text_path.exists():
                            logger.error(f"Model not found at {source_text_path}")
                            continue

                        logger.info(f"Moving model to {local_path_str}")
                        source_projector_path = temp_folder / (data["folder_name"] + "-projector")
                        
                        if source_projector_path.exists():
                            await async_move(str(source_projector_path), local_path_str + "-projector")
                        
                        await async_move(str(source_text_path), local_path_str)
                        
                        if temp_folder.exists():
                            await async_rmtree(str(temp_folder))
                        
                        logger.info(f"Model download complete: {local_path_str}")
                        
                        with open(tracking_path, "wb") as f:
                            pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
                        
                        return local_path_str

            except Exception as e:
                logger.error(f"Download attempt {attempt} failed: {e}", exc_info=True)
                if attempt < max_retries:
                    backoff = min(SLEEP_TIME * (2 ** (attempt - 1)), MAX_RETRY_BACKOFF)
                    logger.info(f"Retrying in {backoff} seconds")
                    await asyncio.sleep(backoff)
                continue

    except Exception as e:
        logger.error(f"Critical error during download: {e}", exc_info=True)
    finally:
        if temp_folder and temp_folder.exists():
            try:
                await async_rmtree(str(temp_folder))
            except Exception as e:
                logger.error(f"Failed to clean up temporary folder: {e}", exc_info=True)
        
        try:
            with open(tracking_path, "wb") as f:
                pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
        except Exception as e:
            logger.error(f"Failed to update tracking file: {e}", exc_info=True)

    logger.error("All download attempts failed")
    return None