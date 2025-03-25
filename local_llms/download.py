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

    if file_path.exists():
        computed_hash = compute_file_hash(file_path)
        if computed_hash == expected_hash:
            print(f"File {cid} already exists with correct hash.")
            return file_path, None
        else:
            print(f"File {cid} exists but hash mismatch. Retrying...")
            file_path.unlink()

    while attempts < max_attempts:
        try:
            url = f"{GATEWAY_URL}{cid}"
            async with session.get(url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get("content-length", 0))
                    with file_path.open("wb") as f, tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {file_name}",
                        ncols=80
                    ) as progress:
                        async for chunk in response.content.iter_chunked(4096):
                            f.write(chunk)
                            progress.update(len(chunk))

                    computed_hash = compute_file_hash(file_path)
                    if computed_hash == expected_hash:
                        print(f"File {cid} downloaded and verified successfully.")
                        return file_path, None
                    else:
                        print(f"Hash mismatch for {cid}. Expected {expected_hash}, got {computed_hash}.")
                        file_path.unlink()
                else:
                    print(f"Failed to download {cid}. Status: {response.status}")

        except Exception as e:
            print(f"Exception downloading {cid}: {e}")

        attempts += 1
        if attempts < max_attempts:
            print(f"Retrying in {SLEEP_TIME}s (Attempt {attempts + 1}/{max_attempts})")
            await asyncio.sleep(SLEEP_TIME)
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

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_single_file_async(session, file_info, folder_path)
            for file_info in data["files"]
        ]
        successful_downloads = []
        for future in asyncio.as_completed(tasks):
            path, error = await future
            if path:
                successful_downloads.append(path)
                print(f"[LAUNCHER_LOGGER] [MODEL_INSTALL] --step {len(successful_downloads)}-{num_of_files} --hash {filecoin_hash}")
            else:
                print(f"Download failed: {error}")

        if len(successful_downloads) == num_of_files:
            print(f"All {num_of_files} files downloaded successfully.")
            return successful_downloads
        else:
            print(f"Downloaded {len(successful_downloads)} out of {num_of_files} files.")
            return []

async def download_model_from_filecoin_async(filecoin_hash: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> str | None:
    """
    Asynchronously download a model from Filecoin using its IPFS hash.

    Args:
        filecoin_hash (str): IPFS hash of the model metadata.
        output_dir (Path): Directory to save the downloaded model.

    Returns:
        str | None: Path to the downloaded model if successful, None otherwise.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    local_path = output_dir / f"{filecoin_hash}{POSTFIX_MODEL_PATH}"
    local_path_str = str(local_path.absolute())

    if check_downloaded_model(filecoin_hash, output_dir):
        print(f"Using existing model at {local_path_str}")
        return local_path_str

    tracking_path = os.environ["TRACKING_DOWNLOAD_HASHES"]
    downloading_files = []
    if os.path.exists(tracking_path):
        with open(tracking_path, "rb") as f:
            downloading_files = pickle.load(f)

    downloading_files.append(filecoin_hash)
    with open(tracking_path, "wb") as f:
        pickle.dump(downloading_files, f)

    input_link = f"{GATEWAY_URL}{filecoin_hash}"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"Downloading model metadata (attempt {attempt}/{MAX_ATTEMPTS})")
            async with aiohttp.ClientSession() as session:
                async with session.get(input_link) as response:
                    response.raise_for_status()
                    data = await response.json()
                    data["filecoin_hash"] = filecoin_hash
                    folder_name = data["folder_name"]
                    folder_path = Path.cwd() / folder_name
                    folder_path.mkdir(exist_ok=True, parents=True)

                    paths = await download_files_from_lighthouse_async(data)
                    if not paths:
                        print("Failed to download model files")
                        continue

                    try:
                        await async_extract_zip(paths)
                    except Exception as e:
                        print(f"Failed to extract files: {e}")
                        continue

                    try:
                        source_text_path = folder_path / folder_name
                        source_text_path = source_text_path.absolute()
                        print(f"Moving model to {local_path_str}")
                        if source_text_path.exists():
                            source_projector_path = folder_path / (folder_name + "-projector")
                            source_projector_path = source_projector_path.absolute()
                            if source_projector_path.exists():
                                await async_move(str(source_projector_path), local_path_str + "-projector")
                            await async_move(str(source_text_path), local_path_str)
                        else:
                            print(f"Model not found at {source_text_path}")
                            continue

                        if folder_path.exists():
                            await async_rmtree(str(folder_path))
                        print(f"Model download complete: {local_path_str}")
                        with open(tracking_path, "wb") as f:
                            pickle.dump([f for f in downloading_files if f != filecoin_hash], f)
                        return local_path_str
                    except Exception as e:
                        print(f"Failed to move model: {e}")
                        continue

        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                backoff = min(SLEEP_TIME * (2 ** (attempt - 1)), 300)
                print(f"Retrying in {backoff} seconds")
                await asyncio.sleep(backoff)

    with open(tracking_path, "wb") as f:
        pickle.dump([f for f in downloading_files if f != filecoin_hash], f)

    print("All download attempts failed")
    return None