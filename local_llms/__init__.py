import os
import shutil
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("local_llms.log")
    ]
)
logger = logging.getLogger(__name__)

"""Local LLMs - A library to manage local language models."""
__version__ = "2.0.1"

# Define command directories with priority order
COMMAND_DIRS = [
    "/usr/local/bin",
    os.path.join(os.path.expanduser("~"), "homebrew", "bin"),
    "/opt/homebrew/bin",
    "/usr/bin",
    "/bin"
]

# Cache for command paths
_command_cache: Dict[str, str] = {}

@lru_cache(maxsize=128)
def find_command(cmd_name: str, search_path: str) -> Optional[str]:
    """
    Find a command in the search path with caching.
    
    Args:
        cmd_name: Name of the command to find
        search_path: Path string to search for the command
        
    Returns:
        Path to the command if found, None otherwise
    """
    return shutil.which(cmd_name, path=search_path)

def find_and_set_command(cmd_name: str, env_var_name: str, search_path: str) -> str:
    """
    Find a command in the search path, set its environment variable, and return its path.
    Uses caching to improve performance.
    
    Args:
        cmd_name: Name of the command to find
        env_var_name: Environment variable name to set
        search_path: Path string to search for the command
        
    Returns:
        Path to the command if found
        
    Raises:
        RuntimeError: If the command is not found
    """
    try:
        # Check cache first
        if cmd_name in _command_cache:
            cmd_path = _command_cache[cmd_name]
            os.environ[env_var_name] = cmd_path
            return cmd_path

        # Find command and cache result
        cmd_path = find_command(cmd_name, search_path)
        if not cmd_path:
            raise RuntimeError(f"{cmd_name} command not found in command directories or PATH")
        
        _command_cache[cmd_name] = cmd_path
        os.environ[env_var_name] = cmd_path
        return cmd_path

    except Exception as e:
        logger.error(f"Failed to find {cmd_name}: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to find {cmd_name}: {str(e)}")

def initialize_environment() -> None:
    """Initialize the environment with required commands and paths."""
    try:
        # Get the current PATH and create search path
        current_path = os.environ.get("PATH", "")
        search_path = os.pathsep.join(COMMAND_DIRS + [current_path])

        # Define required commands and their environment variables
        required_commands = [
            ("llama-server", "LLAMA_SERVER"),
            ("tar", "TAR_COMMAND"),
            ("pigz", "PIGZ_COMMAND"),
            ("cat", "CAT_COMMAND"),
            ("llama-gemma3-cli", "gemma3"),
        ]

        # Find and set all required commands
        for cmd_name, env_var_name in required_commands:
            find_and_set_command(cmd_name, env_var_name, search_path)

        # Set up paths for service and download tracking
        base_dir = Path.cwd()
        running_service_path = (base_dir / "running_service.pkl").absolute()
        tracking_download_hashes = (base_dir / "download_hashes.pkl").absolute()

        # Ensure tracking file exists with empty list
        if not tracking_download_hashes.exists():
            with open(tracking_download_hashes, "wb") as f:
                pickle.dump([], f)

        # Set environment variables
        os.environ["RUNNING_SERVICE_FILE"] = str(running_service_path)
        os.environ["TRACKING_DOWNLOAD_HASHES"] = str(tracking_download_hashes)

        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to initialize environment: {str(e)}")

# Initialize environment when module is imported
initialize_environment()  