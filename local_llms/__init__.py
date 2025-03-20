import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""Local LLMs - A library to manage local language models."""
__version__ = "2.22.0"

COMMAND_DIRS = [
    "/usr/local/bin",
    os.path.join(os.path.expanduser("~"), "homebrew", "bin"),
    "/opt/homebrew/bin",
    "/usr/bin",
    "/bin"
]

# Get the current PATH
current_path = os.environ.get("PATH", "")

# Create a search path with COMMAND_DIRS followed by the system's PATH
search_path = os.pathsep.join(COMMAND_DIRS + [current_path])

# Find commands with logging and error handling
try:
    llama_server_path = shutil.which("llama-server", path=search_path)
    if not llama_server_path:
        logger.error("llama-server binary not found in command directories or PATH")
        raise RuntimeError("llama-server binary not found in command directories or PATH.")
except Exception as e:
    logger.error(f"Failed to find llama-server: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to find llama-server: {str(e)}")

try:
    tar_cmd = shutil.which("tar", path=search_path)
    if not tar_cmd:
        logger.error("tar command not found in command directories or PATH")
        raise RuntimeError("tar command not found in command directories or PATH.")
except Exception as e:
    logger.error(f"Failed to find tar: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to find tar: {str(e)}")

try:
    pigz_cmd = shutil.which("pigz", path=search_path)
    if not pigz_cmd:
        logger.error("pigz command not found in command directories or PATH")
        raise RuntimeError("pigz command not found in command directories or PATH.")
except Exception as e:
    logger.error(f"Failed to find pigz: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to find pigz: {str(e)}")

# New section: search for cat command
try:
    cat_cmd = shutil.which("cat", path=search_path)
    if not cat_cmd:
        logger.error("cat command not found in command directories or PATH")
        raise RuntimeError("cat command not found in command directories or PATH.")
except Exception as e:
    logger.error(f"Failed to find cat: {str(e)}", exc_info=True)
    raise RuntimeError(f"Failed to find cat: {str(e)}")

# Export the found paths to the environment
os.environ["LLAMA_SERVER_PATH"] = llama_server_path
os.environ["TAR_COMMAND"] = tar_cmd
os.environ["PIGZ_COMMAND"] = pigz_cmd
os.environ["CAT_COMMAND"] = cat_cmd