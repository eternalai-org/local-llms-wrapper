import sys
import asyncio
import argparse
from pathlib import Path
from loguru import logger
from local_llms import __version__
from local_llms.core import LocalLLMManager
from local_llms.upload import upload_folder_to_lighthouse
from local_llms.utils import check_downloading
from local_llms.download import check_downloaded_model, download_model_from_filecoin_async

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG" if verbose else "INFO"
    )
    if verbose:
        logger.add(
            "local_llm.log",
            rotation="500 MB",
            retention="10 days",
            level="DEBUG"
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tool for managing local large language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"local_llms {__version__}"
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help="Commands for managing local language models"
    )
    
    # Start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start a local language model server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    start_parser.add_argument(
        "--hash",
        required=True,
        help="Filecoin hash of the model to start"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Port number for the local language model server"
    )
    start_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address for the local language model server"
    )
    start_parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length for the local language model server"
    )
    start_parser.add_argument(
        "--inactivity-threshold",
        type=int,
        default=3600,
        help="Time in seconds before stopping inactive service"
    )
    
    # Stop command
    subparsers.add_parser(
        "stop",
        help="Stop the currently running LLM service"
    )
    
    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download and extract model files from IPFS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    download_parser.add_argument(
        "--hash",
        required=True,
        help="IPFS hash of the model metadata"
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for model files"
    )
    download_parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size for downloading files"
    )
    
    # Upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload model files to IPFS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    upload_parser.add_argument(
        "--folder-name",
        required=True,
        help="Folder containing model files"
    )
    upload_parser.add_argument(
        "--model-family",
        help="Model family (e.g., GPT-3, GPT-4, etc.)"
    )
    upload_parser.add_argument(
        "--hf-repo",
        help="Hugging Face model repository"
    )
    upload_parser.add_argument(
        "--hf-file",
        help="Hugging Face model file"
    )
    upload_parser.add_argument(
        "--ram",
        type=float,
        help="RAM in GB for the serving model at 4096 context length"
    )
    upload_parser.add_argument(
        "--zip-chunk-size",
        type=int,
        default=512,
        help="Chunk size for splitting compressed files"
    )
    upload_parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads for compressing files"
    )
    upload_parser.add_argument(
        "--max-retries",
        type=int,
        default=20,
        help="Maximum number of retries for uploading files"
    )
    
    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Model metadata check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    check_parser.add_argument(
        "--hash",
        required=True,
        help="Model name to check existence"
    )
    
    # Status command
    subparsers.add_parser(
        "status",
        help="Check the running model"
    )
    
    # Restart command
    subparsers.add_parser(
        "restart",
        help="Restart the local language model server"
    )
    
    # Check downloading command
    subparsers.add_parser(
        "downloading",
        help="Check if the model is being downloaded"
    )
    
    return parser.parse_args()

async def handle_start(args):
    """Handle start command."""
    try:
        manager = LocalLLMManager()
        if args.inactivity_threshold:
            manager._inactivity_threshold = args.inactivity_threshold
            
        success = await manager.start(
            args.hash,
            args.port,
            args.host,
            args.context_length
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting service: {str(e)}")
        sys.exit(1)

async def handle_stop():
    """Handle stop command."""
    try:
        manager = LocalLLMManager()
        if not manager.stop():
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error stopping service: {str(e)}")
        sys.exit(1)

async def handle_download(args):
    """Handle download command."""
    try:
        await download_model_from_filecoin_async(args.hash, args.output_dir)
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        sys.exit(1)

def handle_check(args):
    """Handle check command."""
    try:
        is_downloaded = check_downloaded_model(args.hash)
        print("True" if is_downloaded else "False")
        return is_downloaded
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")
        sys.exit(1)

def handle_status():
    """Handle status command."""
    try:
        manager = LocalLLMManager()
        running_model = manager.get_running_model()
        if running_model:
            print(running_model)
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        sys.exit(1)

async def handle_restart():
    """Handle restart command."""
    try:
        manager = LocalLLMManager()
        if not manager.restart():
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error restarting service: {str(e)}")
        sys.exit(1)

def handle_check_downloading():
    """Handle check downloading command."""
    try:
        downloading_files = check_downloading()
        if downloading_files:
            print(",".join(downloading_files))
            return True
        return False
    except Exception as e:
        logger.error(f"Error checking downloads: {str(e)}")
        sys.exit(1)

def handle_upload(args):
    """Handle upload command."""
    try:
        kwargs = {
            "family": args.model_family,
            "ram": args.ram,
            "hf_repo": args.hf_repo,
            "hf_file": args.hf_file,
        }
        upload_folder_to_lighthouse(
            args.folder_name,
            args.zip_chunk_size,
            args.max_retries,
            args.threads,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        sys.exit(1)

async def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    if not args.command:
        logger.error("No command specified")
        sys.exit(2)
        
    try:
        if args.command == "start":
            await handle_start(args)
        elif args.command == "stop":
            await handle_stop()
        elif args.command == "download":
            await handle_download(args)
        elif args.command == "check":
            handle_check(args)
        elif args.command == "status":
            handle_status()
        elif args.command == "upload":
            handle_upload(args)
        elif args.command == "restart":
            await handle_restart()
        elif args.command == "downloading":
            handle_check_downloading()
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
