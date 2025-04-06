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

manager = LocalLLMManager()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for managing local large language models"
    )
    subparsers = parser.add_subparsers(
        dest='command', help="Commands for managing local language models"  
    )
    start_command = subparsers.add_parser(
        "start", help="Start a local language model server"
    )
    start_command.add_argument(
        "--hash", type=str, required=True,
        help="Filecoin hash of the model to start"
    )
    start_command.add_argument(
        "--port", type=int, default=8080,
        help="Port number for the local language model server"
    )
    start_command.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host address for the local language model server"
    )
    start_command.add_argument(
        "--context-length", type=int, default=32768,
        help="Context length for the local language model server"
    )
    stop_command = subparsers.add_parser(
        "stop", help="Stop a local language model server"
    )
    version_command = subparsers.add_parser(
        "version", help="Print the version of local_llms"
    )
    memory_command = subparsers.add_parser(
        "memory", help="Get memory usage information for the running model"
    )
    memory_command.add_argument(
        "--json", action="store_true", 
        help="Output memory information in JSON format"
    )
    download_command = subparsers.add_parser(
       "download", help="Download and extract model files from IPFS"
    )
    download_command.add_argument(
        "--hash", required=True,
        help="IPFS hash of the model metadata"
    )
    download_command.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Chunk size for downloading files"
    )
    download_command.add_argument(
        "--output-dir", type=Path, default = None,
        help="Output directory for model files"
    )
    upload_command = subparsers.add_parser(
        "upload", help="Upload model files to IPFS"
    )
    upload_command.add_argument(
        "--folder-name", type=str, required=True,
        help="Folder containing model files"
    )
    upload_command.add_argument(
        "--model-family", type=str, required=False,
        help = "Model family (e.g., GPT-3, GPT-4, etc.)"
    )   
    upload_command.add_argument(
        "--zip-chunk-size", type=int, default=512,
        help="Chunk size for splitting compressed files"
    )
    upload_command.add_argument(
        "--threads", type=int, default=16,
        help="Number of threads for compressing files"
    )
    upload_command.add_argument(
        "--max-retries", type=int, default=20,
        help="Maximum number of retries for uploading files"
    )
    upload_command.add_argument(
        "--hf-repo", type=str, default = None,
        help="Hugging Face model repository"
    )
    upload_command.add_argument(
        "--hf-file", type=str, default = None,
        help="Hugging Face model file"
    )
    upload_command.add_argument(
        "--ram", type=float, default=None,
        help="RAM in GB for the serving model at 4096 context length"
    )
    check_command = subparsers.add_parser(
        "check", help="Model metadata check"
    )
    check_command.add_argument(
        "--hash", type=str, required=True,
        help="Model name to check existence"
    )
    status_command = subparsers.add_parser(
       "status", help="Check the running model"
    )
    restart_command = subparsers.add_parser(
        "restart", help="Restart the local language model server"
    )
    check_downloading_command = subparsers.add_parser(
        "downloading", help="Check if the model is being downloaded"
    )
    return parser.parse_known_args()

def version_command():
    logger.info(
        f"Local LLMS (Large Language Model Service) version: {__version__}"
    )

def handle_download(args):
    asyncio.run(download_model_from_filecoin_async(args.hash))

def handle_start(args):
    if not manager.start(args.hash, args.port, args.host, args.context_length):
        sys.exit(1)

def handle_stop(args):
    if not manager.stop():
        sys.exit(1)
    
def handle_check(args):
    is_downloaded = check_downloaded_model(args.hash)
    res = "True" if is_downloaded else "False"
    print(res)
    return res

def handle_status(args):
    running_model = manager.get_running_model()
    if running_model:
        print(running_model)

def handle_upload(args):
    kwargs = {
        "family": args.model_family,
        "ram": args.ram,
        "hf_repo": args.hf_repo,
        "hf_file": args.hf_file,
    }
    upload_folder_to_lighthouse(args.folder_name, args.zip_chunk_size, args.max_retries, args.threads, **kwargs)

def handle_restart(args):
    if not manager.restart():
        sys.exit(1)

def handle_check_downloading(args):
    downloading_files = check_downloading()
    if not downloading_files:
        return False
    str_files = ",".join(downloading_files)
    print(str_files)
    return True

def handle_memory(args):
    import json
    memory_info = manager.get_memory_usage()
    if not memory_info:
        logger.error("No running model or unable to get memory information")
        return False
    
    if "error" in memory_info:
        logger.error(f"Error getting memory information: {memory_info['error']}")
        return False
    
    if args.json:
        print(json.dumps(memory_info, indent=2))
    else:
        model_hash = memory_info.get("model_hash", "Unknown")
        rss_mb = memory_info.get("rss_mb", 0)
        vms_mb = memory_info.get("vms_mb", 0)
        percent = memory_info.get("percent", 0)
        
        # Get system memory information
        sys_info = memory_info.get("system", {})
        total_gb = sys_info.get("total_gb", 0)
        available_gb = sys_info.get("available_gb", 0)
        used_gb = sys_info.get("used_gb", 0)
        
        # Get model information
        model_info = memory_info.get("model_info", {})
        context_length = model_info.get("context_length", 0)
        family = model_info.get("family", "unknown")
        
        # Print summary
        print(f"📊 Memory Usage Report for Model: {model_hash}")
        print(f"  Model Family: {family}")
        print(f"  Context Length: {context_length}")
        print(f"\n📈 Process Memory:")
        print(f"  Total RSS: {rss_mb:.2f} MB")
        print(f"  Total VMS: {vms_mb:.2f} MB")
        
        # Process details
        print(f"\n🔍 Process Details:")
        for proc_name, proc_info in memory_info.get("processes", {}).items():
            if "error" in proc_info:
                print(f"  {proc_name}: Error - {proc_info['error']}")
                continue
                
            rss = proc_info.get("rss_mb", 0)
            vms = proc_info.get("vms_mb", 0)
            cpu = proc_info.get("cpu_percent", 0)
            running_time = proc_info.get("running_time_minutes", 0)
            
            print(f"  {proc_name}:")
            print(f"    RSS: {rss:.2f} MB")
            print(f"    VMS: {vms:.2f} MB")
            print(f"    CPU: {cpu:.1f}%")
            print(f"    Running Time: {running_time:.1f} minutes")
            
            # Children processes if any
            if "children" in proc_info and proc_info["children"]:
                children = proc_info["children"]
                print(f"    Child Processes: {len(children)}")
                child_rss = sum(c.get("rss_mb", 0) for c in children)
                print(f"    Children RSS: {child_rss:.2f} MB")
        
        # System memory
        print(f"\n💻 System Memory:")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Used: {used_gb:.2f} GB ({percent:.1f}%)")
        print(f"  Available: {available_gb:.2f} GB")
    
    return True

def main():
    known_args, unknown_args = parse_args()
    for arg in unknown_args:
        logger.error(f'unknown command or argument: {arg}')
        sys.exit(2)

    if known_args.command == "version":
        version_command()
    elif known_args.command == "start":
        handle_start(known_args)
    elif known_args.command == "stop":
        handle_stop(known_args)
    elif known_args.command == "download":
        handle_download(known_args)
    elif known_args.command == "check":
        handle_check(known_args)
    elif known_args.command == "status":
        handle_status(known_args)
    elif known_args.command == "upload":
        handle_upload(known_args)
    elif known_args.command == "restart":
        handle_restart(known_args)
    elif known_args.command == "downloading":
        handle_check_downloading(known_args)
    elif known_args.command == "memory":
        handle_memory(known_args)
    else:
        logger.error(f"Unknown command: {known_args.command}")
        sys.exit(2)


if __name__ == "__main__":
    main()
