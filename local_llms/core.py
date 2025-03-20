import os
import shutil
import time
import pickle
import psutil
import requests
import subprocess
from pathlib import Path
from loguru import logger
from typing import Optional
from local_llms.download import download_model_from_filecoin

class LocalLLMManager:
    """Manages a local Large Language Model (LLM) service."""
    
    def __init__(self):
        """Initialize the LocalLLMManager."""       
        self.pickle_file = Path.cwd()/ "running_service.pkl"

    def start(self, hash: str, port: int = 8080, host: str = "0.0.0.0", context_length: int = 4096) -> bool:
        """
        Start the local LLM service in the background.
        
        Args:
            hash (str, optional): Filecoin hash of the model to download and run
            port (int): Port number for the LLM service (default: 8080)
            
        Returns:
            bool: True if service started successfully, False otherwise
            
        Raises:
            ValueError: If hash is not provided when no model is running
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")
        
        try:
            logger.info(f"Starting local LLM service for model with hash: {hash}")
            local_model_path = download_model_from_filecoin(hash)
            model_running = self.get_running_model()
            if model_running:
                if model_running == hash:
                    logger.warning(f"Model '{hash}' is already running on port {port}")
                    return True
                else:
                    logger.info(f"Stopping existing model '{model_running}' running on port {port}")
                    self.stop()

            
            if not os.path.exists(local_model_path):
                logger.error(f"Model file not found at: {local_model_path}")
                return False
                
            logger.info(f"Local LLM service starting for model: {local_model_path}")
            llama_server_path = os.getenv("LLAMA_SERVER_PATH")
            if not llama_server_path:
                logger.error("llama-server executable not found in PATH or LLAMA_SERVER_PATH environment variable.")

            # Run llama-server in the background with additional safety checks
            # Removed unnecessary quotes, 'nohup' and '&' since subprocess.Popen handles background execution.
            command = [
                llama_server_path,
                "--jinja",
                "--model", str(local_model_path),
                "--port", str(port),
                "--host", host,
                "-c", str(context_length),
                "--pooling", "cls"
            ]
            logger.info(f"Starting process with command: {' '.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,  # Suppress output (or redirect to a file)
                stderr=subprocess.DEVNULL,  # Suppress errors (or redirect to a file)
                preexec_fn=os.setsid       # Detach process into a new session (Unix-like systems only)
            )
            health_check_url = f"http://localhost:{port}/health"
            # 20 minutes timeout for starting the service
            maximum_start_time = 1200  # 20 minutes
            start_time = time.time()
            while True:
                if time.time() - start_time > maximum_start_time:
                    logger.error(f"Failed to start local LLM service within {maximum_start_time} seconds.")
                    # Capture any output from the process for diagnosis
                    stdout, stderr = process.communicate(timeout=1)
                    logger.error(f"Process stdout: {stdout.decode() if stdout else 'None'}")
                    logger.error(f"Process stderr: {stderr.decode() if stderr else 'None'}")
                    return False
                try:
                    logger.debug(f"Attempting health check at {health_check_url}")
                    status = requests.get(health_check_url, timeout=5)
                    logger.debug(f"Health check response: {status.status_code}")
                    if status.status_code == 200:
                        status_json = status.json()
                        logger.debug(f"Health check JSON: {status_json}")
                        if status_json.get("status") == "ok":
                            break
                except requests.exceptions.ConnectionError as e:
                    logger.debug(f"Failed to connect to the service: {str(e)}")
                    # Check if process is still running
                    if not psutil.pid_exists(process.pid):
                        logger.error(f"Process with PID {process.pid} died unexpectedly")
                        return False
                except Exception as e:
                    logger.debug(f"Health check error: {str(e)}")
                time.sleep(1)
            self._dump_running_service(hash, port, process.pid)
            logger.info(f"Local LLM service started successfully on port {port} "
                       f"for model: {hash}")
            return True
            
        except FileNotFoundError:
            logger.error("llama-server executable not found in system PATH")
            return False
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start local LLM service: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting LLM service: {str(e)}", exc_info=True)
            return False
        
    def _dump_running_service(self, hash, port, pid):

        """Dump the running service details to a file."""
        service_info = {"hash": hash, "port": port, "pid": pid}
        with open("running_service.pkl", "wb") as f:
            pickle.dump(service_info, f)

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.pickle_file.exists():
            return None

        try:
            with self.pickle_file.open("rb") as f:
                service_info = pickle.load(f)
            service_port = service_info.get("port")
            response = requests.get(f"http://localhost:{service_port}/health", timeout=2)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return service_info.get("hash")
        except (requests.exceptions.RequestException, OSError, pickle.UnpicklingError):
            pass

        # Clean up if the health check fails or an error occurs
        if self.pickle_file.exists():
            self.pickle_file.unlink()
        return None

    def stop(self) -> bool:
        """
        Stop the running LLM service.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists("running_service.pkl"):
            logger.warning("No running LLM service to stop.")
            return False

        try:
            # Load service details from the pickle file
            with open("running_service.pkl", "rb") as f:
                service_info = pickle.load(f)
            
            port = service_info.get("port")
            hash = service_info.get("hash")
            pid = service_info.get("pid")

            logger.info(f"Stopping LLM service '{hash}' running on port {port} (PID: {pid})...")

            # Terminate process by PID
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)  # Allow process to shut down gracefully
                
                if process.is_running():  # Force kill if still alive
                    logger.warning("Process did not terminate, forcing kill.")
                    process.kill()

            # Remove the tracking file
            os.remove("running_service.pkl")
            logger.info("LLM service stopped successfully.")
            return True

        except Exception as e:
            logger.error(f"Error stopping LLM service: {str(e)}", exc_info=True)
            return False
        