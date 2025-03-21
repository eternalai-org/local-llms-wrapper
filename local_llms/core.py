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
        self.pickle_file = os.getenv("RUNNING_SERVICE_FILE")

    def _wait_for_service(self, port: int, timeout: int = 600) -> bool:
        """
        Wait for the LLM service to become healthy.

        Args:
            port (int): Port number of the service.
            timeout (int): Maximum time to wait in seconds (default: 600).

        Returns:
            bool: True if service is healthy, False otherwise.
        """
        health_check_url = f"http://localhost:{port}/health"
        start_time = time.time()
        wait_time = 1  # Initial wait time in seconds
        while time.time() - start_time < timeout:
            try:
                status = requests.get(health_check_url, timeout=5)
                if status.status_code == 200 and status.json().get("status") == "ok":
                    logger.debug(f"Service healthy at {health_check_url}")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed: {str(e)}")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
        return False

     def start(self, hash: str, port: int = 8080, host: str = "0.0.0.0", context_length: int = 4096) -> bool:
        """
        Start the local LLM service in the background.

        Args:
            hash (str): Filecoin hash of the model to download and run.
            port (int): Port number for the LLM service (default: 8080).
            host (str): Host address for the LLM service (default: "0.0.0.0").
            context_length (int): Context length for the model (default: 4096).

        Returns:
            bool: True if service started successfully, False otherwise.

        Raises:
            ValueError: If hash is not provided when no model is running.
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")

        try:
            logger.info(f"Starting local LLM service for model with hash: {hash}")
            local_model_path = download_model_from_filecoin(hash)
            model_running = self.get_running_model()
            if model_running:
                if model_running == hash:
                    logger.warning(f"Model '{hash}' already running on port {port}")
                    return True
                logger.info(f"Stopping existing model '{model_running}' on port {port}")
                self.stop()

            if not os.path.exists(local_model_path):
                logger.error(f"Model file not found at: {local_model_path}")
                return False

            llama_server_path = os.getenv("LLAMA_SERVER")
            if not llama_server_path or not os.path.exists(llama_server_path):
                logger.error("llama-server executable not found in LLAMA_SERVER or PATH")
                return False

            command = [
                llama_server_path,
                "--jinja",
                "--model", local_model_path,
                "--port", str(port),
                "--host", host,
                "-c", str(context_length),
                "--pooling", "cls"
            ]
            logger.info(f"Starting process: {' '.join(command)}")

            log_file_path = f"llm_service_{port}.log"
            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=log_file,
                    preexec_fn=os.setsid
                )

            if not self._wait_for_service(port):
                logger.error(f"Service failed to start within 1200 seconds")
                with open(log_file_path, "r") as f:
                    last_lines = f.readlines()[-10:]
                    logger.error(f"Last 10 lines of log:\n{''.join(last_lines)}")
                process.terminate()
                return False

            logger.info(f"Service started on port {port} for model: {hash}")

            service_metadata = {
                "hash": hash,
                "port": port,
                "pid": process.pid,
                "multimodal": False,
                "local_text_path": local_model_path
            }
            projector_path = f"{local_model_path}-projector"
            if os.path.exists(projector_path):
                service_metadata["multimodal"] = True
                service_metadata["local_projector_path"] = projector_path
                filecoin_url = f"https://gateway.lighthouse.storage/ipfs/{hash}"
                for attempt in range(3):
                    try:
                        response = requests.get(filecoin_url, timeout=10)
                        if response.status_code == 200:
                            service_metadata["family"] = response.json().get("family")
                            break
                    except requests.exceptions.RequestException:
                        time.sleep(2)  # Delay between retries

            self._dump_running_service(service_metadata)
            return True

        except Exception as e:
            logger.error(f"Error starting LLM service: {str(e)}", exc_info=True)
            if "process" in locals():
                process.terminate()
            return False
        
    def _dump_running_service(self, metadata: dict):

        """Dump the running service details to a file."""
        with open("running_service.pkl", "wb") as f:
            pickle.dump(metadata, f)

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
        