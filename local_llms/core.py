import os
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
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE"))

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

    def start(self, hash: str, port: int = 11434, host: str = "0.0.0.0", context_length: int = 4096) -> bool:
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

            llm_running_port = port + 1

            running_llm_command = [
                llama_server_path,
                "--jinja",
                "--model", str(local_model_path),
                "--port", str(llm_running_port),
                "--host", host,
                "-c", str(context_length),
                "--pooling", "mean",
                "--no-webui"
            ]

            logger.info(f"Starting process: {' '.join(running_llm_command)}")

            log_file_path = f"llm_service_{port}.log"
            try:
                llm_process = subprocess.Popen(
                    running_llm_command,
                    preexec_fn=os.setsid
                )
            except Exception as e:
                logger.error(f"Error starting LLM service: {str(e)}", exc_info=True)
                return False

            if not self._wait_for_service(port):
                logger.error(f"Service failed to start within 600 seconds")
                with open(log_file_path, "r") as f:
                    last_lines = f.readlines()[-10:]
                    logger.error(f"Last 10 lines of log:\n{''.join(last_lines)}")
                llm_process.terminate()
                return False

            app_running_port = port
            uvicorn_path = os.getenv("UVICORN_COMMAND") 
            # start the FastAPI app in the background

            uvicorn_command = [
                uvicorn_path,
                "local_llms.apis:app",
                "--host", host,
                "--port", str(app_running_port),
                "--log-level", "info"
            ]

            logger.info(f"Starting process: {' '.join(uvicorn_command)}")

            try:
                apis_process = subprocess.Popen(
                    uvicorn_command,
                    preexec_fn=os.setsid
                )
            except Exception as e:
                logger.error(f"Error starting FastAPI app: {str(e)}", exc_info=True)
                llm_process.terminate()
                return False
            
            if not self._wait_for_service(app_running_port):
                logger.error(f"API service failed to start within 600 seconds")
                with open("apis.log", "r") as f:
                    last_lines = f.readlines()[-10:]
                    logger.error(f"Last 10 lines of log:\n{''.join(last_lines)}")
                llm_process.terminate()
                apis_process.terminate()
                return False

            logger.info(f"Service started on port {port} for model: {hash}")

            service_metadata = {
                "hash": hash,
                "port": llm_running_port,
                "pid": llm_process.pid,
                "app_pid": apis_process.pid,
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

            # update service metadata to the FastAPI app
            try:
                update_url = f"http://localhost:{app_running_port}/v1/update"
                response = requests.post(update_url, json=service_metadata, timeout=10)
                response.raise_for_status()  # Raise exception for HTTP error responses
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to update service metadata: {str(e)}")
                # Stop the partially started service
                self.stop()
                return False
            
            return True

        except Exception as e:
            logger.error(f"Error starting LLM service: {str(e)}", exc_info=True)
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
            app_pid = service_info.get("app_pid")

            logger.info(f"Stopping LLM service '{hash}' running on port {port} (PID: {pid})...")

            # Terminate process by PID
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=5)  # Allow process to shut down gracefully
                
                if process.is_running():  # Force kill if still alive
                    logger.warning("Process did not terminate, forcing kill.")
                    process.kill()

            # Terminate FastAPI app by PID
            if psutil.pid_exists(app_pid):
                app_process = psutil.Process(app_pid)
                app_process.terminate()
                app_process.wait(timeout=5)  # Allow process to shut down gracefully
                
                if app_process.is_running():  # Force kill if
                    logger.warning("API process did not terminate, forcing kill.")
                    app_process.kill()

            # Remove the tracking file
            os.remove("running_service.pkl")
            logger.info("LLM service stopped successfully.")
            return True

        except Exception as e:
            logger.error(f"Error stopping LLM service: {str(e)}", exc_info=True)
            return False
        