import os
import time
import pickle
import psutil
import asyncio
import requests
import subprocess
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from local_llms.download import download_model_from_filecoin_async

class LocalLLMManager:
    """Manages a local Large Language Model (LLM) service."""
    
    def __init__(self):
        """Initialize the LocalLLMManager."""       
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE"))
        self._process: Optional[subprocess.Popen] = None
        self._service_info: Optional[Dict[str, Any]] = None
        self._last_activity: Optional[datetime] = None
        self._inactivity_threshold = timedelta(hours=1)  # Default 1 hour
        self._cleanup_task: Optional[asyncio.Task] = None
        logger.add("local_llm.log", rotation="500 MB", retention="10 days")

    async def start(self, hash: str, port: int = 11434, host: str = "0.0.0.0", context_length: int = 4096) -> bool:
        """
        Start the local LLM service in the background.

        Args:
            hash (str): Filecoin hash of the model to download and run.
            port (int): Port number for the LLM service (default: 11434).
            host (str): Host address for the LLM service (default: "0.0.0.0").
            context_length (int): Context length for the model (default: 4096).

        Returns:
            bool: True if service started successfully, False otherwise.
        """
        if not hash:
            raise ValueError("Filecoin hash is required to start the service")

        try:
            # Check if service is already running
            if self.get_running_model():
                logger.warning("A service is already running. Use 'stop' to stop it first.")
                return False

            # Check system resources
            if not self._check_resources():
                logger.error("Insufficient system resources to start service")
                return False

            logger.info(f"Starting local LLM service for model with hash: {hash}")
            local_model_path = await download_model_from_filecoin_async(hash)
            
            # Start the service process
            self._process = subprocess.Popen(
                ["ollama", "serve", "--host", host, "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for service to become healthy
            if not self._wait_for_service(port):
                logger.error("Service failed to become healthy")
                self.stop()
                return False

            # Save service information
            self._service_info = {
                "hash": hash,
                "port": port,
                "app_port": port + 1,  # API port is service port + 1
                "context_length": context_length,
                "process_id": self._process.pid,
                "start_time": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            self._last_activity = datetime.now()
            self._dump_running_service(self._service_info)

            # Start cleanup task if not already running
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_inactive_service())

            logger.info(f"Service started successfully on port {port}")
            return True

        except Exception as e:
            logger.error(f"Error starting service: {str(e)}")
            if self._process:
                self.stop()
            return False

    def stop(self) -> bool:
        """
        Stop the currently running LLM service.

        Returns:
            bool: True if service stopped successfully, False otherwise.
        """
        try:
            if not self.pickle_file.exists():
                logger.warning("No running LLM service to stop.")
                return False

            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
                process_id = service_info.get("process_id")

            if process_id:
                try:
                    process = psutil.Process(process_id)
                    process.terminate()
                    process.wait(timeout=30)
                    logger.info(f"Successfully terminated process {process_id}")
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {process_id} already terminated")
                except psutil.TimeoutExpired:
                    logger.warning(f"Process {process_id} did not terminate gracefully, forcing kill")
                    process.kill()

            self.pickle_file.unlink()
            self._service_info = None
            self._last_activity = None
            logger.info("LLM service stopped successfully")
            return True
        except Exception as e:
            logger.error(f"Error stopping LLM service: {str(e)}")
            return False

    def update_activity(self):
        """Update the last activity timestamp of the service."""
        if self._service_info:
            self._last_activity = datetime.now()
            self._service_info["last_activity"] = self._last_activity.isoformat()
            self._dump_running_service(self._service_info)

    async def _cleanup_inactive_service(self):
        """
        Background task to check and stop inactive services.
        """
        while True:
            try:
                if self._service_info and self._last_activity:
                    time_since_last_activity = datetime.now() - self._last_activity
                    if time_since_last_activity > self._inactivity_threshold:
                        logger.info(f"Service inactive for {time_since_last_activity}, stopping...")
                        self.stop()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def _check_resources(self) -> bool:
        """
        Check if system has enough resources to run the LLM service.
        
        Returns:
            bool: True if system has sufficient resources, False otherwise.
        """
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if memory.percent > 90:
                logger.error(f"System memory usage too high: {memory.percent}%")
                return False
                
            if cpu_percent > 90:
                logger.error(f"System CPU usage too high: {cpu_percent}%")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return False

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
        max_retries = 3
        
        while time.time() - start_time < timeout:
            for attempt in range(max_retries):
                try:
                    status = requests.get(health_check_url, timeout=5)
                    if status.status_code == 200 and status.json().get("status") == "ok":
                        logger.info(f"Service healthy at {health_check_url}")
                        return True
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Health check attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff between retries
                    continue
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
        logger.error(f"Service failed to become healthy within {timeout} seconds")
        return False

    def _dump_running_service(self, metadata: dict):
        """Save service information to pickle file."""
        try:
            with open(self.pickle_file, "wb") as f:
                pickle.dump(metadata, f)
            logger.info("Service information saved successfully")
        except Exception as e:
            logger.error(f"Error saving service information: {str(e)}")
            raise

    def get_running_model(self) -> Optional[str]:
        """Get information about the currently running model."""
        try:
            if not self.pickle_file.exists():
                return None
                
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
                self._service_info = service_info
                self._last_activity = datetime.fromisoformat(service_info.get("last_activity", datetime.now().isoformat()))
                return service_info.get("hash")
        except Exception as e:
            logger.error(f"Error reading service information: {str(e)}")
            return None

    def restart(self) -> bool:
        """
        Restart the currently running LLM service.

        Returns:
            bool: True if service restarted successfully, False otherwise.
        """
        if not self.pickle_file.exists():
            logger.warning("No running LLM service to restart.")
            return False
        
        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash = service_info.get("hash")
            port = service_info.get("app_port")
            context_length = service_info.get("context_length")

            logger.info(f"Restarting LLM service '{hash}' running on port {port}...")

            # Stop the current service
            self.stop()

            # Start the service with the same parameters
            return asyncio.run(self.start(hash, port, context_length=context_length))
        except Exception as e:
            logger.error(f"Error restarting LLM service: {str(e)}")
            return False