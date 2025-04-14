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
import gc
import signal
import pkg_resources
from local_llms.download import download_model_from_filecoin_async

class LocalLLMManager:
    """Manages a local Large Language Model (LLM) service."""
    
    def __init__(self):
        """Initialize the LocalLLMManager."""       
        self.pickle_file = Path(os.getenv("RUNNING_SERVICE_FILE", "running_service.pkl"))
        self.loaded_models: Dict[str, Any] = {}
        self.idle_timeout = int(os.getenv("LLM_IDLE_TIMEOUT", "1800"))  # Default 30 minutes
        self.last_activity = time.time()
        self._initialize_activity_tracker()

    def _initialize_activity_tracker(self):
        """Initialize a background activity tracker."""
        signal.signal(signal.SIGUSR1, self._handle_activity_signal)
        
        # Start idle checker in a separate thread if needed
        if self.idle_timeout > 0:
            import threading
            self.activity_thread = threading.Thread(target=self._check_idle_status, daemon=True)
            self.activity_thread.start()
    
    def _handle_activity_signal(self, signum, frame):
        """Handle activity signal to update last activity time."""
        self.last_activity = time.time()
        logger.debug("Activity tracker updated")
    
    def track_activity(self):
        """Mark the LLM as active."""
        self.last_activity = time.time()
    
    def _check_idle_status(self):
        """Background thread to check for idle status."""
        while True:
            time.sleep(60)  # Check every minute
            
            if not self.pickle_file.exists():
                continue
                
            current_time = time.time()
            idle_time = current_time - self.last_activity
            
            if idle_time > self.idle_timeout:
                logger.info(f"Model has been idle for {idle_time:.1f} seconds, unloading...")
                try:
                    self.unload_model()
                except Exception as e:
                    logger.error(f"Error unloading idle model: {str(e)}")

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
                if status.status_code == 200 and (status.json().get("status") == "ok" or status.json().get("status") == "starting"):
                    logger.debug(f"Service healthy at {health_check_url}")
                    return True
            except requests.exceptions.RequestException as e:
                logger.debug(f"Health check failed: {str(e)}")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max 60s
        return False
    
    def restart(self):
        """
        Restart the currently running LLM service.

        Returns:
            bool: True if the service restarted successfully, False otherwise.
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
            llm_port = service_info.get("port")
            context_length = service_info.get("context_length")

            logger.info(f"Restarting LLM service '{hash}' running on port {port}...")

            # Stop the current service
            self.stop()

            # Start the service with the same parameters
            return self.start(hash, port, context_length=context_length)
        except Exception as e:
            logger.error(f"Error restarting LLM service: {str(e)}", exc_info=True)
            return False

    def start(self, hash: str, port: int = 11434, host: str = "0.0.0.0", context_length: int = 32768) -> bool:
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
            
            # Reset activity timer when starting a new model
            self.track_activity()
            
            local_model_path = asyncio.run(download_model_from_filecoin_async(hash))
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

            service_metadata = {
                "hash": hash,
                "port": llm_running_port,
                "multimodal": False,
                "local_text_path": local_model_path,
                "app_port": port,
                "context_length": context_length,
                "last_activity": time.time()
            }
            filecoin_url = f"https://gateway.lighthouse.storage/ipfs/{hash}"
            is_gemma = False
            for attempt in range(3):
                try:
                    response = requests.get(filecoin_url, timeout=10)
                    if response.status_code == 200:
                        response_json = response.json()
                        service_metadata["family"] = response_json.get("family", "")
                        folder_name = response_json["folder_name"]
                        is_gemma = True if "gemma" in folder_name.lower() else False
                        break
                except requests.exceptions.RequestException:
                    time.sleep(5)  # Delay between retries

            if is_gemma:
                # Use pkg_resources to get the absolute path to the template file
                template_path = pkg_resources.resource_filename("local_llms", "examples/gemma3_template.jinja")
                
                running_llm_command = [
                    llama_server_path,
                    "--jinja",
                    "--model", str(local_model_path),
                    "--port", str(llm_running_port),
                    "--host", host,
                    "-c", str(context_length),
                    "-fa",
                    "--pooling", "cls",
                    "--no-webui",
                    "-ngl", "-1",
                    "--no-mmap",
                    "--mlock",
                    "--chat-template-file", template_path
                ]
            else:
                running_llm_command = [
                    llama_server_path,
                    "--jinja",
                    "--model", str(local_model_path),
                    "--port", str(llm_running_port),
                    "--host", host,
                    "-c", str(context_length),
                    "-fa",
                    "--pooling", "cls",
                    "-ngl", "-1",
                    "--no-mmap",
                    "--mlock",
                    "--no-webui"
                ]
            # Add memory optimization parameters if defined
            memory_limit = os.getenv("LLM_MEMORY_LIMIT")
            if memory_limit:
                running_llm_command.extend(["--memory-limit", memory_limit])
                
            # Add dynamic quantization option if available
            if os.getenv("LLM_DYNAMIC_QUANT", "false").lower() == "true":
                running_llm_command.append("--dynamic-quant")

            logger.info(f"Starting process: {' '.join(running_llm_command)}")

            # Create log files for stdout and stderr for LLM process
            os.makedirs("logs", exist_ok=True)
            llm_log_stdout = Path(f"logs/llm_stdout_{llm_running_port}.log")
            llm_log_stderr = Path(f"logs/llm_stderr_{llm_running_port}.log")
            llm_process = None
            try:
                with open(llm_log_stdout, 'w') as stdout_log, open(llm_log_stderr, 'w') as stderr_log:
                    llm_process = subprocess.Popen(
                        running_llm_command,
                        stdout=stdout_log,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"LLM logs written to {llm_log_stdout} and {llm_log_stderr}")
            except Exception as e:
                logger.error(f"Error starting LLM service: {str(e)}", exc_info=True)
                return False
    
            if not self._wait_for_service(llm_running_port):
                logger.error(f"Service failed to start within 600 seconds")
                llm_process.terminate()
                return False


            # start the FastAPI app in the background           
            uvicorn_command = [
                "uvicorn",
                "local_llms.apis:app",
                "--host", host,
                "--port", str(port),
                "--log-level", "info"
            ]


            logger.info(f"Starting process: {' '.join(uvicorn_command)}")

            # Create log files for stdout and stderr
            os.makedirs("logs", exist_ok=True)
            log_path_stdout = Path(f"logs/api_stdout_{port}.log")
            log_path_stderr = Path(f"logs/api_stderr_{port}.log")
            
            try:
                with open(log_path_stdout, 'w') as stdout_log, open(log_path_stderr, 'w') as stderr_log:
                    apis_process = subprocess.Popen(
                        uvicorn_command,
                        stdout=stdout_log,
                        stderr=stderr_log,
                        preexec_fn=os.setsid
                    )
                logger.info(f"API logs written to {log_path_stdout} and {log_path_stderr}")
            except Exception as e:
                logger.error(f"Error starting FastAPI app: {str(e)}", exc_info=True)
                llm_process.terminate()
                return False
            
            if not self._wait_for_service(port):
                logger.error(f"API service failed to start within 600 seconds")
                llm_process.terminate()
                apis_process.terminate()
                return False

            logger.info(f"Service started on port {port} for model: {hash}")

            service_metadata["pid"] =llm_process.pid
            service_metadata["app_pid"] = apis_process.pid
            projector_path = f"{local_model_path}-projector"    
            if os.path.exists(projector_path):
                service_metadata["multimodal"] = True
                service_metadata["local_projector_path"] = projector_path

            self._dump_running_service(service_metadata)    

            # update service metadata to the FastAPI app
            try:
                update_url = f"http://localhost:{port}/update"
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
        with open(self.pickle_file, "wb") as f:
            pickle.dump(metadata, f)
            
    def update_activity(self):
        """Update the last activity timestamp for the running model."""
        if not self.pickle_file.exists():
            return False
        
        try:
            # Load service details
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            # Update timestamp
            service_info["last_activity"] = time.time()
            self.track_activity()
            
            # Save updated info
            with open(self.pickle_file, "wb") as f:
                pickle.dump(service_info, f)
                
            return True
        except Exception as e:
            logger.error(f"Error updating activity timestamp: {str(e)}")
            return False

    def get_running_model(self) -> Optional[str]:
        """
        Get currently running model hash if the service is healthy.

        Returns:
            Optional[str]: Running model hash or None if no healthy service exists.
        """
        if not self.pickle_file.exists():
            return None

        try:
            # Load service info from pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            model_hash = service_info.get("hash")
            app_port = service_info.get("app_port")
            llm_port = service_info.get("port")
            context_length = service_info.get("context_length")
            
            # Update activity timestamp
            self.track_activity()
            
            # Check both services with minimal timeout
            llm_healthy = False
            api_healthy = False
            
            # Use a single session for connection pooling
            with requests.Session() as session:
                try:
                    llm_status = session.get(f"http://localhost:{llm_port}/health", timeout=2)
                    llm_healthy = llm_status.status_code == 200
                except requests.exceptions.RequestException:
                    pass
            
                try:
                    app_status = session.get(f"http://localhost:{app_port}/v1/health", timeout=2)
                    api_healthy = app_status.status_code == 200
                except requests.exceptions.RequestException:
                    pass

            if llm_healthy and api_healthy:
                # Update the last activity timestamp whenever we verify a model is running
                self.update_activity()
                return model_hash
                
            logger.warning(f"Service not healthy: LLM {llm_healthy}, API {api_healthy}")
            self.stop()  
            try:
                logger.info("Restarting service...")  
                if self.start(model_hash, app_port, context_length=context_length):
                    return model_hash
                return None
            except Exception as e:
                logger.error(f"Failed to restart service: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error getting running model: {str(e)}")
            return None
    
    def unload_model(self) -> bool:
        """
        Unload the model from memory but keep the server running.
        
        Returns:
            bool: True if model was unloaded successfully, False otherwise.
        """
        if not self.pickle_file.exists():
            logger.warning("No running LLM service to unload.")
            return False
            
        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
                
            app_port = service_info.get("app_port")
            
            # Send unload signal to the API
            unload_url = f"http://localhost:{app_port}/unload"
            try:
                response = requests.post(unload_url, timeout=10)
                if response.status_code == 200:
                    logger.info("Model unloaded from memory successfully.")
                    
                    # Force garbage collection to free memory
                    gc.collect()
                    
                    # Update pickle file with unloaded state
                    service_info["unloaded"] = True
                    with open(self.pickle_file, "wb") as f:
                        pickle.dump(service_info, f)
                        
                    return True
                else:
                    logger.error(f"Failed to unload model: {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Error unloading model: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error during model unload: {str(e)}")
            return False

    def reload_model(self) -> bool:
        """
        Reload a previously unloaded model.
        
        Returns:
            bool: True if model was reloaded successfully, False otherwise.
        """
        if not self.pickle_file.exists():
            logger.warning("No model service to reload.")
            return False
            
        try:
            # Load service details
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
                
            if not service_info.get("unloaded", False):
                logger.info("Model is not unloaded, no need to reload.")
                return True
                
            app_port = service_info.get("app_port")
            model_path = service_info.get("local_text_path")
            
            # Send reload signal to the API
            reload_url = f"http://localhost:{app_port}/reload"
            payload = {"model_path": model_path}
            
            try:
                response = requests.post(reload_url, json=payload, timeout=60)
                if response.status_code == 200:
                    logger.info("Model reloaded successfully.")
                    
                    # Update pickle file with loaded state
                    service_info["unloaded"] = False
                    service_info["last_activity"] = time.time()
                    self.track_activity()
                    with open(self.pickle_file, "wb") as f:
                        pickle.dump(service_info, f)
                        
                    return True
                else:
                    logger.error(f"Failed to reload model: {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Error reloading model: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error during model reload: {str(e)}")
            return False

    def stop(self) -> bool:
        """
        Stop the running LLM service.

        Returns:
            bool: True if the service stopped successfully, False otherwise.
        """
        if not os.path.exists(self.pickle_file):
            logger.warning("No running LLM service to stop.")
            return False

        try:
            # Load service details from the pickle file
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
            
            hash = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            app_port = service_info.get("app_port")

            logger.info(f"Stopping LLM service '{hash}' running on port {app_port} (PID: {app_pid})...")

            # Trigger unload first to properly release memory
            try:
                self.unload_model()
            except:
                pass
                
            # Terminate process by PID
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                process.terminate()
                process.wait(timeout=120)  # Allow process to shut down gracefully
                
                if process.is_running():  # Force kill if still alive
                    logger.warning("Process did not terminate, forcing kill.")
                    process.kill()

            # Terminate FastAPI app by PIDd
            if psutil.pid_exists(app_pid):
                app_process = psutil.Process(app_pid)
                app_process.terminate()
                app_process.wait(timeout=120)  # Allow process to shut down gracefully
                
                if app_process.is_running():  # Force kill if still alive
                    logger.warning("API process did not terminate, forcing kill.")
                    app_process.kill()

            # Force garbage collection to free memory
            gc.collect()

            # Remove the tracking file
            os.remove(self.pickle_file)
            logger.info("LLM service stopped successfully.")
            return True

        except Exception as e:
            logger.error(f"Error stopping LLM service: {str(e)}", exc_info=True)
            return False

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get detailed memory usage statistics for the running model and system.
        
        Returns:
            Dict[str, Any]: Dictionary with memory usage statistics
        """
        if not self.pickle_file.exists():
            logger.warning("No running model found")
            return None
            
        try:
            with open(self.pickle_file, "rb") as f:
                service_info = pickle.load(f)
                
            model_hash = service_info.get("hash")
            pid = service_info.get("pid")
            app_pid = service_info.get("app_pid")
            
            result = {
                "model_hash": model_hash,
                "timestamp": time.time(),
                "unloaded": service_info.get("unloaded", False),
                "processes": {},
                "system": {},
                "rss_mb": 0,  # Total RSS across all processes
                "vms_mb": 0,  # Total VMS across all processes
            }
            
            # System memory information
            try:
                sys_memory = psutil.virtual_memory()
                result["system"] = {
                    "total_gb": sys_memory.total / (1024**3),
                    "available_gb": sys_memory.available / (1024**3),
                    "used_gb": (sys_memory.total - sys_memory.available) / (1024**3),
                    "percent": sys_memory.percent
                }
                result["percent"] = sys_memory.percent  # System memory usage percentage
            except Exception as e:
                logger.error(f"Error getting system memory info: {e}")
                result["system"] = {"error": str(e)}
            
            # Get memory usage of model server process and its children
            total_rss = 0
            total_vms = 0
            
            if pid and psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    proc_info = {
                        "rss": memory_info.rss,
                        "rss_mb": memory_info.rss / (1024 * 1024),
                        "vms": memory_info.vms,
                        "vms_mb": memory_info.vms / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "create_time": process.create_time(),
                        "running_time_minutes": (time.time() - process.create_time()) / 60
                    }
                    
                    # Add to totals
                    total_rss += memory_info.rss
                    total_vms += memory_info.vms
                    
                    # Try to get memory maps info for shared vs private memory
                    try:
                        memory_maps = process.memory_maps()
                        private_rss = sum(m.private for m in memory_maps)
                        shared_rss = sum(m.shared for m in memory_maps)
                        proc_info["private_rss_mb"] = private_rss / (1024 * 1024)
                        proc_info["shared_rss_mb"] = shared_rss / (1024 * 1024)
                    except:
                        # Memory maps might not be available on all platforms
                        pass
                        
                    # Check for child processes
                    children = []
                    try:
                        for child in process.children(recursive=True):
                            child_info = child.memory_info()
                            children.append({
                                "pid": child.pid,
                                "rss_mb": child_info.rss / (1024 * 1024),
                                "vms_mb": child_info.vms / (1024 * 1024)
                            })
                            total_rss += child_info.rss
                            total_vms += child_info.vms
                    except:
                        pass
                        
                    if children:
                        proc_info["children"] = children
                        
                    result["processes"]["model_server"] = proc_info
                except Exception as e:
                    logger.error(f"Error getting model process info: {e}")
                    result["processes"]["model_server"] = {"error": str(e)}
            else:
                result["processes"]["model_server"] = {"error": "Process not found"}
                
            # Get memory usage of API server process
            if app_pid and psutil.pid_exists(app_pid):
                try:
                    process = psutil.Process(app_pid)
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    proc_info = {
                        "rss": memory_info.rss,
                        "rss_mb": memory_info.rss / (1024 * 1024),
                        "vms": memory_info.vms,
                        "vms_mb": memory_info.vms / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "create_time": process.create_time(),
                        "running_time_minutes": (time.time() - process.create_time()) / 60
                    }
                    
                    # Add to totals
                    total_rss += memory_info.rss
                    total_vms += memory_info.vms
                    
                    result["processes"]["api_server"] = proc_info
                except Exception as e:
                    logger.error(f"Error getting API process info: {e}")
                    result["processes"]["api_server"] = {"error": str(e)}
            else:
                result["processes"]["api_server"] = {"error": "Process not found"}
                
            # Add totals
            result["rss_mb"] = total_rss / (1024 * 1024)
            result["vms_mb"] = total_vms / (1024 * 1024)
            
            # Add model information
            result["model_info"] = {
                "context_length": service_info.get("context_length", 0),
                "family": service_info.get("family", "unknown"),
                "multimodal": service_info.get("multimodal", False),
                "app_port": service_info.get("app_port", 0),
                "llm_port": service_info.get("port", 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}", exc_info=True)
            return {"error": str(e)}
        