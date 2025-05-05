#!/bin/bash
set -o pipefail

# Logging functions
log_message() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --message \"$message\""
    fi
}

log_error() {
    local message="$1"
    if [[ -n "${message// }" ]]; then
        echo "[LAUNCHER_LOGGER] [MODEL_INSTALL_LLAMA] --error \"$message\"" >&2
    fi
}

# Error handling function
handle_error() {
    local exit_code=$1
    local error_msg=$2
    log_error "$error_msg (Exit code: $exit_code)"
    
    # Clean up if needed
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_message "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

log_message "Creating virtual environment 'local_llms'..."
"$PYTHON_CMD" -m venv local_llms || handle_error $? "Failed to create virtual environment"

log_message "Activating virtual environment..."
source local_llms/bin/activate || handle_error $? "Failed to activate virtual environment"
log_message "Virtual environment activated."

log_message "Setting up local-llms toolkit..."
if pip show local-llms &>/dev/null; then
    log_message "local-llms is installed. Checking for updates..."
    
    # Get installed version
    INSTALLED_VERSION=$(pip show local-llms | grep Version | awk '{print $2}')
    log_message "Current version: $INSTALLED_VERSION"
    
    # Get remote version (from GitHub repository without installing)
    log_message "Checking latest version from repository..."
    TEMP_VERSION_FILE=$(mktemp)
    if curl -s https://raw.githubusercontent.com/eternalai-org/local-llms-wrapper/main/local_llms/__init__.py | grep -o "__version__ = \"[0-9.]*\"" | cut -d'"' -f2 > "$TEMP_VERSION_FILE"; then
        REMOTE_VERSION=$(cat "$TEMP_VERSION_FILE")
        rm "$TEMP_VERSION_FILE"
        
        log_message "Latest version: $REMOTE_VERSION"
        
        # Compare versions
        if [ "$INSTALLED_VERSION" != "$REMOTE_VERSION" ]; then
            log_message "New version available. Updating..."
            pip uninstall local-llms -y || handle_error $? "Failed to uninstall local-llms"
            pip install -q git+https://github.com/eternalai-org/local-llms-wrapper.git || handle_error $? "Failed to update local-llms toolkit"
            log_message "local-llms toolkit updated to version $REMOTE_VERSION."
        else
            log_message "Already running the latest version. No update needed."
        fi
    else
        log_message "Could not check latest version. Proceeding with update to be safe..."
        pip uninstall local-llms -y || handle_error $? "Failed to uninstall local-llms"
        pip install -q git+https://github.com/eternalai-org/local-llms-wrapper.git || handle_error $? "Failed to update local-llms toolkit"
        log_message "local-llms toolkit updated."
    fi
else
    log_message "Installing local-llms toolkit..."
    pip install -q git+https://github.com/eternalai-org/local-llms-wrapper.git || handle_error $? "Failed to install local-llms toolkit"
    log_message "local-llms toolkit installed."
fi

log_message "Setup completed successfully."