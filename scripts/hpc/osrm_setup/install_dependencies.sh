#!/bin/bash
# install_dependencies.sh - Initialize VM with Docker, osmium, and Python

set -e

echo "=== Starting VM initialization ==="
echo "Timestamp: $(date)"

# Update system
echo "=== Updating system packages ==="
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
echo "=== Installing Docker ==="
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
else
    echo "Docker already installed"
fi

# Install osmium
echo "=== Installing osmium ==="
sudo apt-get install -y osmium-tool

# Install Python and dependencies
echo "=== Setting up Python environment ==="
sudo apt-get install -y python3-pip python3-venv python3-dev libgeos-dev

python3 -m venv ~/osrm_env
source ~/osrm_env/bin/activate
pip install --upgrade pip
pip install geopandas pandas h3 requests pyproj shapely fiona

# Create directories
echo "=== Creating directories ==="
mkdir -p ~/osrm-data ~/cities ~/results

# Pull OSRM Docker image
echo "=== Pulling OSRM Docker image ==="
sudo docker pull osrm/osrm-backend

echo "=== VM initialization complete ==="
echo "Timestamp: $(date)"
echo "NOTE: Log out and back in for Docker permissions to take effect"
