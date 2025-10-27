# !/bin/bash

python3 -m venv venv # Create virtual environment 
source venv/bin/activate # Activate virtual environment
pip uninstall gym -y # Uninstall old gym package 
pip install -r requirements.tx # Install requirements
