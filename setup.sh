# !/bin/bash

rm -rf venv
python3 -m venv venv # Create virtual environment 
source venv/bin/activate # Activate virtual environment
pip uninstall gym -y # Uninstall old gym package 
pip install -r requirements.txt # Install requirements
