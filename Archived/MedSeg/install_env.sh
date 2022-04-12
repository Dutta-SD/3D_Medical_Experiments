#!/bin/bash

# Checks
cd "$(dirname '$0')"
echo "WORKING AT $(pwd)"
python3 --version
pip3 --version

# Installation
pip3 install --quiet pipenv


# pipenv install
pipenv update
pipenv install --dev