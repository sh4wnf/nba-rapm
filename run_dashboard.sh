#!/bin/bash
# Simple script to run the Streamlit dashboard

cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run app.py
