"""Streamlit entry point for the graph robustness toolkit."""

from pathlib import Path

from gnn_attack_detector.web_app import render_app


PROJECT_ROOT = Path(__file__).resolve().parent

render_app(PROJECT_ROOT)
