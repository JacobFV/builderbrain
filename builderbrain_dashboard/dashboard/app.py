"""
BuilderBrain Dashboard - Main Streamlit Application

Real-time monitoring and visualization dashboard for BuilderBrain training and inference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime, timedelta
import psutil
import time
from typing import Dict, List, Any, Optional

# Add parent directory to path for BuilderBrain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from builderbrain_dashboard.dashboard.pages.overview import render_overview_page
from builderbrain_dashboard.dashboard.pages.training import render_training_page
from builderbrain_dashboard.dashboard.pages.inference import render_inference_page
from builderbrain_dashboard.dashboard.pages.config import render_config_page
from builderbrain_dashboard.dashboard.utils.data_loader import DataLoader
from builderbrain_dashboard.dashboard.utils.api_client import APIClient


class BuilderBrainDashboard:
    """Main dashboard application class."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.api_client = APIClient()

        # Initialize session state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Overview'

        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True

        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5  # seconds

        # Available pages
        self.pages = {
            'Overview': render_overview_page,
            'Training': render_training_page,
            'Inference': render_inference_page,
            'Configuration': render_config_page,
        }

    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("🧠 BuilderBrain Dashboard")

        # Page selection
        st.sidebar.subheader("Navigation")
        for page_name in self.pages.keys():
            if st.sidebar.button(
                page_name,
                key=f"nav_{page_name.lower()}",
                use_container_width=True,
                type='primary' if st.session_state.current_page == page_name else 'secondary'
            ):
                st.session_state.current_page = page_name
                st.rerun()

        # Auto-refresh settings
        st.sidebar.subheader("Auto-Refresh")
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.auto_refresh
        )

        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=st.session_state.refresh_interval,
                step=1
            )

        # System status
        st.sidebar.subheader("System Status")

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.sidebar.progress(cpu_percent / 100, text=f"CPU: {cpu_percent:.1f}%")

        # Memory usage
        memory = psutil.virtual_memory()
        st.sidebar.progress(memory.percent / 100, text=f"Memory: {memory.percent:.1f}%")

        # Disk usage
        disk = psutil.disk_usage('/')
        st.sidebar.progress(disk.percent / 100, text=f"Disk: {disk.percent:.1f}%")

        # Active processes
        st.sidebar.metric("Active Processes", len(psutil.pids()))

        # Last updated
        st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

    def render_main_content(self):
        """Render the main content area."""
        page_renderer = self.pages[st.session_state.current_page]
        page_renderer(self.data_loader, self.api_client)

    def run(self):
        """Run the dashboard application."""
        # Set page config
        st.set_page_config(
            page_title="BuilderBrain Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
        }
        .sidebar-content {
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Render sidebar
        self.render_sidebar()

        # Render main content
        st.markdown('<h1 class="main-header">🧠 BuilderBrain Dashboard</h1>', unsafe_allow_html=True)
        self.render_main_content()

        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()


def main():
    """Main entry point for the dashboard."""
    dashboard = BuilderBrainDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
