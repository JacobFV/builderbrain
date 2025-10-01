#!/usr/bin/env python3
"""
Demo script for BuilderBrain Dashboard.

Shows how to run the dashboard and interact with the API.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'plotly', 'fastapi', 'uvicorn', 'pandas', 'psutil'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing with uv...")
        subprocess.run(['uv', 'add'] + missing_packages, check=True)
        return True

    return False


def start_api_server():
    """Start the FastAPI server in background."""
    print("🚀 Starting BuilderBrain API server...")

    api_process = subprocess.Popen([
        sys.executable, '-m', 'huggingface_pipeline.inference_api.app'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for server to start
    time.sleep(3)

    if api_process.poll() is None:
        print("✅ API server started successfully")
        return api_process
    else:
        print("❌ Failed to start API server")
        return None


def start_dashboard():
    """Start the Streamlit dashboard."""
    print("📊 Starting BuilderBrain Dashboard...")

    dashboard_process = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run',
        'builderbrain_dashboard/dashboard/app.py',
        '--server.port', '8501',
        '--server.address', 'localhost'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for dashboard to start
    time.sleep(5)

    if dashboard_process.poll() is None:
        print("✅ Dashboard started successfully")
        return dashboard_process
    else:
        print("❌ Failed to start dashboard")
        return None


def demo_workflow():
    """Run a demo workflow showing dashboard features."""

    print("\n🎯 Demo Workflow:")
    print("=" * 50)

    print("\n1. 📈 Training Monitoring")
    print("   - Navigate to 'Training' tab")
    print("   - View loss curves and constraint evolution")
    print("   - Monitor dual variable adjustments")

    print("\n2. 🧠 Interactive Inference")
    print("   - Go to 'Inference' tab")
    print("   - Try grammar-constrained generation")
    print("   - Test plan validation")

    print("\n3. ⚙️ Configuration Management")
    print("   - Visit 'Configuration' tab")
    print("   - View model parameters and constraints")
    print("   - Monitor system health")

    print("\n4. 📊 Real-time Monitoring")
    print("   - Check the 'Overview' tab")
    print("   - Watch live system metrics")
    print("   - Monitor constraint compliance")

    print("\n💡 Tips:")
    print("   - Use auto-refresh for real-time updates")
    print("   - Export training data as CSV")
    print("   - Test different model scales")
    print("   - Try grammar constraint validation")

    print("\n🌐 Access URLs:")
    print("   Dashboard: http://localhost:8501")
    print("   API: http://localhost:8000/docs")


def main():
    """Main demo function."""
    print("🧠 BuilderBrain Dashboard Demo")
    print("=" * 50)

    # Check and install dependencies
    if check_dependencies():
        print("Dependencies installed. Please run the demo again.")
        return

    # Start services
    api_process = start_api_server()
    dashboard_process = start_dashboard()

    if not api_process or not dashboard_process:
        print("❌ Failed to start services")
        return

    # Show demo workflow
    demo_workflow()

    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        print("Could not open browser automatically")

    print("\n🔄 Services are running. Press Ctrl+C to stop.")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)

            # Check if processes are still running
            if api_process.poll() is not None:
                print("❌ API server crashed")
                break
            if dashboard_process.poll() is not None:
                print("❌ Dashboard crashed")
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")

    finally:
        # Clean up processes
        if api_process:
            api_process.terminate()
            api_process.wait()

        if dashboard_process:
            dashboard_process.terminate()
            dashboard_process.wait()

        print("✅ Services stopped")


if __name__ == "__main__":
    main()
