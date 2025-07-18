#!/usr/bin/env python3
"""
Main script to run the Greyhound TrueSkill Analytics Dashboard.

Usage:
    python run_dashboard.py
    or
    streamlit run run_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the Streamlit dashboard."""
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    dashboard_script = script_dir / "analytics" / "dashboard.py"
    
    if not dashboard_script.exists():
        print("❌ Dashboard script not found!")
        print(f"Expected: {dashboard_script}")
        sys.exit(1)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not installed. Please run: poetry install")
        sys.exit(1)
    
    # Run the dashboard
    print("🚀 Starting Greyhound TrueSkill Analytics Dashboard...")
    print("📊 Dashboard will open in your web browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_script),
            "--server.headless", "false",
            "--server.enableCORS", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()