#!/usr/bin/env python3
"""
Build script for creating standalone alex binary using PyInstaller.

This script creates a standalone executable that can be distributed
without requiring Python to be installed on the target system.
"""

import os
import sys
import subprocess
from pathlib import Path

def build_binary():
    """Build standalone binary using PyInstaller."""
    
    print("🔨 Building alex-treBENCH standalone binary...")
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Define the build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "alex",
        "--onefile",
        "--console",
        "--clean",
        f"--distpath={project_root}/dist",
        f"--workpath={project_root}/build",
        f"--specpath={project_root}",
        # Include necessary data files
        f"--add-data={project_root}/config:config",
        f"--add-data={project_root}/src:src",
        # Hidden imports that PyInstaller might miss
        "--hidden-import=src.core.config",
        "--hidden-import=src.storage.models",
        "--hidden-import=src.models.openrouter",
        "--hidden-import=click",
        "--hidden-import=rich",
        "--hidden-import=sqlalchemy",
        # Entry point
        str(project_root / "alex_cli.py")
    ]
    
    try:
        # Install PyInstaller if not already installed
        print("📦 Checking PyInstaller installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
        print("🏗️  Building binary...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            binary_path = project_root / "dist" / ("alex.exe" if os.name == 'nt' else "alex")
            print(f"✅ Binary built successfully: {binary_path}")
            print(f"📏 Binary size: {binary_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Test the binary
            print("🧪 Testing binary...")
            test_result = subprocess.run([str(binary_path), "--help"], 
                                       capture_output=True, text=True)
            if test_result.returncode == 0:
                print("✅ Binary test successful!")
                return str(binary_path)
            else:
                print(f"❌ Binary test failed: {test_result.stderr}")
                return None
        else:
            print(f"❌ Build failed: {result.stderr}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during build: {e}")
        return None

if __name__ == "__main__":
    build_binary()