#!/usr/bin/env python3
"""
Script to upload modular libraries to Google Drive for Colab access
"""

import os
import shutil
from pathlib import Path

def copy_libraries_to_drive():
    """Copy all modular libraries to the correct Google Drive location"""
    
    # Source directory (where libraries are created locally)
    source_dir = Path(__file__).parent / 'lib'
    
    # Target directory (Google Drive path structure)
    # This assumes your Google Drive is mounted at the standard location
    target_base = Path.home() / 'Google Drive' / 'TesisMagister' / 'acumulative' / 'colab_data'
    
    # Alternative paths to try
    possible_targets = [
        target_base,
        Path.home() / 'GoogleDrive' / 'TesisMagister' / 'acumulative' / 'colab_data',
        Path('/Volumes/GoogleDrive/My Drive/TesisMagister/acumulative/colab_data'),  # macOS
        Path('~/Google Drive/TesisMagister/acumulative/colab_data').expanduser(),
    ]
    
    target_dir = None
    for possible_target in possible_targets:
        if possible_target.exists():
            target_dir = possible_target / 'lib'
            break
    
    if not target_dir:
        print("❌ Could not find Google Drive directory")
        print("📁 Please manually copy the lib folder to:")
        print("   Google Drive/TesisMagister/acumulative/colab_data/lib/")
        print(f"🔍 Source folder: {source_dir}")
        return False
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all .py files
    copied_files = []
    for py_file in source_dir.glob('*.py'):
        target_file = target_dir / py_file.name
        shutil.copy2(py_file, target_file)
        copied_files.append(py_file.name)
        print(f"✅ Copied: {py_file.name}")
    
    print(f"\n🎉 Successfully copied {len(copied_files)} files to:")
    print(f"📁 {target_dir}")
    print(f"📋 Files: {', '.join(copied_files)}")
    
    return True

if __name__ == "__main__":
    print("🚀 Uploading modular libraries to Google Drive...")
    success = copy_libraries_to_drive()
    
    if success:
        print("\n✅ Upload completed!")
        print("🔄 You can now run the Colab notebook")
    else:
        print("\n❌ Upload failed")
        print("💡 Please manually copy the lib folder to Google Drive")