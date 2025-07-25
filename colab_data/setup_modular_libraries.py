#!/usr/bin/env python3
"""
Setup script for modular libraries in Google Colab
Run this first to ensure all libraries are properly configured
"""

import os
import sys
import subprocess
from pathlib import Path

def check_and_mount_drive():
    """Check if Google Drive is mounted, mount if necessary"""
    drive_path = '/content/drive'
    if not os.path.exists(drive_path):
        try:
            from google.colab import drive
            print("🔄 Mounting Google Drive...")
            drive.mount('/content/drive')
            print("✅ Google Drive mounted successfully")
            return True
        except ImportError:
            print("⚠️ Not running in Google Colab - skipping drive mount")
            return False
        except Exception as e:
            print(f"❌ Error mounting Google Drive: {e}")
            return False
    else:
        print("✅ Google Drive already mounted")
        return True

def find_lib_directory():
    """Find the lib directory with modular libraries"""
    possible_paths = [
        '/content/drive/MyDrive/TesisMagister/acumulative/colab_data/lib',
        '/content/drive/MyDrive/TesisMagister/acumulative/colab_data',
        './lib',
        '../lib',
        str(Path.cwd() / 'lib'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it contains our libraries
            expected_files = [
                'colab_setup.py',
                'evaluation_metrics.py', 
                'rag_evaluation.py',
                'data_manager.py',
                'results_processor.py'
            ]
            
            if os.path.isdir(path):
                files_in_path = os.listdir(path)
                if all(f in files_in_path for f in expected_files):
                    print(f"📂 Found modular libraries in: {path}")
                    return path
    
    print("❌ Modular libraries not found")
    return None

def check_library_imports(lib_path):
    """Test if libraries can be imported"""
    if lib_path:
        sys.path.insert(0, lib_path)
    
    libraries_to_test = [
        ('colab_setup', 'quick_setup'),
        ('evaluation_metrics', 'create_metrics_calculator'),
        ('rag_evaluation', 'create_rag_pipeline'),
        ('data_manager', 'create_data_pipeline'), 
        ('results_processor', 'process_and_save_results')
    ]
    
    import_results = {}
    
    for lib_name, function_name in libraries_to_test:
        try:
            module = __import__(lib_name)
            if hasattr(module, function_name):
                import_results[lib_name] = "✅ OK"
            else:
                import_results[lib_name] = f"⚠️ Missing {function_name}"
        except ImportError as e:
            import_results[lib_name] = f"❌ Import Error: {e}"
        except Exception as e:
            import_results[lib_name] = f"❌ Error: {e}"
    
    print("\n📚 Library Import Status:")
    for lib, status in import_results.items():
        print(f"  {lib}: {status}")
    
    return all("✅" in status for status in import_results.values())

def install_requirements():
    """Install required packages"""
    required_packages = [
        'sentence-transformers',
        'ragas',
        'datasets', 
        'bert-score',
        'openai',
        'scikit-learn',
        'tqdm',
        'pytz'
    ]
    
    print("📦 Installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package} already installed")
        except ImportError:
            print(f"  🔄 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"  ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {package}: {e}")

def create_init_file(lib_path):
    """Create __init__.py if it doesn't exist"""
    if lib_path:
        init_file = os.path.join(lib_path, '__init__.py')
        if not os.path.exists(init_file):
            print("📄 Creating __init__.py file...")
            with open(init_file, 'w') as f:
                f.write('# Modular Libraries Package\n')
                f.write('__version__ = "2.0.0"\n')
            print("✅ __init__.py created")
        else:
            print("✅ __init__.py already exists")

def main():
    """Main setup function"""
    print("🚀 Setting up modular libraries for Colab...")
    print("=" * 50)
    
    # Step 1: Mount Google Drive
    drive_mounted = check_and_mount_drive()
    
    # Step 2: Find library directory
    lib_path = find_lib_directory()
    
    if not lib_path:
        print("\n❌ Setup failed: Cannot find modular libraries")
        print("💡 Make sure the libraries are in one of these locations:")
        print("  - /content/drive/MyDrive/TesisMagister/acumulative/colab_data/lib/")
        print("  - ./lib/")
        print("  - ../lib/")
        return False
    
    # Step 3: Install requirements
    install_requirements()
    
    # Step 4: Create init file
    create_init_file(lib_path)
    
    # Step 5: Test imports
    imports_successful = check_library_imports(lib_path)
    
    print("\n" + "=" * 50)
    if imports_successful and lib_path:
        print("🎉 Setup completed successfully!")
        print(f"📂 Libraries path: {lib_path}")
        print("🔧 You can now run the main evaluation notebook")
        
        # Provide usage example
        print("\n📝 Usage example:")
        print("```python")
        print("import sys")
        print(f"sys.path.append('{lib_path}')")
        print("from colab_setup import quick_setup")
        print("setup_result = quick_setup()")
        print("```")
        
        return True
    else:
        print("❌ Setup failed - please check the errors above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)