#!/usr/bin/env python3
"""
Script to install memory optimization dependencies for BERT training.
"""

import subprocess
import sys


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False


def main():
    print("📦 Installing memory optimization packages...")

    packages = [
        "bitsandbytes",  # For 8-bit optimizers
        "transformers>=4.20.0",  # For gradient checkpointing improvements
    ]

    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1

    print(f"\n📊 Installation Summary:")
    print(f"   Successful: {success_count}/{len(packages)}")

    if success_count == len(packages):
        print("🎉 All memory optimization packages installed successfully!")
        print("\n🔧 To verify installation, run:")
        print("   python -c 'import bitsandbytes as bnb; print(\"bitsandbytes version:\", bnb.__version__)'")
    else:
        print("⚠️  Some packages failed to install. Training will fallback to standard optimizers.")

    print("\n💡 Tips for memory optimization:")
    print("   - Use smaller batch sizes (2-4)")
    print("   - Enable gradient accumulation")
    print("   - Use mixed precision training")
    print("   - Enable gradient checkpointing")
    print("   - Clear CUDA cache regularly")


if __name__ == "__main__":
    main()