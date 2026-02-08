#!/usr/bin/env python3
import os
folders = ['data', 'models', 'outputs', 'outputs/eda', 'outputs/models', 'mlruns']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✓ Created: {folder}")
