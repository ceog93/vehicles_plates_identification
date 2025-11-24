#!/usr/bin/env python
"""Wrapper to run the detector+OCR pipeline on video_prueba.mp4"""
import os
import sys

os.chdir(r'c:\AI_PROJECT\vehicles_plates_identification')
sys.path.insert(0, os.getcwd())

# Override MODEL_PATH BEFORE importing config
import src.config
src.config.MODEL_PATH = r'02_models/model_20251122_2238M21_EfficientNetB0/detector_model.keras'

from src.inference.run_detector_and_ocr import main

# Mock argv to pass args to argparse
sys.argv = [
    'run_detector_and_ocr.py',
    '--input', '03_production/input_feed/video_prueba.mp4',
    '--ocr', 'saved_models/ocr_model.h5'
]

if __name__ == '__main__':
    main()
