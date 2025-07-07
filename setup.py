#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:24:57 2025

@author: tsukudayuuga
"""

from setuptools import setup

APP = ['cam_GUI.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'includes': [
        'ezdxf',
        'ezdxf.math.bulge_to_arc',
        'numpy',
        'shapely.geometry',
        'shapely.ops',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.collections',
        'matplotlib.patches',
        'PySide6',
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'csm_new',
        'cam_merged_dxf',
        'scipy',
        'scipy.spatial',  # for KDTree
    ],
    'packages': [
        'ezdxf',
        'numpy',
        'shapely',
        'matplotlib',
        'PySide6',
        'scipy',
    ],
    'plist': {
        'CFBundleName': 'GcodeGeneratorGUI',
        'CFBundleIdentifier': 'com.yourname.gcodegui',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0',
    },
    # アイコンファイルがあれば指定
    'iconfile': 'MyIcon.icns',
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    install_requires=['ezdxf', 'numpy', 'shapely', 'matplotlib', 'PySide6', 'scipy']
)
