#!/usr/bin/env python3
"""
gui/app.py
==========
Refactored PyQt5-based GUI for onedcelltrack.

This application lets the user select an ND2 image file, view frames,
and update the displayed frame using a slider.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from nd2reader import ND2Reader

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSlider, QPushButton, QHBoxLayout, QVBoxLayout,
    QWidget, QFileDialog, QLabel, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Onedcelltrack - GUI")
        self.resize(1200, 800)
        
        self.nd2_file = None
        self.f = None
        self.current_frame = 0
        
        # Create central widget and layouts
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # File selection button
        self.open_button = QPushButton("Select ND2 Image")
        self.open_button.clicked.connect(self.openFileDialog)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.updateFrame)
        
        self.frame_label = QLabel("Frame: 0")
        
        # Matplotlib figure for image display
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Initialize with a placeholder image
        self.image_handle = self.ax.imshow(np.zeros((100, 100)), cmap='gray', vmin=0, vmax=255)
        self.ax.axis('off')
        self.fig.tight_layout()
        
        # Layout for control widgets
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Main layout splits controls and image display
        main_layout = QHBoxLayout(central_widget)
        main_layout.addLayout(controls_layout)
        
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.canvas)
        image_layout.addWidget(self.toolbar)
        main_layout.addLayout(image_layout)
        
    def openFileDialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select ND2 Image", "", "ND2 Files (*.nd2)")
        if file_path:
            self.nd2_file = file_path
            self.f = ND2Reader(self.nd2_file)
            num_frames = self.f.sizes['t']
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_slider.setValue(0)
            self.updateFrame(0)
            
    def updateFrame(self, value):
        if self.f is None:
            return
        self.current_frame = value
        self.frame_label.setText(f"Frame: {value}")
        # Retrieve frame from ND2 file
        frame_img = self.f.get_frame_2D(t=value)
        self.image_handle.set_data(frame_img)
        self.ax.set_title(f"Frame {value}")
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
