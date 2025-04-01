#!/usr/bin/env python3
"""
webapp/app.py
=============
Refactored Flask web application for onedcelltrack.

This app serves an experiment viewer page where users can update the displayed
image by selecting a frame, channel, and field-of-view (fov). It also registers
a simulation blueprint from simulations_app.
"""

import os
from flask import Flask, render_template, request, jsonify, Response
from nd2reader import ND2Reader
import numpy as np
from io import BytesIO
import base64
from PIL import Image

from onedcelltrack import functions  # Assuming functions contains utility methods
from webapp.simulations_app import SimulationsApp  # Import the simulation blueprint

class OnedcelltrackApp:
    def __init__(self, experiments, base_path):
        self.app = Flask(__name__)
        # Register simulations blueprint
        sim_app = SimulationsApp()
        self.app.register_blueprint(sim_app.blueprint)
        
        self.experiments = experiments
        self.base_path = base_path
        # For demonstration, pick the first experiment's ND2 file
        self.experiment = experiments[0]
        self.nd2_file = os.path.join(base_path, self.experiment, "timelapse.nd2")
        self.f = ND2Reader(self.nd2_file)
        self.frame = 0
        self.channel = 0
        self.fov = 0
        
        @self.app.route("/")
        def index():
            img = self.get_image(self.frame, self.channel, self.fov)
            img_str = self.numpy_to_b64(img)
            return render_template("experiment_viewer.html",
                                   image=img_str,
                                   max_frame=self.f.sizes['t'],
                                   max_channel=self.f.sizes['c'],
                                   max_fov=self.f.sizes['v'],
                                   experiments=self.experiments)
        
        @self.app.route("/update_image", methods=["POST"])
        def update_image():
            try:
                self.frame = int(request.form.get("frame", 0))
                self.channel = int(request.form.get("channel", 0))
                self.fov = int(request.form.get("fov", 0))
                contrast = request.form.get("contrast", "0,65535")
                contrast_vals = [int(x) for x in contrast.split(",")]
                img = self.get_image(self.frame, self.channel, self.fov, contrast_vals)
                img_str = self.numpy_to_b64(img)
                return jsonify({"image_data": img_str})
            except Exception as e:
                return Response(str(e), status=500)
        
    def get_image(self, frame, channel, fov, contrast=(0, 65535)):
        img = self.f.get_frame_2D(t=frame, c=channel, v=fov)
        vmin, vmax = contrast
        img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        img_uint8 = (img_norm * 255).astype("uint8")
        return img_uint8
    
    def numpy_to_b64(self, image):
        buffer = BytesIO()
        im = Image.fromarray(image)
        im.save(buffer, format="JPEG")
        buffer.seek(0)
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_b64

def main():
    base_path = "/path/to/experiments"  # Adjust this path as needed
    experiments = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    app_instance = OnedcelltrackApp(experiments, base_path)
    app_instance.app.run(debug=True, host="0.0.0.0", port=8899)

if __name__ == "__main__":
    main()
