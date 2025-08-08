# Adele-vision-sensor-navigation
![Demo Visualization](docs/demo.gif)  
*Example of drone path prediction with wind influence*

## 📌 Project Overview
A deep learning system that predicts optimal drone flight paths to avoid trees while accounting for wind conditions. Combines visual data from onboard cameras with wind sensor inputs.

## 🛠 Key Features
- **Multi-modal architecture**: Fuses visual (CNN) and sensor (Transformer) data
- **Real-time capable**: Processes 640x640 images at 30+ FPS on NVIDIA Jetson
- **Wind adaptation**: Adjusts path based on wind speed/direction (x/y vectors)
- **Sigmoid output**: Predicts normalized [0,1] avoidance coordinates
