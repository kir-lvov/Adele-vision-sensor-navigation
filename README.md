# Adele-vision-sensor-navigation
<img width="619" height="660" alt="Результат работы нейросети" src="https://github.com/user-attachments/assets/0c5e85b7-dc55-4842-b01f-3cc3e66b20d0" />

*Example of drone path prediction with wind influence*

## 📌 Project Overview
A deep learning system that predicts optimal drone flight paths to avoid trees while accounting for wind conditions. Combines visual data from onboard cameras with wind sensor inputs.

## 🛠 Key Features
- **Multi-modal architecture**: Fuses visual (CNN) and sensor (Transformer) data
- **Real-time capable**: Processes 640x640 images at 30+ FPS on NVIDIA Jetson
- **Wind adaptation**: Adjusts path based on wind speed/direction (x/y vectors)
- **Sigmoid output**: Predicts normalized [0,1] avoidance coordinates

## 📊 Training Progress

<img width="1920" height="1038" alt="Результат 1" src="https://github.com/user-attachments/assets/d8d906c5-f570-4fac-9b6d-460e9a94c832" />

*Train and val loss over 25 epochs*

### Key Metrics
| Metric         | Training | Validation |
|----------------|----------|------------|
| **Total Loss** | 64.0     | 62.0       |
| **Accuracy**   | 110px    | 104px      |
