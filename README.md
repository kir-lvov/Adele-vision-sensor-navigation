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
![Training Metrics](docs/training_metrics.png)  
*Loss and accuracy over 50 epochs*

### Key Metrics
| Epoch | Loss | Position Error (px) | Wind Adaptation Error |
|-------|------|---------------------|-----------------------|
| 10 | 0.87 | 112 | 0.23 |
| 25 | 0.45 | 98 | 0.15 |
| 50 | 0.21 | 108 | 0.12 |

---
