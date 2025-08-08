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

Key observations:
- The loss curves demonstrate stable learning without overfitting
- Final training loss reached 62.0, validation loss stabilized at 58.0
- The gap between curves remains consistent (~4.0 difference)

For optimal performance, we recommend:
1. Monitoring loss trends beyond 25 epochs
2. Adjusting learning rate if plateaus occur
3. Verifying batch normalization layers

<img width="1920" height="1041" alt="Результат 2" src="https://github.com/user-attachments/assets/c0b7c212-6ddc-4f61-a4d4-0f6029b877cc" />

*Distance loss during train and val over 25 epochs*


### Key Metrics
| Metric         | Training | Validation |
|----------------|----------|------------|
| **Total Loss** | 64.0     | 62.0       |
| **Accuracy**   | 110px    | 104px      |
