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
- Final training loss reached 64.0, validation loss stabilized at 62.0
- The gap between curves remains consistent (~4.0 difference)

For optimal performance, we recommend:
1. Monitoring loss trends beyond 25 epochs
2. Adjusting learning rate if plateaus occur
3. Verifying batch normalization layers

<img width="1920" height="1041" alt="Результат 2" src="https://github.com/user-attachments/assets/c0b7c212-6ddc-4f61-a4d4-0f6029b877cc" />

*Distance loss during train and val over 25 epochs*

- Training loss decreased from 240px → 110px  
- Validation loss stabilized at 105px (±2px variance)  
- Consistent ~5px gap indicates proper regularization

### Key Metrics
| Metric         | Training | Validation |
|----------------|----------|------------|
| **Total Loss** | 64.0     | 62.0       |
| **Accuracy**   | 110px    | 105px      |

### 1. Installation
```bash
git clone https://github.com/kir-lvov/Adele-vision-sensor-navigation.git
cd Adele-vision-sensor-navigation
```

### 2. Running via Docker
```bash
# Building an image
docker build -t adele-drone .

# Startup (CPU)
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/weights:/app/weights \
  adele-drone

# For GPU
docker run -it --rm --gpus all adele-drone
```

## 📦 Project structure
```
/data          # Datesets (not included in the repository)
/weights       # Models (use Git LFS)
/src           # Source code
Dockerfile     # Container configuration
```

## 🛠 Requirements
- Docker 20+
- NVIDIA Docker (for GPU)

## 🌳 Dataset Specification

### Required JSON Format (Example)
```json
{
  "image": "example_frame.jpg",  // Filename must match exactly
  "annotations": [
    {
      "coordinates": {
        "x": 100,      // Top-left X in pixels (absolute)
        "y": 150,      // Top-left Y in pixels
        "width": 45,   // Bounding box width
        "height": 60   // Bounding box height
      }
    }
  ]
}
