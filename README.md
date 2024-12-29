# Exploration_Project_DAU_2024

## Overview
**Exploration_Project_DAU_2024** implements **real-time ArUco marker detection** to control **3D model movement** based on marker motion. It combines **OpenCV** for marker detection and **PyVista** for 3D visualization.

## Features
- **Real-Time Marker Detection** - Tracks ArUco markers using a webcam.
- **3D Model Control** - Manipulates 3D objects in response to marker movements.
- **Camera Calibration** - Ensures precise pose estimation using pre-calibrated camera parameters.
- **Interactive 3D Visualization** - Displays the motion and transformation of 3D models.
- **Multiple Object Support** - Controls multiple 3D models mapped to unique markers.

### Key Files:
1. **markerNBone.py** - Main code handling camera setup, marker detection, and 3D model rendering.
2. **cameraCalibration.py** - Script for calibrating the camera and generating calibration data.
3. **aruco_marker.png** - Pre-generated ArUco marker image.
4. **cameraMatrix.pkl & dist.pkl** - Calibration data files for camera matrix and distortion coefficients.
5. **config.json** - Configuration settings for the project.

## Dependencies
Ensure the following libraries are installed:
```bash
pip install opencv-python
pip install numpy
pip install pyvista
```

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Exploration_Project_DAU_2024.git
```
2. Navigate to the project folder:
```bash
cd Exploration_Project_DAU_2024/ExplorationProject
```
3. Run the main script:
```bash
python markerNBone.py
```
4. Press **'q'** to exit the visualization.

## Configuration
- Camera calibration data should be generated using **cameraCalibration.py** if not already available.
- Update **config.json** if any parameter tuning is required.

## Demonstration
1. Place the **aruco_marker.png** in front of the webcam.
2. Observe real-time transformation of the associated 3D model.
3. Test with different markers to control multiple objects.

## Future Enhancements
- Support for dynamic marker addition and removal.
- Advanced marker filtering to reduce noise.
- Integration with AR frameworks for augmented reality applications.

## Contact
For any queries or suggestions, please contact https://github.com/Anushka019.

