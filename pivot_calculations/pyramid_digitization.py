import cv2
import pyvista as pv
import numpy as np
import threading
import copy


# Function to load camera calibration parameters
def load_camera_calibration(camera_matrix_file, dist_coeffs_file):
    camera_matrix = np.load(camera_matrix_file, allow_pickle=True)
    dist_coeffs = np.load(dist_coeffs_file, allow_pickle=True)
    return camera_matrix, dist_coeffs

# Create a pyramid geometry in PyVista
def create_pyramid():
    # Define the points of the pyramid
    points = np.array([
        [0.5, 0.5, 0],  # Base corner 1
        [-0.5, 0.5, 0],  # Base corner 2
        [-0.5, -0.5, 0],  # Base corner 3
        [0.5, -0.5, 0],  # Base corner 4
        [0, 0, 1]  # Tip of the pyramid along the z-axis
    ])
    # Define the faces of the pyramid
    faces = np.hstack([[4, 0, 1, 2, 3],  # Base (4 points)
                       [3, 0, 1, 4],  # Side 1
                       [3, 1, 2, 4],  # Side 2
                       [3, 2, 3, 4],  # Side 3
                       [3, 3, 0, 4]])  # Side 4
    # Create a PolyData object
    pyramid = pv.PolyData(points, faces)
    return pyramid

saved_pyramids = []  # To store transformation matrices or pyramid copies

# Load the 3D pyramid model and make a copy for resetting
pyramid_model = create_pyramid()
original_pyramid_model = pyramid_model.copy()

# Create the PyVista plotter
plotter = pv.Plotter()
pyramid_actor = plotter.add_mesh(pyramid_model)
plotter.show_axes()

# Open the PyVista window in non-blocking mode
plotter.show(interactive_update=True, auto_close=False)

# Initialize ArUco marker detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

# Capture video from the camera
cap = cv2.VideoCapture(0)

# Load camera calibration data
camera_matrix, dist_coeffs = load_camera_calibration("cameraMatrix.pkl", "dist.pkl")

# Variables for threading and frame capture
running = True
frame = None

# Function to capture video frames in a separate thread
def capture_frames():
    global running, frame
    while running:
        ret, temp_frame = cap.read()
        if not ret:
            running = False
            break
        frame = temp_frame

# Start the video capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Frame counter to control update frequency
frame_counter = 0

while running:
    if frame is None:
        continue  # Wait until the first frame is captured

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for corner in corners:
            # Reset the pyramid model to its original state
            pyramid_model.copy_from(original_pyramid_model)

            # Estimate pose for each detected marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)

            # Convert rvec to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec[0])

            # Create the transformation matrix (4x4)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec[0].flatten()

            # Apply the transformation to the pyramid model
            pyramid_model.transform(transformation_matrix)

    # Update the PyVista plot periodically
    #if frame_counter % 5 == 0:  # Update every 5 frames
    plotter.update()

    # Show the frame using OpenCV
    cv2.imshow('ArUco Tracker', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        running = False
        
    elif key == ord('d'):
        if ids is not None and len(ids) > 0:
            # Create a static copy of the transformed pyramid
            static_pyramid = pyramid_model.copy()
            plotter.add_mesh(static_pyramid, color='orange')  # Optional: different color to distinguish
            saved_pyramids.append(static_pyramid)
            print("Pyramid position saved and copy placed in scene.")

    frame_counter += 1

# Stop the video capture thread and release resources
capture_thread.join()
cap.release()
cv2.destroyAllWindows()
plotter.close()
