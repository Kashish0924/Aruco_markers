import json
import threading
import numpy as np
import cv2
import pyvista as pv

# === Load Configuration from JSON File ===
with open("config.json", "r") as f:
    config = json.load(f)


# === Camera Class for Managing Camera Operations ===
class Camera:

    # Constructor
    def __init__(self, camera_matrix_file, dist_coeffs_file, capture_device_index=0):
        # Load camera calibration parameters
        self.camera_matrix, self.dist_coeffs = self.load_camera_calibration(camera_matrix_file, dist_coeffs_file)
         
        # Initialize video capture
        self.cap = cv2.VideoCapture(capture_device_index)
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")
        
        self.running = True
        self.frame = None

    # Load camera calibration parameters
    def load_camera_calibration(self, camera_matrix_file, dist_coeffs_file):
        camera_matrix = np.load(camera_matrix_file, allow_pickle=True)
        dist_coeffs = np.load(dist_coeffs_file, allow_pickle=True)
        return camera_matrix, dist_coeffs

    # Capture a frame from the camera
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.running = False
        return frame

    # Separate threads for capturing frames
    def start_capture(self):
        def capture_frames():
            while self.running:
                self.frame = self.capture_frame()

        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.start()

    # Stop capturing frames
    def stop_capture(self):
        self.running = False
        self.cap.release()

    # Destructor
    def _del_(self):
        self.stop_capture()


# === Generate the ArUco Marker ===
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco"]["dictionary"]))
aruco_marker = cv2.aruco.generateImageMarker(
    aruco_dict, config["aruco"]["marker"]["id"], config["aruco"]["marker"]["size_px"]
)
aruco_marker_rgb = cv2.cvtColor(aruco_marker, cv2.COLOR_GRAY2RGB)
cv2.imwrite(config["aruco"]["marker"]["output_file"], aruco_marker_rgb)


# === Load the Image as a Texture in PyVista ===
texture_0 = pv.read_texture(config["marker_texture_map"]["0"])
texture_1 = pv.read_texture(config["marker_texture_map"]["1"])


# === Create a Thin Cuboid Geometry in PyVista ===
def create_cuboid():
    dims = config["cuboid"]["dimensions"]
    width, depth, height = dims["width"], dims["depth"], dims["height"]
    points = np.array([
        [width/2, depth/2, -height/2],
        [-width/2, depth/2, -height/2],
        [-width/2, -depth/2, -height/2],
        [width/2, -depth/2, -height/2],
        [width/2, depth/2, height/2],
        [-width/2, depth/2, height/2],
        [-width/2, -depth/2, height/2],
        [width/2, -depth/2, height/2]
    ])
    faces = np.hstack([
        [4, 0, 1, 2, 3],
        [4, 4, 5, 6, 7],
        [4, 0, 3, 7, 4],
        [4, 1, 2, 6, 5],
        [4, 0, 1, 5, 4],
        [4, 3, 2, 6, 7]
    ])
    cuboid = pv.PolyData(points, faces)
    cuboid.texture_map_to_plane(
        origin=(-width / 2, -depth / 2, height / 2),
        point_u=(width / 2, -depth / 2, height / 2),
        point_v=(-width / 2, depth / 2, height / 2),
        inplace=True
    )    
    return cuboid


# === Create and Load 3D Models ===
cuboid_model_1 = create_cuboid()
cuboid_model_2 = create_cuboid()
bone_model = pv.read(config["models"]["bone_model"]["file"])
bone_model.scale(config["models"]["bone_model"]["scale_factor"], inplace=True)
bone_model.rotate_z(config["models"]["bone_model"]["initial_transformations"]["rotate_z"], inplace=True)
bone_model.rotate_x(config["models"]["bone_model"]["initial_transformations"]["rotate_x"], inplace=True)

# === PyVista Plotter ===
plotter = pv.Plotter()
plotter.add_mesh(cuboid_model_1, texture=texture_0)
plotter.add_mesh(cuboid_model_2, texture=texture_1)
plotter.add_mesh(bone_model)
plotter.show_axes()
plotter.show(**config["display"]["plotter_settings"])


# === Initialize ArUco Detector and Camera ===
parameters = cv2.aruco.DetectorParameters()
camera = Camera(
    config["camera"]["calibration_files"]["camera_matrix"],
    config["camera"]["calibration_files"]["dist_coeffs"],
    config["camera"]["capture_device_index"]
)
camera.start_capture()


# === Marker to Model Mapping ===
marker_model_map = {int(k): v for k, v in config["marker_model_map"].items()}
original_model_map = {int(k): eval(f"{v}.copy()") for k, v in config["marker_model_map"].items()}


# === Process Frame ===
# === Process Frame ===
def process_frame(frame, camera_matrix, dist_coeffs, marker_model_map, original_model_map):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_positions = {}  # Dictionary to store positions of markers (tvec)

    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id in marker_model_map:
                model = eval(marker_model_map[marker_id])
                original_model = original_model_map[marker_id].copy()  # .copy() to create a fresh copy of the model
                model.copy_from(original_model)

                # Estimate pose (rotation and translation) for the marker
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, config["transformation"]["pose_estimation"]["marker_length_meters"], camera_matrix, dist_coeffs)
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                tvec_scaled = tvec[0] * config["transformation"]["pose_estimation"]["scale_factor"]
                transformation_matrix[:3, 3] = tvec_scaled.flatten()

                # Save the translation (position) of the marker
                marker_positions[marker_id] = tvec_scaled.flatten()

                # Apply transformation to the model
                model.transform(transformation_matrix)

        # If both marker 0 and marker 1 are detected, calculate relative position
        if 0 in marker_positions and 1 in marker_positions:
            pos_marker_0 = marker_positions[0]
            pos_marker_1 = marker_positions[1]
            relative_position = pos_marker_1 - pos_marker_0
            x_diff, y_diff, z_diff = relative_position*100
            # print(f"Relative position of marker 1 with respect to marker 0:")
            # print(f"  x: {x_diff:.3f} cm")
            # print(f"  y: {y_diff:.3f} cm")
            # print(f"  z: {z_diff:.3f} cm")

            # Calculate the Euclidean distance (real-life distance) between the two markers
            distance = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

            print(f"  Real-life distance between marker 1 and marker 0: {distance:.3f} cm")

        plotter.update()



# === Main Loop ===
while camera.running:
    if camera.frame is None:
        continue
    frame = camera.frame
    process_frame(frame, camera.camera_matrix, camera.dist_coeffs, marker_model_map, original_model_map)
    cv2.namedWindow(config["display"]["opencv_window"]["title"], cv2.WINDOW_NORMAL)
    cv2.imshow(config["display"]["opencv_window"]["title"], frame)
    if cv2.waitKey(1) & 0xFF == ord(config["display"]["opencv_window"]["exit_key"]):
        camera.stop_capture()

cv2.destroyAllWindows()
plotter.close()
