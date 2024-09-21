import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime


# Constants
PIXELS_PER_CM_DEFAULT = 20.0  # Default pixels per cm (can be calibrated)
THRESH_VALUE = 30
KERNEL = np.ones((7, 7), np.uint8)
TEXT_POSITION = (20, 20)
LOG_FILE = 'log.txt'
KNOWN_WIDTH_CM = 5.0
KNOWN_HEIGHT_CM = 5.0
CALIBRATION_IMG = 'imge_project/res/calibrating_img.JPG'
VIDEO_FILE = 'imge_project/res/test_footage_4.mp4'
viscosity = 0.01

# Ensure log file exists
def initialize_log_file(log_file):
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create and initialize the log file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            header = f"{'Fluid ID':<30} | {'Date-Time':<20} | {'Avg Velocity (m/s)':<20} | {'Avg Shear Stress (Pa)':<20}\n"
            f.write(header)
            f.write('-' * len(header) + '\n')


# Generate unique fluid ID
def generate_fluid_id():
    return f"fluid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Calibration part
def calibrate_pixels_per_cm():
    image = cv2.imread(CALIBRATION_IMG)
    drawing = False
    ix, iy = -1, -1
    pixels_per_cm_x = PIXELS_PER_CM_DEFAULT
    pixels_per_cm_y = PIXELS_PER_CM_DEFAULT

    def draw(event, x, y, flags, param):
        nonlocal ix, iy, drawing, pixels_per_cm_x, pixels_per_cm_y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            pixels_per_cm_x, pixels_per_cm_y = calculate_conversion_factor(ix, iy, x, y)

    def calculate_conversion_factor(ix, iy, x, y):
        width_pixels = abs(x - ix)
        height_pixels = abs(y - iy)
        return width_pixels / KNOWN_WIDTH_CM, height_pixels / KNOWN_HEIGHT_CM

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()
    # Return the calculated pixels per cm values
    return pixels_per_cm_x, pixels_per_cm_y


# Base video processing function
def process_video(pixels_per_cm_x, pixels_per_cm_y):
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)

    # window_name = "Frame"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # # Set the window property to stay on top
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


    ret, first_frame = cap.read()
    if not ret:
        print("Error: Unable to read the video file.")
        return

    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    pos = [[0, 0]]
    velocity = [[0, 0]]
    velocity_magnitude = []
    i = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_sub = cv2.absdiff(frame_gray, first_frame_gray)
        _, frame_bin = cv2.threshold(frame_sub, THRESH_VALUE, 255, cv2.THRESH_BINARY)
        frame_bin = cv2.morphologyEx(frame_bin, cv2.MORPH_OPEN, KERNEL)

        contours, _ = cv2.findContours(frame_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        process_frame(frame, contours, pos, velocity, velocity_magnitude, i, fps, pixels_per_cm_x, pixels_per_cm_y)
        cv2.imshow('Frame', frame)
        i += 1

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return pos, velocity_magnitude, frame_width, frame_height


# Process individual frames and calculate velocity
def process_frame(frame, contours, pos, velocity, velocity_magnitude, i, fps, pixels_per_cm_x, pixels_per_cm_y):
    if contours:
        contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            pos.append([cx, cy])

            if i != 0:
                Vx_pixels = (pos[i][0] - pos[i - 1][0]) * fps
                Vy_pixels = (pos[i][1] - pos[i - 1][1]) * fps
                Vx_meters = Vx_pixels / pixels_per_cm_x / 100
                Vy_meters = Vy_pixels / pixels_per_cm_y / 100
                velocity.append([Vx_meters, Vy_meters])
                magnitude = np.sqrt(Vx_meters**2 + Vy_meters**2)
                velocity_magnitude.append(magnitude)
                display_text(frame, f"Velocity: {magnitude:.2f} m/s")
            else:
                velocity.append([0, 0])
                velocity_magnitude.append(0)
        draw_contour(frame, contour, cx, cy)
    else:
        pos.append([0, 0])


# Display velocity on frame
def display_text(frame, text):
    cv2.putText(frame, text, TEXT_POSITION, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Draw contour and centroid on frame
def draw_contour(frame, contour, cx, cy):
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
    cv2.putText(frame, "centroid", (cx - 25, cy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Plot particle paths and individual traces
def plot_paths(pos, velocity_magnitude, average_velocity, frame_width, frame_height, sub_arrays):
    plt.figure(figsize=(10, 5))

    # Particle Paths Plot
    plt.subplot(1, 2, 1)
    plot_particle_paths(pos, velocity_magnitude, average_velocity, frame_width, frame_height)

    # Individual Traces Plot
    plt.subplot(1, 2, 2)
    plot_individual_traces(sub_arrays, frame_width, frame_height)

    plt.savefig("plots")
    plt.show()


# Particle paths with velocity color coding
def plot_particle_paths(pos, velocity_magnitude, average_velocity, frame_width, frame_height):
    valid_positions = [p for p in pos if p != [0, 0]]
    max_deviation = max(abs(mag - average_velocity) for mag in velocity_magnitude)

    for idx, (x, y) in enumerate(valid_positions):
        magnitude = velocity_magnitude[idx] if idx < len(velocity_magnitude) else 0
        deviation = magnitude - average_velocity
        intensity = min(abs(deviation) / max_deviation, 1.0)
        color = (intensity, 0, 0) if deviation > 0 else (0, 0, intensity)
        plt.plot(x, y, 'o', color=color)

    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Particle Paths with Velocity Color Coding")


# Individual traces from sub-arrays
def plot_individual_traces(sub_arrays, frame_width, frame_height):
    valid_points = {k: v for k, v in sub_arrays.items() if v and all(isinstance(item, list) for item in v)}
    for key, point_sets in valid_points.items():
        x, y = zip(*point_sets)
        plt.plot(x, y, label=f"{key + 1}", color=(random.random(), random.random(), random.random()))

    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Individual Paths")
    plt.legend()


# Log velocity data
def log_velocity_data_shear_stress(log_file, fluid_id, average_velocity, average_shear_stress):
    # Log the data in an elegant, aligned format
    with open(log_file, 'a') as log:
        log_entry = f"{fluid_id:<30} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} | {average_velocity:<20.2f} | {average_shear_stress:<20.2f}\n"
        log.write(log_entry)


# calculate shear stress
def calculate_shear_stress(velocity_magnitude, pixels_per_cm_y, viscosity):
    shear_stress = []
    for i in range(1, len(velocity_magnitude)):
        # Approximate velocity gradient (du/dy)
        du_dy = (velocity_magnitude[i] - velocity_magnitude[i - 1]) / (1 / pixels_per_cm_y)  # Change in velocity over distance
        shear = viscosity * du_dy
        shear_stress.append(shear)
    return np.mean(shear_stress)




# Main function
def main():
    initialize_log_file(LOG_FILE)
    fluid_id = generate_fluid_id()

    # Calibrate if needed
    if input("Do you want to perform calibration? (y/n): ").lower() == 'y':
        pixels_per_cm_x, pixels_per_cm_y = calibrate_pixels_per_cm()
    else:
        pixels_per_cm_x, pixels_per_cm_y = PIXELS_PER_CM_DEFAULT, PIXELS_PER_CM_DEFAULT

    #visocity inputable
    viscosity = float(input("Enter the fluid's dynamic viscosity (in Pa·s): "))

    pos, velocity_magnitude, frame_width, frame_height = process_video(pixels_per_cm_x, pixels_per_cm_y)
    
    
    # Calculate average velocity
    average_velocity = np.mean(velocity_magnitude)
    print(f"Average Velocity: {average_velocity:.2f} m/s")
    
    # calculate average shear stress
    average_shear_stress = calculate_shear_stress(velocity_magnitude, pixels_per_cm_y, viscosity)
    print(f"Average Shear Stress: {average_shear_stress:.2f} Pa")
    
    

    log_velocity_data_shear_stress(LOG_FILE, fluid_id, average_velocity,average_shear_stress)


    sub_arrays = {}
    current_sub_array = []
    for element in pos:
        if element == [0, 0]:
            if current_sub_array:
                sub_arrays[len(sub_arrays)] = current_sub_array
                current_sub_array = []
        else:
            current_sub_array.append(element)
    if current_sub_array:
        sub_arrays[len(sub_arrays)] = current_sub_array

    plot_paths(pos, velocity_magnitude, average_velocity, frame_width, frame_height, sub_arrays)


if __name__ == "__main__":
    main()
