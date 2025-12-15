"""
Created by Mathieu Ribeaud

Dependencies required :
- matplotlib
- OpenCV
- ffmpeg

IMPORTANT : install them on the OFM server python environment !
"""

"""
DISCLAIMER : there is a deep analysis option, which makes an analysis of individual cell movements 
(in addition of the diff image analysis). This is an experimental method that hasn't been statistically tested,
and which only works with bakers yeast (in the actual state). That's only a proof of concept that seems to work!
"""

"""
If DEBUG = True, the program will run locally and execute only the video processing part

You may need to set analysis option (search for "DEBUG = True" in this file)

Have to be set to False to run on the microscope
"""
DEBUG = False

if not DEBUG:
    import logging
    from labthings import fields, find_component
    from labthings.extensions import BaseExtension
    from labthings.views import ActionView
    from openflexure_microscope.api.utilities.gui import build_gui
    from openflexure_microscope.captures.capture_manager import generate_basename
    from openflexure_microscope.api.default_extensions.autofocus import AutofocusExtension
    from openflexure_microscope.paths import data_file_path


import ffmpeg
import cv2
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import copy
import os
import time

# Global variables about microscope specs
OPTIC_MAGNIFICATION = 40 # 40 if magnification is 40x,...
OPTIC_CORR = OPTIC_MAGNIFICATION/40 # Correction of calculations depending on the optic magnification

# Global variables corresponding to video processing parameters
BLOCKSIZE = 41
C = 2
SCALE_FACTOR = 200

# Global variables corresponding to base extension options
# Set analysis options here if DEBUG = True
ANALYSIS = True
CELL_TYPE = "Universal" # Can also be "Bakers yeast" 
DEEP_ANALYSIS = False # /!\ This works only with Bakers yeast /!\  
MIN_CELL_THRESHOLD = 1
DEFINITION = "1080p"
MP4 = False

# Global variables about paths and folder/file names
BASE_FOLDER = "/var/openflexure/data/micrographs/"
COUNT_CELL_FOLDER = "Count cells"
ANALYSIS_FOLDER = "Standard processing"
DEEP_ANALYSIS_FOLDER = "Deep processing"
CELL_COUNT_VIDEO_FILENAME = "cell_count_video"

def chmod_recursive(path, mode):
    """
    Changes the permission of a folder and all of its subfolders and files
    
    Parameters:
        path: path of the folder
        mode: permission to set
    """
    
    # Change the permission of the directory itself
    os.chmod(path, mode)
    
    # Parse all subfolders and files to change permission
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            os.chmod(os.path.join(root, dir), mode)
        for file in files:
            os.chmod(os.path.join(root, file), mode)

def get_contours(picture, folder):
    """
    Analyze picture to detect cells
    
    Parameters:
        picture: frame to analyze
        folder: folder in which the program "actually is" (used to determine if the picture should be saved or not)
    """
    
    # Create a copy of the image to avoid modifying the original one
    cell_count_pic = copy.deepcopy(picture)
    
    # Convert the frame to grayscale, apply an adaptive threshold and inverse the frame
    cell_count_pic = cv2.cvtColor(cell_count_pic, cv2.COLOR_BGR2GRAY)
    cell_count_pic = cv2.adaptiveThreshold(cell_count_pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BLOCKSIZE, C)
    cell_count_pic = cv2.bitwise_not(cell_count_pic)
    
    # Save the treated image use to count cells (folder will contains COUNT_CELL_FOLDER only when analyzing the first frame)
    # In other words, save only the treated image used as reference to count cells
    if COUNT_CELL_FOLDER in folder:
        cell_count_path = os.path.join(folder, "treated_cell_count_picture.png")
        log("Treated cell count picture path: {}".format(cell_count_path))
        cv2.imwrite(cell_count_path, cell_count_pic)
        
    # Detect all contours
    contours, _ = cv2.findContours(cell_count_pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, cell_count_pic

def get_center(contour):
    """
    Compute the center of a contour
    
    Parameters:
        contour: the contour
        
    Returns:
        The center
    """
    
    M = cv2.moments(contour)
    if M["m00"] != 0:
        X = int(M["m10"] / M["m00"])
        Y = int(M["m01"] / M["m00"])
    else:
        X = 0
        Y = 0
    
    return [X, Y]

def def_count_cells(folder):
    """
    Counts the number of cells detected in the reference frame (10th frame)
    
    Parameters:
        folder: folder in which the program "actually is" (used to determine in get_contours() if the image used for cell detection should be saved or not)
    
    Returns:
        Number of cells detected
    """
    
    # Open video
    video = cv2.VideoCapture(os.path.join(folder, CELL_COUNT_VIDEO_FILENAME + ".h264"))

    # Check if video opened successfully
    if not video.isOpened():
        log("Error: Could not open video.")
        return

    # Read the 10th frame (9 first ones are of bad quality)
    for i in range(10):
        ret, picture = video.read()
                        
    # Release the video capture object
    video.release()
    
    # Save the frame use to count cells
    original_cell_count_path = os.path.join(folder, "cell_count_picture_original.png")
    log("Cell count picture path: {}".format(original_cell_count_path))
    cv2.imwrite(original_cell_count_path, picture)
    
    # Get contours (detect cells)
    contours, picture_copy = get_contours(picture, folder)
    
    #  Calculate min and max areas with margins depending on the type of cell analyzed
    if CELL_TYPE == "Bakers yeast":   
        area_min = 3.14 * ((2.5 / 2 * OPTIC_CORR) ** 2) * SCALE_FACTOR * 0.9
        area_max = 3.14 * ((5 / 2 * OPTIC_CORR) ** 2) * SCALE_FACTOR * 1.1
    elif CELL_TYPE == "Universal": 
        area_min = 3.14 * ((1 / 2 * OPTIC_CORR) ** 2) * SCALE_FACTOR
        area_max = 3.14 * ((10 / 2 * OPTIC_CORR) ** 2) * SCALE_FACTOR
    
    cells = [[],[]]
    
    # Check each contour and determine if it's a cell
    for contour in contours:  
        match_criterias = False
        if len(contour) >= 5:
            # Filter by area
            area = cv2.contourArea(contour)
            if area_min < area < area_max:
                # Filter by circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if CELL_TYPE == "Bakers yeast" and circularity > 0.77:
                    match_criterias = True
                elif (CELL_TYPE == "Universal") and 0.2 < circularity:
                    match_criterias = True
                    
                if match_criterias:
                    cells[0].append(contour)
                    cells[1].append(area)
            
    cell_count = len(cells[0])
    log(str(cell_count) + " cells found")

    # Compute and saved center of each cell
    cells_with_centers = []
    i = 0
    for cell in cells[0]:
        cells_with_centers.append([i, [get_center(cell)], [0, 0], 0, cells[1][i]])
        i = i + 1
                    
    # Save all data in a txt as JSON
    results_path = os.path.join(folder, "cells_with_center.txt")
    with open(results_path, 'w') as file:
        file.write(json.dumps(jsonify(cells_with_centers), indent=4))
    log("Cells with center saved as JSON here: {}".format(results_path))

    cell_count_picture = cv2.drawContours(picture, cells[0], -1, (255, 0, 0), 2)
    
    # Save the frame use to count cells with cells highlighted
    cell_count_path = os.path.join(folder, "cell_count_picture.png")
    log("Cell count picture path: {}".format(cell_count_path))
    cv2.imwrite(cell_count_path, cell_count_picture)
    
    return cell_count

def analyze_movements(picture, cells, folder):
    """
    Analyzes displacement of each cell compared to the precedent frames
    
    Parameters:
        picture: frame to analyze
        cells: reference cells (the ones found in the first frame) which contains all the displacement analyze already done
        folder: folder in which the program "actually is" (used to determine in get_contours() if the image used for cell detection should be saved or not)
    """
    
    # Analyze picture to detect cells
    contours, picture_copy = get_contours(picture, folder) 
    
    # Parse every cell found in the current frame
    for contour in contours:
        # Get the XY coordinates of the center of the cell
        center = get_center(contour)
        
        # Parse every reference cells (the ones found in the first frame) to find the corresponding reference cell
        for cell in cells:
            # Get XY position of the reference cell in the last frame where the cell has been detected (should be the last frame analyzed)
            # cell[1] contains all centers coordinates of precedent frames where the cell has been detected 
            # cell[1][n] contains center for frame n (here the last frame where the cell has been detected) 
            # cell[1][n][0] contains x coordinate of center for frame n
            # cell[1][n][1] contains y coordinate of center for frame n
            X_cell = cell[1][len(cell[1]) - 1][0]
            Y_cell = cell[1][len(cell[1]) - 1][1]
            
            # Adds computed cell displacement only if it's not bigger than 5px (in X and Y), which means that
            # the current reference cell corresponds to the current cell found in the current frame
            if (X_cell - 5) < center[0] < (X_cell + 5) and (Y_cell - 5) < center[1] < (Y_cell + 5):
                cell[1].append(center)
                # cell[2][0] contains the accumulated x displacement of the cell
                # cell[2][1] contains the accumulated y displacement of the cell
                cell[2][0] = cell[2][0] + abs(X_cell - center[0])
                cell[2][1] = cell[2][1] + abs(Y_cell - center[1])

def log(text):
    if DEBUG:
        print(text) 
    else:
        logging.info(text)

def jsonify(data):
    """
    Recursively converts numpy types in a given data structure to native Python types,
    making it JSON serializable.
    
    Parameters:
        data: The input data (e.g., list, dict, numpy array, etc.) to be converted.
    
    Returns:
        The data with all numpy types converted to their native Python equivalents.
    """
    
    if isinstance(data, dict):
        return {key: jsonify(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [jsonify(element) for element in data]
    elif isinstance(data, tuple):
        return tuple(jsonify(element) for element in data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data

def change_def(microscope):
    """
    Set the microscope's camera stream resolution to a defined one
    
    Parameters:
        microscope: microscope component, necessary to control the microscope 
        
    Returns:    
        old_stream_resolution: old stream resolution (before changing it)
    """
    
    # Stop the stream during acquisition. Important for high framerates due to bandwith limitations
    try:
        microscope.camera.stop_stream()
    except:
        log("Stream already stopped")
            
    # Set resolution
    old_stream_resolution = microscope.camera.stream_resolution
    if DEFINITION == "1080p":
        microscope.camera.stream_resolution = (1920,1080)
        microscope.camera.picamera.resolution = (1920,1080)
    else:
        microscope.camera.stream_resolution = (640,480) 
        microscope.camera.picamera.resolution = (640,480)
        
    log("Definition set to {}".format(DEFINITION))
        
    return old_stream_resolution

def reset_def(microscope, old_stream_resolution):
    """
    Reset the microscope's camera stream resolution to the old one
    
    Parameters:
        microscope: microscope component, necessary to control the microscope 
        old_stream_resolution: old stream resolution
    """
    
    # Reset the camera properties
    microscope.camera.stream_resolution = old_stream_resolution
    microscope.camera.picamera.resolution = old_stream_resolution
    microscope.camera.start_stream()
    log("Definition reset")

def full_comparison(folder_path):
    """
    Compare all results of subfolders and put them in a plot. 
    Create additional plots if there is deep processing data and/or "Alive" or "Dead" in at least one subfolder name.
    
    Parameters:
        folder_path: path of the folder 
    """
    
    all_data = [[],[]]
    versus = [[],[]]
    deep_processing_data = [[],[]]
    deep_processing_versus = [[],[]]
    
    for root, subfolders, files in os.walk(folder_path):
        subfolders.sort()
        for subfolder in subfolders:
            for root, subfolders, files in os.walk(os.path.join(folder_path, subfolder)):
                for file in files:
                    if file == "results.txt":
                        with open(os.path.join(folder_path, subfolder, "results.txt"), 'r') as file_opened:
                            data = json.load(file_opened)
                            for name in data[0]:
                                all_data[0].append("{}_{}".format(subfolder, name))
                            for value in data[3]:
                                all_data[1].append(value)
                            
                            if "alive" in subfolder:
                                versus[0].append("Alive")
                                versus[1].append(np.nanmean(data[3]))
                            elif "dead" in subfolder:
                                versus[0].append("Dead")
                                versus[1].append(np.nanmean(data[3]))
                                
                    elif file == "deep_processing_results.txt":
                        with open(os.path.join(folder_path, subfolder, "deep_processing_results.txt"), 'r') as file_opened:
                            data = json.load(file_opened)
                            for name in data[0]:
                                deep_processing_data[0].append("{}_{}".format(subfolder, name))
                            for value in data[1]:
                                deep_processing_data[1].append(value)
                            
                            if "alive" in subfolder:
                                deep_processing_versus[0].append("Alive")
                                deep_processing_versus[1].append(np.nanmean(data[1]))
                            elif "dead" in subfolder:
                                deep_processing_versus[0].append("Dead")
                                deep_processing_versus[1].append(np.nanmean(data[1]))
                                
    
    # Create a plot to compare all data of standard analysis
    plt.figure()    
    plt.bar(all_data[0], all_data[1])
    plt.xlabel("Video")
    plt.xticks(rotation=90)
    plt.ylabel("Number of changed pixels")
    plt.title("Comparison of all data")
    plt.tight_layout()
    comparison_plot_path = os.path.join(folder_path, "Full_comparison.png")
    plt.savefig(comparison_plot_path)
    plt.close()
    log("Full comparison plot path: {}".format(comparison_plot_path))

    # If at least one sufolder name contains "Alive" or "Dead", create an additional plot to compare alive versus dead cells of standard analysis
    if len(versus[0]) > 0:
        plt.figure() 
        plt.bar(versus[0], versus[1])
        plt.xlabel("Type")
        plt.xticks(rotation=90)
        plt.ylabel("Number of changed pixels")
        plt.title("Comparison of dead vs alive cells")
        plt.tight_layout()
        versus_comparison_plot_path = os.path.join(folder_path, "Versus_comparison.png")
        plt.savefig(versus_comparison_plot_path)
        plt.close()
        log("Versus comparison plot path: {}".format(versus_comparison_plot_path))
    
    # If there is deep processing data, create an additional plot to compare all data of deep processing 
    if len(deep_processing_data[0]) > 0:
        plt.figure()    
        plt.bar(deep_processing_data[0], deep_processing_data[1])
        plt.xlabel("Video")
        plt.xticks(rotation=90)
        plt.ylabel("Mean of individual cell movements")
        plt.title("Comparison of all data (deep processing)")
        plt.tight_layout()
        deep_comparison_plot_path = os.path.join(folder_path, "Full_deep_comparison.png")
        plt.savefig(deep_comparison_plot_path)
        plt.close()
        log("Full deep comparison plot path: {}".format(deep_comparison_plot_path))

        # If there is deep processing data and at least one sufolder name contains "Alive" or "Dead", create an additional plot to compare alive versus dead cells with deep processing
        if len(deep_processing_versus[0]) > 0:
            plt.figure() 
            plt.bar(deep_processing_versus[0], deep_processing_versus[1])
            plt.xlabel("Type")
            plt.xticks(rotation=90)
            plt.ylabel("Mean of individual cell movements")
            plt.title("Comparison of dead vs alive cells (deep processing)")
            plt.tight_layout()
            deep_versus_comparison_plot_path = os.path.join(folder_path, "Versus_deep_comparison.png")
            plt.savefig(deep_versus_comparison_plot_path)
            plt.close()
            log("Versus deep comparison plot path: {}".format(deep_versus_comparison_plot_path)) 

def process_video(folder_path):
    """
    Perform video processing of the video in the folder.
    
    Parameters:
        folder_path: path of the folder

    Returns:
        total_diff: total difference value (indicator of all cells nanomotion)
        movement_mean: mean of individual cells movements (only if DEEP_ANALYSIS=True, if False this value is set to 0)
        total_area: total area of detected cells    
    """
    
    # Get path of the video file
    video_path = os.path.join(folder_path, "video.h264")
    log("Analysis of next video...")
    log("Video path: {}".format(video_path))
       
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        log("Error: Could not open video.")
        return

    # Read the first frame
    ret, prev_frame = video.read()

    # Check if frame is read correctly
    if not ret:
        log("Error reading first frame")
        return
    
    # Skip the 9 following frames (those first frames are of lowest quality)
    for i in range(9):
        ret, prev_frame = video.read()

    # Convert the frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize the accumulated difference image
    accumulated_diff = np.zeros_like(prev_gray, dtype=np.uint16)
    
    # Get the cells previously detected
    with open(os.path.join(folder_path, COUNT_CELL_FOLDER, "cells_with_center.txt"), 'r') as file_opened:
        cells_with_centers = json.load(file_opened)
    
    # Calculate total area of detected cells
    total_area = 0
    for cell in cells_with_centers:
        total_area = total_area + cell[4]

    # Initialization of variables
    diffs = []
    illumination = []
    diff_count = 0
    
    # Read the next frame
    ret, frame = video.read()
    while ret:
        diff_count += 1
        
        if DEEP_ANALYSIS and CELL_TYPE == "Bakers yeast":
            # Analyze individual cell movements
            analyze_movements(frame, cells_with_centers, folder_path)
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        illumination.append(np.sum(gray))
        
        # Compute the absolute difference between the current frame and the previous frame
        diff = cv2.absdiff(prev_gray, gray)
        
        # Store the number of pixels that have changed
        diffs.append(np.sum(diff))

        # Accumulate the difference
        accumulated_diff = cv2.add(accumulated_diff, diff, dst=accumulated_diff, dtype=cv2.CV_16U)

        # Update the previous frame
        prev_gray = gray
 
        # Read the next frame
        ret, frame = video.read()

    # Release the video capture object
    video.release()
    
    log("Diff count : {}".format(diff_count))
    
    # Normalize accumulated_diff by the number of diff
    accumulated_diff = accumulated_diff // (diff_count / 30)
    accumulated_diff = accumulated_diff.astype(np.uint8)
    
    # Create a specific subfolder
    analysis_subfolder = os.path.join(folder_path, ANALYSIS_FOLDER)
    os.makedirs(analysis_subfolder, exist_ok=True)

    # Save the accumulated difference image
    dif_path = os.path.join(folder_path, ANALYSIS_FOLDER, "2 - BW_dif.png")
    log("BW dif picture path: {}".format(dif_path))
    cv2.imwrite(dif_path, accumulated_diff)
    
    # Create a mask to get only movements of detected cells
    mask = np.zeros_like(accumulated_diff)
    for cell in cells_with_centers:
        if CELL_TYPE == "Bakers yeast":
            radius = int(round((math.sqrt(cell[4] / 3.14)) * 1.2))
        elif CELL_TYPE == "Universal":
            radius = int(round((math.sqrt(cell[4] / 3.14)) * 4))   
        
        cv2.circle(mask, cell[1][0], radius, (255), cv2.FILLED)
                   
    # Save the mask
    mask_path = os.path.join(folder_path, ANALYSIS_FOLDER, "3 - mask.png")
    log("Mask path: {}".format(mask_path))
    cv2.imwrite(mask_path, mask)
    
    # Apply the mask
    accumulated_diff = cv2.bitwise_and(accumulated_diff, mask)   
    
    # Save the accumulated_diff image after maask application
    after_mask_path = os.path.join(folder_path, ANALYSIS_FOLDER, "4 - after mask.png")
    log("After mask path: {}".format(after_mask_path))
    cv2.imwrite(after_mask_path, accumulated_diff)
    
    # Compute total difference value
    total_diff = np.sum(accumulated_diff)
    
    # Apply false colors
    accumulated_diff = cv2.applyColorMap(accumulated_diff, cv2.COLORMAP_JET)

    # Save the accumulated difference image
    dif_path = os.path.join(folder_path, ANALYSIS_FOLDER, "1 - dif.png")
    log("Dif picture path: {}".format(dif_path))
    cv2.imwrite(dif_path, accumulated_diff)
    
    movement_mean = 0
    if DEEP_ANALYSIS and CELL_TYPE == "Bakers yeast":
        subfolder = os.path.join(folder_path, DEEP_ANALYSIS_FOLDER)
        os.makedirs(subfolder, exist_ok=True)

        # Show movements
        picture_with_movements = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        for cell in cells_with_centers:
            for points in cell[1]:    
                picture_with_movements = cv2.circle(picture_with_movements, (points[0], points[1]), radius=1, color=(0, 255, 0), thickness=-1)
                cell[3] = math.sqrt(cell[2][0] ** 2 + cell[2][1] ** 2)
            log("Cell {} moved of {} pixels".format(cell[0], int(cell[3])))
            
            # Save a plot showing cell movements
            plt.figure()
            plt.plot([coord[0] for coord in cell[1]], [coord[1] for coord in cell[1]], marker='o', linestyle='-')
            plt.xlabel('X-axis [Pixel]')
            plt.ylabel('Y-axis [Pixel]')
            plt.title('Cell {} movements'.format(str(cell[0])))
            cell_movements_plot_path = os.path.join(subfolder + "/Cell_{}_movements.png".format(str(cell[0])))
            plt.savefig(cell_movements_plot_path)
            plt.close()
            log("Illumination plot path: {}".format(cell_movements_plot_path))
            
        # Save the image with movements
        mov_path = os.path.join(subfolder + "/mov.png")
        log("Mov picture path: {}".format(mov_path))
        cv2.imwrite(mov_path, picture_with_movements)
        
        # Save cell analysis data
        json_approved = cells_with_centers   
        cell_analysis_path = os.path.join(subfolder + "/cell_analysis.txt")
        with open(cell_analysis_path, "w") as file:
            file.write(json.dumps(json_approved, indent=4))
        log("Cell analysis file path: {}".format(cell_analysis_path))

        # Create a plot to compare all cell movement 
        plt.figure()    
        x = []
        y = []
        for cell in cells_with_centers:
            x.append(cell[0])
            y.append(cell[3])
        plt.bar(x, y)
        movement_mean = np.nanmean(y) / (diff_count / 30)
        plt.axhline(movement_mean, color="red", label="Mean : {:e}".format(movement_mean))
        plt.legend()
        plt.xlabel("Cell")
        plt.ylabel("Individual mean cell movement (per second)")
        plt.title("Individual mean cell movement (per second)")
        total_movement_plot_path = os.path.join(folder_path, "3 - individual_mean_cell_movement.png")
        plt.savefig(total_movement_plot_path)
        plt.close()
        log("Individual mean cell movement plot path: {}".format(total_movement_plot_path))

    # Create a plot to see variation of illumination (used during development)
    plt.figure()    
    plt.plot(illumination)
    plt.xlabel("Frame number")
    plt.ylabel("Sum of pixels")
    plt.title("Variation of illumination ")
    illumination_plot_path = os.path.join(folder_path, "2 - illumination_variation.png")
    plt.savefig(illumination_plot_path)
    plt.close()
    log("Illumination plot path: {}".format(illumination_plot_path))
    
    # Create a plot to see pixel changes between consecutive frames (used during development)
    plt.figure()    
    plt.plot(diffs)
    plt.xlabel("Frame number")
    plt.ylabel("Number of changed pixels")
    plt.title("Pixel Changes Between Consecutive Frames")
    change_over_frames_plot_path = os.path.join(folder_path, "1 - change_over_frames.png")
    plt.savefig(change_over_frames_plot_path)
    plt.close()
    log("Change over frames plot path: {}".format(change_over_frames_plot_path))
    log("")
    
    return total_diff, movement_mean, total_area

def process_videos(folder_path):
    """
    Perform video processing of all videos found in a folder.
    
    Parameters:
        folder_path: path of the folder
    """ 
    
    # Check if processing as already be done
    process = True
    deep_process = True
    for root, subfolders, files in os.walk(folder_path):
        for file in files:
            if file == "results.txt":
                process = False
                log("Folder already processed : {}".format(folder_path)) 

            if file == "deep_processing_results.txt":
                deep_process = False
                log("Folder already deep processed : {}".format(folder_path))
                
    if (process) or (DEEP_ANALYSIS and deep_process):
        diffs = [[],[],[],[]]
        if DEEP_ANALYSIS and CELL_TYPE == "Bakers yeast":
            movement_means = [[],[]]
            
        # Process all videos
        for root, subfolders, files in os.walk(folder_path):
            subfolders.sort()
            for subfolder in subfolders:
                diff, movement_mean, total_area = process_video(os.path.join(root, subfolder))
                diffs[0].append(subfolder)
                diffs[1].append(diff)
                diffs[2].append(total_area)
                if DEEP_ANALYSIS and CELL_TYPE == "Bakers yeast":
                    movement_means[0].append(subfolder)
                    movement_means[1].append(movement_mean)
            break
    
        # Create a plot to compare total pixel changes between each video 
        plt.figure() 
        ajusted_values = []
        for i in range(len(diffs[0])):
            diffs[3].append(float(diffs[1][i]) / float(diffs[2][i]))  
        plt.bar(diffs[0], diffs[3])
        mean = np.nanmean(diffs[3])
        plt.axhline(mean, color="red", label="Mean : {:e}".format(mean))
        plt.legend()
        plt.xlabel("Video")
        plt.ylabel("Number of changed pixels / total cell areas")
        plt.title("Comparison of total pixel changes divided by total cell areas")
        comparison_plot_path = os.path.join(folder_path, "Total_pixels_comparison.png")
        plt.savefig(comparison_plot_path)
        plt.close()
        log("Total pixels comparison plot path: {}".format(comparison_plot_path))

        # Save all data in a txt as JSON
        results_path = os.path.join(folder_path, "results.txt")
        with open(results_path, 'w') as file:
            file.write(json.dumps(jsonify(diffs), indent=4))
        log("Results saved as JSON here: {}".format(results_path))
    
        if DEEP_ANALYSIS and CELL_TYPE == "Bakers yeast":
            # Create a plot to compare total pixel changes between each video 
            plt.figure()    
            plt.bar(movement_means[0], movement_means[1])
            mean = np.nanmean(movement_means[1])
            plt.axhline(mean, color="red", label="Mean : {:e}".format(mean))
            plt.legend()
            plt.xlabel("Video")
            plt.ylabel("Overall mean of individual cell movements / second")
            plt.title("Comparison of overall mean of individual cell movements")
            movement_comparison_plot_path = os.path.join(folder_path, "Movement_mean_comparison.png")
            plt.savefig(movement_comparison_plot_path)
            plt.close()
            log("Movements means comparison plot path: {}".format(movement_comparison_plot_path))

            # Save all data in a txt as JSON
            results_path = os.path.join(folder_path, "deep_processing_results.txt")
            with open(results_path, 'w') as file:
                file.write(json.dumps(jsonify(movement_means), indent=4))
            log("Deep processing results saved as JSON here: {}".format(results_path))

def record_video(
    microscope,
    folder,
    video_length,
    filename = "video"
    ):
    """
    Perform video recording.
    
    Parameters:
        microscope: microscope component, necessary to control the microscope
        folder: path of the folder where the video file will be saved
        video_length: length of the video file in seconds
        filename: name of the video file ("video" by default)
        
    Returns:
        Path of the video file
    """ 
    

    # Create video output
    output = microscope.captures.new_video(False, filename, folder, "h264")

    # Lock the camera to avoid unexpected control from another thread
    with microscope.camera.lock:
        old_stream_resolution = change_def(microscope)
        
        # Set framerate
        old_framerate = microscope.camera.picamera.framerate 
        video_framerate = 30
        microscope.camera.picamera.framerate = video_framerate

        # Record the video
        microscope.camera.picamera.start_recording(output=output.file, format="h264", splitter_port=2, intra_period=0)
        microscope.camera.picamera.wait_recording(timeout=video_length, splitter_port=2)
        microscope.camera.picamera.stop_recording(2)
        
        # Reset the camera properties
        microscope.camera.picamera.framerate  = old_framerate
        reset_def(microscope, old_stream_resolution)

    output_file = output.file    
    log("Video recorded, path : {}".format(output_file))

    if MP4:
        # Convert the h264 stream to mp4
        input_file = output.file
        output_file = input_file[:-5] + ".mp4"
        (ffmpeg.input(input_file, r=video_framerate).output(output_file, r=video_framerate).run())

    return output_file

if not DEBUG:
    # Create extensions for Openflexure

    ##################
    # BASE EXTENSION #
    ##################
    
    ## Extension view
    class BaseAPI(ActionView):
        args = {
            "video_length": fields.Number(
                missing=1, example=1, description="Length (seconds) of video"
            ),
            "x_displacement": fields.Number(
                missing=1000, example=1000, description="Displacement in x axis"
            ),
            "x_displacement_nbr": fields.Number(
                missing=1, example=1, description="Number of displacements in x axis"
            ),
            "y_displacement": fields.Number(
                missing=1000, example=1000, description="Displacement in y axis"
            ),
            "y_displacement_nbr": fields.Number(
                missing=1, example=1, description="Number of displacements in y axis"
            ),
            "foldername": fields.String(
                missing = generate_basename(),
                example = generate_basename(),
                allow_none = True,
                description = "Name of the folder"
            ),
            "definition": fields.String(
                missing="1080p", example="1080p", description="Video definition"
            ),
            "cell_type": fields.String(
                missing="Universal", example="Universal", description="Cell type"
            ),
            "min_cell_threshold": fields.Number(
                missing=20, example=20, description="Minimum number of cells"
            ),
            "analysis": fields.List(
                fields.String,
                missing=[],
                allow_none=True,
                description="Analysis option"
            ),
            "deep_analysis": fields.List(
                fields.String,
                missing=[],
                allow_none=True,
                description="Deep analysis option"
            ),
            "mp4": fields.List(
                fields.String,
                missing=[],
                allow_none=True,
                description="MP4 video conversion option"
            ),
        }

        def post(self, args):
            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)
            
            # Get parameters
            video_length = args.get("video_length")
            x_displacement = args.get("x_displacement")
            y_displacement = args.get("y_displacement")
            x_displacement_nbr = args.get("x_displacement_nbr")
            y_displacement_nbr = args.get("y_displacement_nbr")
            folder = args.get("foldername")

            log("Folder name: {}".format(folder))
            
            global DEFINITION
            DEFINITION = args.get("definition")
            
            global CELL_TYPE
            CELL_TYPE = args.get("cell_type")
            
            global MIN_CELL_THRESHOLD
            MIN_CELL_THRESHOLD = args.get("min_cell_threshold")
            log("min cell thresh {}".format(MIN_CELL_THRESHOLD))
            
            analysis = args.get("analysis")
            global ANALYSIS
            if "Yes" in analysis:
                ANALYSIS = True 
            else:
                ANALYSIS = False
                
            deep_analysis = args.get("deep_analysis")
            global DEEP_ANALYSIS
            if "Yes" in deep_analysis:
                DEEP_ANALYSIS = True 
            else:
                DEEP_ANALYSIS = False  
                    
            mp4 = args.get("mp4")
            global MP4
            if "Yes" in mp4:
                MP4 = True 
            else:
                MP4 = False  

            # Get the microscope component
            microscope = find_component("org.openflexure.microscope")
        
            # Perform autofocus
            autofocus = AutofocusExtension()
            autofocus.autofocus(microscope, list(np.linspace(-150, 150, 20)))
            
            # Perform displacements and video recording
            for x in range(int(x_displacement_nbr) + 1):
                for y in range(int(y_displacement_nbr) + 1):
                    # Create a new folder
                    video_id = str(x + 1) + "_" + str(y + 1)
                    subfolder = os.path.join(BASE_FOLDER, folder, video_id)
                    os.makedirs(subfolder, exist_ok=True)
   
                    # Create a specific subfolder for cell count
                    cell_count_subfolder = os.path.join(subfolder, COUNT_CELL_FOLDER)
                    os.makedirs(cell_count_subfolder, exist_ok=True)
                        
                    # Ensure minimal number of cells
                    record_video(microscope, cell_count_subfolder, 1, CELL_COUNT_VIDEO_FILENAME)
                    cell_count = def_count_cells(cell_count_subfolder)
                    while cell_count < MIN_CELL_THRESHOLD:
                        microscope.stage.move_rel(displacement=(0,y_displacement,0))
                        autofocus.autofocus(microscope, list(np.linspace(-150, 150, 20)))
                        record_video(microscope, cell_count_subfolder, 1, CELL_COUNT_VIDEO_FILENAME)
                        cell_count = def_count_cells(cell_count_subfolder)   
            
                    # Record video
                    record_video(microscope, subfolder, video_length) 
                    
                    if y != (y_displacement_nbr):
                        microscope.stage.move_rel(displacement=(0,y_displacement,0))
                        autofocus.autofocus(microscope, list(np.linspace(-150, 150, 20)))

                if x != (x_displacement_nbr):
                    microscope.stage.move_rel(displacement=(x_displacement,-(y_displacement * y_displacement_nbr),0))
                    autofocus.autofocus(microscope, list(np.linspace(-150, 150, 20)))
                
            if ANALYSIS or DEEP_ANALYSIS:
                # Process videos
                process_videos(os.path.join(BASE_FOLDER + folder))
                
            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)

    ## Extension GUI (OpenFlexure eV)
    # Alternate form without any dynamic parts
    base_extension_gui = {
        "icon": "videocam",  # Name of an icon from https://material.io/resources/icons/
        "forms": [  # List of forms. Each form is a collapsible accordion panel
            {
                "name": "Record and analyse nanomotion",  # Form title
                "route": "/base",  # The URL rule (as given by "add_view") of your submission view
                "isTask": True,  # This forms submission starts a background task
                "isCollapsible": False,  # This form cannot be collapsed into an accordion
                "submitLabel": "Start recording",  # Label for the form submit button
                "schema": [  # List of dictionaries. Each element is a form component.
                    {
                        "fieldType": "numberInput",
                        "name": "video_length",
                        "label": "Length (seconds) of video",
                        "min": 1, 
                        "step": 1,  
                        "default": 1, 
                    },
                    {
                        "fieldType": "numberInput",
                        "name": "x_displacement",
                        "label": "Displacement in x axis",
                        "min": 1, 
                        "step": 1,  
                        "default": 1000,  
                    },
                    {
                        "fieldType": "numberInput",
                        "name": "x_displacement_nbr",
                        "label": "Number of displacements in x axis",
                        "min": 1, 
                        "step": 1, 
                        "default": 1, 
                    },
                    {
                        "fieldType": "numberInput",
                        "name": "y_displacement",
                        "label": "Displacement in y axis",
                        "min": 1, 
                        "step": 1,
                        "default": 1000,
                    },
                    {
                        "fieldType": "numberInput",
                        "name": "y_displacement_nbr",
                        "label": "Number of displacements in y axis",
                        "min": 1, 
                        "step": 1,  
                        "default": 1,  
                    },
                    {
                        "fieldType": "textInput",
                        "name": "foldername",
                        "label": "Folder name",
                        "value": "test",
                        "default": "test",
                        "placeholder": "test",
                    },
                    {
                        "fieldType": "selectList",
                        "name": "definition",
                        "label": "Video definition",
                        "value": "1080p",
                        "options": [
                            "1080p",
                            "480p",
                        ],
                    },
                    {
                        "fieldType": "selectList",
                        "name": "cell_type",
                        "label": "Cell type",
                        "value": "Universal",
                        "options": [
                            "Universal",
                            "Bakers yeast",
                        ],
                    },
                    {
                        "fieldType": "numberInput",
                        "name": "min_cell_threshold",
                        "label": "Minimum number of cells",
                        "min": 1,
                        "step": 1,  
                        "default": 5, 
                    },
                    {
                        "fieldType": "checkList",
                        "name": "analysis",
                        "label": "Video processing",
                        "value": [],
                        "options": ["Yes"],
                    },
                    {
                        "fieldType": "checkList",
                        "name": "deep_analysis",
                        "label": "Deep video processing",
                        "value": [],
                        "options": ["Yes"],
                    },
                    {
                        "fieldType": "checkList",
                        "name": "mp4",
                        "label": "MP4 video",
                        "value": [],
                        "options": ["Yes"],
                    },
                ],
            }
        ],
    }

    # Create your extension object
    base_extension = BaseExtension("org.openflexure.base_extension", version="0.0.0")

    # Add methods to your extension
    base_extension.add_method(record_video, "record_video")

    # Add API views to your extension
    base_extension.add_view(BaseAPI, "/base")

    # Add OpenFlexure eV GUI to your extension
    base_extension.add_meta("gui", build_gui(base_extension_gui, base_extension))
    
           
    ##########################
    # POST PROCESS EXTENSION #
    ##########################
    
    ## Extension view
    class PostProcessAPI(ActionView):
        args = {
            "path": fields.String(
                missing = "",
                example = BASE_FOLDER,
                allow_none = False,
                description = "Path"
            ),
            "subfolders": fields.List(
                fields.String,
                missing=[],
                allow_none=True,
                description="Analyse in sufolders"
            ),
            "deep_analysis": fields.List(
                fields.String,
                missing=[],
                allow_none=True,
                description="Deep analysis option"
            ),
            "cell_type": fields.String(
                missing="Universal", example="Universal", description="Cell type"
            ),
        }

        def post(self, args):
            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)
            
            # Get parameters
            global CELL_TYPE
            CELL_TYPE = args.get("cell_type")
                
            deep_analysis = args.get("deep_analysis")
            global DEEP_ANALYSIS
            if "Yes" in deep_analysis:
                DEEP_ANALYSIS = True 
            else:
                DEEP_ANALYSIS = False  
            
            path = args.get("path")
            
            subfolders = args.get("subfolders")
            if "Yes" in subfolders:
                for _, subfolders, _ in os.walk(path):
                    for subfolder in subfolders: 
                        # Process videos
                        process_videos(os.path.join(path, subfolder))
                    break
            else:  
                # Process videos
                process_videos(path)

            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)

    ## Extension GUI (OpenFlexure eV)
    # Alternate form without any dynamic parts
    post_process_extension_gui = {
        "icon": "settings",  # Name of an icon from https://material.io/resources/icons/
        "forms": [  # List of forms. Each form is a collapsible accordion panel
            {
                "name": "Analyse videos",  # Form title
                "route": "/post_process",  # The URL rule (as given by "add_view") of your submission view
                "isTask": True,  # This forms submission starts a background task
                "isCollapsible": False,  # This form cannot be collapsed into an accordion
                "submitLabel": "Start analysis",  # Label for the form submit button
                "schema": [  # List of dictionaries. Each element is a form component.
                    {
                        "fieldType": "textInput",
                        "name": "path",
                        "label": "Path",
                        "value": BASE_FOLDER,
                        "default": "",
                        "placeholder": "path",
                    },
                    {
                        "fieldType": "checkList",
                        "name": "subfolders",
                        "label": "Analyse subfolders",
                        "value": [],
                        "options": ["Yes"],
                    },
                    {
                        "fieldType": "selectList",
                        "name": "cell_type",
                        "label": "Cell type",
                        "value": "Universal",
                        "options": [
                            "Universal",
                            "Bakers yeast",
                        ],
                    },
                    {
                        "fieldType": "checkList",
                        "name": "deep_analysis",
                        "label": "Deep video processing",
                        "value": [],
                        "options": ["Yes"],
                    },
                ],
            }
        ],
    }

    # Create your extension object
    post_process_extension = BaseExtension("org.openflexure.post_process_extension", version="0.0.0")

    # Add API views to your extension
    post_process_extension.add_view(PostProcessAPI, "/post_process")

    # Add OpenFlexure eV GUI to your extension
    post_process_extension.add_meta("gui", build_gui(post_process_extension_gui, post_process_extension))
    
    ########################
    # COMPARISON EXTENSION #
    ########################
    
    ## Extension view
    class ComparisonAPI(ActionView):
        args = {
            "path": fields.String(
                missing = "",
                example = BASE_FOLDER,
                allow_none = False,
                description = "Path"
            )
        }

        def post(self, args):
            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)

            # Get parameters
            path = args.get("path")
            
            # Compare data
            full_comparison(path)
            
            # Change permissions of all folders/files created to simplify future deletions
            chmod_recursive(os.path.join(BASE_FOLDER), 0o777)

    ## Extension GUI (OpenFlexure eV)
    # Alternate form without any dynamic parts
    comparison_gui = {
        "icon": "settings",  # Name of an icon from https://material.io/resources/icons/
        "forms": [  # List of forms. Each form is a collapsible accordion panel
            {
                "name": "Compare nanomotion analysis",  # Form title
                "route": "/comparison",  # The URL rule (as given by "add_view") of your submission view
                "isTask": True,  # This forms submission starts a background task
                "isCollapsible": False,  # This form cannot be collapsed into an accordion
                "submitLabel": "Compare",  # Label for the form submit button
                "schema": [  # List of dictionaries. Each element is a form component.
                    {
                        "fieldType": "textInput",
                        "name": "path",
                        "label": "Path",
                        "value": BASE_FOLDER,
                        "default": "",
                        "placeholder": "path",
                    },
                ],
            }
        ],
    }

    # Create your extension object
    comparison_extension = BaseExtension("org.openflexure.comparison_extension", version="0.0.0")

    # Add API views to your extension
    comparison_extension.add_view(ComparisonAPI, "/comparison")

    # Add OpenFlexure eV GUI to your extension
    comparison_extension.add_meta("gui", build_gui(comparison_gui, comparison_extension))
      
else:
    # This code is executed if DEBUG = True
    # The / in path needs to be put as \\   
    
    # Examples : 
    # process_videos("path")
    full_comparison("path")
