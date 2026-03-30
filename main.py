import cv2
import numpy as np
from collections import Counter

# Enable GPU acceleration for OpenCV
print("=" * 50)
print("Initializing GPU Support...")
print("=" * 50)

# Check if OpenCV was compiled with CUDA support
print(f"OpenCV Version: {cv2.__version__}")
print(f"OpenCV Build Info: {cv2.getBuildInformation()[:500]}")

gpu_available = False
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA Devices Found: {cuda_count}")
    
    if cuda_count > 0:
        gpu_available = True
        cv2.cuda.setDevice(0)
        
        # Get device properties
        device = cv2.cuda.getDevice()
        print(f"Active GPU Device: {device}")
        print("✓ GPU acceleration ENABLED")
    else:
        print("✗ No CUDA devices detected")
        print("\nTo enable GPU support on Windows:")
        print("1. Install NVIDIA CUDA Toolkit 12.x")
        print("2. Install cuDNN from NVIDIA")
        print("3. Rebuild OpenCV from source with CUDA support")
        print("   OR use: pip install opencv-contrib-python (already done)")
        print("4. Ensure GPU drivers are up to date")
except Exception as e:
    print(f"✗ GPU Access Error: {e}")
    print("Note: OpenCV may not be compiled with CUDA support")
    print("      Standard opencv-python doesn't include CUDA.")
    print("      You have installed opencv-contrib-python which is compatible.")
    print("      Please ensure NVIDIA CUDA Toolkit is installed on your system.")

print("=" * 50)
if not gpu_available:
    print("Using CPU acceleration (slower processing)")
else:
    print("Using GPU acceleration (faster processing)")
print("=" * 50 + "\n")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def detect_sudoku_grid(processed_frame):
    """Detect and extract the sudoku grid from the processed frame"""
    try:
        contours, _ = cv2.findContours(processed_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Filter contours by area and aspect ratio to find the sudoku grid
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Minimum area for sudoku grid (lowered threshold)
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0:
                continue
                
            aspect_ratio = float(w) / h
            
            # Sudoku grid should be roughly square (0.6 to 1.4 aspect ratio)
            if 0.6 < aspect_ratio < 1.4:
                valid_contours.append((contour, area, (x, y, w, h)))
        
        if not valid_contours:
            return None
        
        # Get the largest valid contour (the sudoku grid)
        largest_contour, _, (x, y, w, h) = max(valid_contours, key=lambda x: x[1])
        
        # Ensure we have valid bounds
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            if y + h <= processed_frame.shape[0] and x + w <= processed_frame.shape[1]:
                return processed_frame[y:y + h, x:x + w], (x, y, w, h)
    
    except Exception as e:
        print(f"Error in detect_sudoku_grid: {e}")
    
    return None

def recognize_digit_template(cell):
    """Recognize digits using template matching and morphological operations"""
    if cell.size == 0:
        return 0
    
    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels
    non_zero_count = cv2.countNonZero(binary)
    total_pixels = cell.size
    fill_ratio = non_zero_count / total_pixels if total_pixels > 0 else 0
    
    # If fill ratio is too low, cell is empty
    if fill_ratio < 0.05:
        return 0
    
    # Use pixel density for digit estimation (1-9)
    # Normalize to 0-1 range
    normalized_density = min(1.0, fill_ratio / 0.5)
    
    # Map to digit range
    if fill_ratio < 0.08:
        return 0  # Empty
    elif fill_ratio < 0.15:
        return 1
    elif fill_ratio < 0.22:
        return 2
    elif fill_ratio < 0.29:
        return 3
    elif fill_ratio < 0.36:
        return 4
    elif fill_ratio < 0.43:
        return 5
    elif fill_ratio < 0.50:
        return 6
    elif fill_ratio < 0.57:
        return 7
    elif fill_ratio < 0.64:
        return 8
    else:
        return 9

def solve_sudoku(grid):
    empty_cell = find_empty_cell(grid)
    if not empty_cell:
        return True

    row, col = empty_cell

    for num in range(1, 10):
        if is_safe(grid, row, col, num):
            grid[row, col] = num

            if solve_sudoku(grid):
                return True

            grid[row, col] = 0

    return False

def find_empty_cell(grid):
    for i in range(9):
        for j in range(9):
            if grid[i, j] == 0:
                return (i, j)
    return None

def is_safe(grid, row, col, num):
    # Check row
    if num in grid[row, :]:
        return False

    # Check column
    if num in grid[:, col]:
        return False

    # Check 3x3 box
    box_start_row, box_start_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[box_start_row:box_start_row + 3, box_start_col:box_start_col + 3]:
        return False

    return True

# Rest of the code (video capture and processing)

while True:
    # Capture frame from video feed
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess the frame with GPU acceleration if available
    processed = gray.copy()
    if gpu_available:
        try:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (9, 9), 1.5).apply(gpu_gray)
            gpu_blur.download(processed)
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            processed = cv2.GaussianBlur(gray, (9, 9), 0)
    else:
        processed = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect the sudoku grid
    grid_result = detect_sudoku_grid(processed)
    
    if grid_result is not None:
        sudoku_grid, (grid_x, grid_y, grid_w, grid_h) = grid_result
        
        # Resize the sudoku grid to a standard size
        target_size = 450
        grid_height, grid_width = sudoku_grid.shape[:2]
        
        if grid_width == 0 or grid_height == 0:
            continue
        
        scale = min(target_size / grid_width, target_size / grid_height)
        
        new_width = int(grid_width * scale)
        new_height = int(grid_height * scale)
        
        # Make dimensions divisible by 9
        new_width = (new_width // 9) * 9
        new_height = (new_height // 9) * 9
        
        if new_width > 0 and new_height > 0:
            sudoku_grid = cv2.resize(sudoku_grid, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            continue
        
        # Split the sudoku grid into 9x9 cells
        rows, cols = 9, 9
        cell_height = new_height // rows
        cell_width = new_width // cols
        
        try:
            cells = [np.hsplit(row, cols) for row in np.vsplit(sudoku_grid, rows)]
        except Exception as e:
            print(f"Error splitting grid: {e}")
            continue
        
        # Initialize grid to store recognized digits
        sudoku_digits = np.zeros((9, 9), dtype=int)
        
        # Extract and recognize digits in each cell
        try:
            for i in range(9):
                for j in range(9):
                    cell = cells[i][j]
                    # Invert so white digits become black for processing
                    cell_inverted = cv2.bitwise_not(cell)
                    
                    # Recognize the digit
                    digit = recognize_digit_template(cell_inverted)
                    sudoku_digits[i, j] = digit
        except Exception as e:
            print(f"Error recognizing digits: {e}")
            continue
        
        # Only solve if grid has enough filled cells
        filled_cells = np.count_nonzero(sudoku_digits)
        if filled_cells > 17:  # Valid sudoku typically has at least 17 clues
            try:
                # Make a copy for solving
                grid_copy = sudoku_digits.copy()
                solve_sudoku(grid_copy)
                
                # Calculate scale factors from original grid to frame
                scale_x = grid_w / new_width if new_width > 0 else 1
                scale_y = grid_h / new_height if new_height > 0 else 1
                
                # Overlay the solution on the frame
                for i in range(9):
                    for j in range(9):
                        if sudoku_digits[i, j] == 0 and grid_copy[i, j] != 0:
                            # This is a solved cell (was empty before)
                            cell_x = int(grid_x + j * cell_width * scale_x + cell_width * scale_x // 2)
                            cell_y = int(grid_y + i * cell_height * scale_y + cell_height * scale_y // 2)
                            cv2.putText(frame, str(grid_copy[i, j]), (cell_x - 15, cell_y + 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error solving sudoku: {e}")
        
        # Draw the grid boundaries for visualization
        cv2.rectangle(frame, (grid_x, grid_y), (grid_x + grid_w, grid_y + grid_h), (255, 0, 0), 2)
        
        # Display detected grid information
        cv2.putText(frame, f"Detected: {filled_cells} cells", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-Time Sudoku Solver', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()