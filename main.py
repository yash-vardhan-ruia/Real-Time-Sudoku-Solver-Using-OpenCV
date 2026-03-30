import cv2
import numpy as np
from collections import deque
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9

CELL_MARGIN_RATIO = 0.22
MIN_COMPONENT_AREA_RATIO = 0.03
MIN_COMPONENT_FILL_RATIO = 0.12
MIN_PREDICTION_CONFIDENCE = 0.70
MIN_CONFIDENCE_MARGIN = 0.12
CORNER_SMOOTHING_WINDOW = 5

WARP_DESTINATION = np.array(
    [[0, 0], [GRID_SIZE - 1, 0], [GRID_SIZE - 1, GRID_SIZE - 1], [0, GRID_SIZE - 1]],
    dtype="float32",
)

print("=" * 50)
print("Initializing GPU Support...")
print("=" * 50)
print(f"OpenCV Version: {cv2.__version__}")

gpu_available = False
try:
    cv2.ocl.setUseOpenCL(True)
    gpu_available = bool(cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL())
    print(f"OpenCL Available: {cv2.ocl.haveOpenCL()}")
    print(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")
    if gpu_available:
        print("✓ GPU acceleration ENABLED via OpenCL")
    else:
        print("✗ OpenCL runtime not active, using CPU")
except Exception as error:
    print(f"✗ GPU Access Error: {error}")

print("=" * 50)
print("Using GPU acceleration (faster processing)" if gpu_available else "Using CPU acceleration (slower processing)")
print("=" * 50 + "\n")


def train_digit_model():
    digits = load_digits()
    classifier = KNeighborsClassifier(n_neighbors=5, weights="distance")
    classifier.fit(digits.data, digits.target)
    return classifier


DIGIT_MODEL = train_digit_model()


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def find_and_warp_grid(gray, binary):
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = gray.shape[0] * gray.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4:
            continue

        points = approximation.reshape(4, 2).astype("float32")
        ordered = order_points(points)

        perspective_matrix = cv2.getPerspectiveTransform(ordered, WARP_DESTINATION)
        inverse_matrix = cv2.getPerspectiveTransform(WARP_DESTINATION, ordered)

        warped_gray = cv2.warpPerspective(gray, perspective_matrix, (GRID_SIZE, GRID_SIZE))
        warped_binary = cv2.warpPerspective(binary, perspective_matrix, (GRID_SIZE, GRID_SIZE))
        return warped_gray, warped_binary, ordered, inverse_matrix

    return None


def preprocess_cell(cell_binary):
    margin = int(CELL_SIZE * CELL_MARGIN_RATIO)
    cropped = cell_binary[margin:CELL_SIZE - margin, margin:CELL_SIZE - margin]
    if cropped.size == 0:
        return None

    kernel = np.ones((2, 2), dtype=np.uint8)
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cropped, connectivity=8)
    if num_labels <= 1:
        return None

    largest_label = None
    largest_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        touches_border = x == 0 or y == 0 or (x + w) >= cropped.shape[1] or (y + h) >= cropped.shape[0]
        if touches_border:
            continue

        if area > largest_area:
            largest_area = area
            largest_label = label

    if largest_label is None:
        return None

    minimum_area = cropped.shape[0] * cropped.shape[1] * MIN_COMPONENT_AREA_RATIO
    if largest_area < minimum_area:
        return None

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    if w <= 2 or h <= 6:
        return None

    fill_ratio = largest_area / float(w * h)
    if fill_ratio < MIN_COMPONENT_FILL_RATIO:
        return None

    aspect_ratio = w / float(h)
    if aspect_ratio < 0.08 or aspect_ratio > 1.4:
        return None

    mask = np.zeros_like(cropped, dtype=np.uint8)
    mask[labels == largest_label] = 255

    digit = mask[y:y + h, x:x + w]
    if digit.size == 0:
        return None

    canvas = np.zeros((28, 28), dtype=np.uint8)
    scale = min(20.0 / w, 20.0 / h)
    resized_w = max(1, int(w * scale))
    resized_h = max(1, int(h * scale))
    resized_digit = cv2.resize(digit, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    offset_x = (28 - resized_w) // 2
    offset_y = (28 - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized_digit
    return canvas


def predict_digit(cell_binary):
    prepared = preprocess_cell(cell_binary)
    if prepared is None:
        return 0

    sample = cv2.resize(prepared, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    sample = (sample / 255.0) * 16.0
    sample = sample.reshape(1, -1)

    probabilities = DIGIT_MODEL.predict_proba(sample)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    prediction = int(sorted_indices[0])
    confidence = float(probabilities[sorted_indices[0]])
    runner_up_confidence = float(probabilities[sorted_indices[1]])

    if prediction == 0:
        return 0

    if confidence < MIN_PREDICTION_CONFIDENCE:
        return 0

    if (confidence - runner_up_confidence) < MIN_CONFIDENCE_MARGIN:
        return 0

    return prediction


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
    for row in range(9):
        for col in range(9):
            if grid[row, col] == 0:
                return row, col
    return None


def is_safe(grid, row, col, num):
    if num in grid[row, :]:
        return False
    if num in grid[:, col]:
        return False

    box_start_row = 3 * (row // 3)
    box_start_col = 3 * (col // 3)
    if num in grid[box_start_row:box_start_row + 3, box_start_col:box_start_col + 3]:
        return False
    return True


def is_valid_initial_grid(grid):
    for row in range(9):
        values = [value for value in grid[row, :] if value != 0]
        if len(values) != len(set(values)):
            return False

    for col in range(9):
        values = [value for value in grid[:, col] if value != 0]
        if len(values) != len(set(values)):
            return False

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = grid[box_row:box_row + 3, box_col:box_col + 3].flatten()
            values = [value for value in box if value != 0]
            if len(values) != len(set(values)):
                return False

    return True


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

corner_history = deque(maxlen=CORNER_SMOOTHING_WINDOW)
previous_grid = None
cached_solution = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gpu_available:
        try:
            gray_umat = cv2.UMat(gray)
            blurred = cv2.GaussianBlur(gray_umat, (7, 7), 0).get()
        except Exception:
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    else:
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    warped_result = find_and_warp_grid(gray, binary)
    if warped_result is None:
        cv2.putText(frame, "Grid not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Real-Time Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    _, warped_binary, corners, _ = warped_result

    corner_history.append(corners)
    smoothed_corners = np.mean(np.array(corner_history), axis=0).astype(np.float32)
    smoothed_perspective = cv2.getPerspectiveTransform(smoothed_corners, WARP_DESTINATION)
    inverse_matrix = cv2.getPerspectiveTransform(WARP_DESTINATION, smoothed_corners)
    warped_binary = cv2.warpPerspective(binary, smoothed_perspective, (GRID_SIZE, GRID_SIZE))

    sudoku_digits = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE
            y2 = (row + 1) * CELL_SIZE
            x1 = col * CELL_SIZE
            x2 = (col + 1) * CELL_SIZE
            cell = warped_binary[y1:y2, x1:x2]
            sudoku_digits[row, col] = predict_digit(cell)

    filled_cells = int(np.count_nonzero(sudoku_digits))
    valid_clues = is_valid_initial_grid(sudoku_digits)

    polygon = smoothed_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    if filled_cells >= 17 and valid_clues:
        grid_matches_previous = previous_grid is not None and np.array_equal(sudoku_digits, previous_grid)

        if grid_matches_previous and cached_solution is not None:
            solved_grid = cached_solution
            solved = True
        else:
            solved_grid = sudoku_digits.copy()
            solved = solve_sudoku(solved_grid)
            previous_grid = sudoku_digits.copy()
            cached_solution = solved_grid.copy() if solved else None

        if solved:
            for row in range(9):
                for col in range(9):
                    if sudoku_digits[row, col] != 0:
                        continue

                    warped_point = np.array(
                        [[[col * CELL_SIZE + CELL_SIZE / 2, row * CELL_SIZE + CELL_SIZE / 2]]], dtype=np.float32
                    )
                    frame_point = cv2.perspectiveTransform(warped_point, inverse_matrix)[0][0]
                    text_x = int(frame_point[0] - 10)
                    text_y = int(frame_point[1] + 10)

                    cv2.putText(
                        frame,
                        str(int(solved_grid[row, col])),
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
    else:
        previous_grid = None
        cached_solution = None

    status_color = (0, 255, 0) if valid_clues else (0, 0, 255)
    status_text = f"Detected: {filled_cells} cells"
    if not valid_clues:
        status_text += " | invalid clues"

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow("Real-Time Sudoku Solver", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()