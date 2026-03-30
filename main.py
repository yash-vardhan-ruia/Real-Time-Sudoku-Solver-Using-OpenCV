import cv2
import numpy as np
import pickle
from pathlib import Path
from collections import deque
from sklearn.neural_network import MLPClassifier

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9

CELL_MARGIN_RATIO = 0.16
MIN_COMPONENT_AREA_RATIO = 0.015
MIN_COMPONENT_FILL_RATIO = 0.12
MIN_PREDICTION_CONFIDENCE = 0.58
MIN_CONFIDENCE_MARGIN = 0.06
CORNER_SMOOTHING_WINDOW = 5
DIGIT_SMOOTHING_WINDOW = 5
MIN_EMPTY_PIXEL_RATIO = 0.03
MIN_DIGIT_VOTE_SHARE = 0.55
TEMPORAL_DECAY = 0.72
MIN_TEMPORAL_CONFIDENCE = 0.45
MODEL_IMAGE_SIZE = 28
SAMPLES_PER_DIGIT = 450
EMPTY_SAMPLES = 1200
MODEL_CACHE_PATH = Path(__file__).with_name("printed_digit_model.pkl")

BLUR_KERNEL_SIZE = (7, 7)
THRESH_BLOCK_SIZE = 11
THRESH_C = 2
GRID_OPEN_KERNEL = np.ones((3, 3), dtype=np.uint8)
CELL_CLEAN_KERNEL = np.ones((2, 2), dtype=np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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


def generate_printed_digit_dataset(samples_per_digit=SAMPLES_PER_DIGIT, empty_samples=EMPTY_SAMPLES):
    rng = np.random.default_rng(42)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]

    features = []
    labels = []

    for digit in range(1, 10):
        text = str(digit)
        for _ in range(samples_per_digit):
            canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.uint8)

            font = fonts[int(rng.integers(0, len(fonts)))]
            font_scale = float(rng.uniform(0.75, 1.25))
            thickness = int(rng.integers(1, 4))

            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            jitter_x = int(rng.integers(-3, 4))
            jitter_y = int(rng.integers(-3, 4))
            x = max(0, (MODEL_IMAGE_SIZE - text_w) // 2 + jitter_x)
            y = min(MODEL_IMAGE_SIZE - 1, (MODEL_IMAGE_SIZE + text_h) // 2 + jitter_y)

            cv2.putText(canvas, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)

            angle = float(rng.uniform(-15, 15))
            scale = float(rng.uniform(0.90, 1.10))
            tx = float(rng.uniform(-2.5, 2.5))
            ty = float(rng.uniform(-2.5, 2.5))

            transform = cv2.getRotationMatrix2D((MODEL_IMAGE_SIZE / 2, MODEL_IMAGE_SIZE / 2), angle, scale)
            transform[0, 2] += tx
            transform[1, 2] += ty
            canvas = cv2.warpAffine(canvas, transform, (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), borderValue=0)

            if rng.random() < 0.35:
                kernel_size = int(rng.choice([3, 5]))
                canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), 0)

            if rng.random() < 0.35:
                op_kernel = np.ones((2, 2), dtype=np.uint8)
                if rng.random() < 0.5:
                    canvas = cv2.dilate(canvas, op_kernel, iterations=1)
                else:
                    canvas = cv2.erode(canvas, op_kernel, iterations=1)

            noise = rng.normal(0.0, 12.0, size=canvas.shape).astype(np.float32)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            features.append((canvas.astype(np.float32) / 255.0).reshape(-1))
            labels.append(digit)

    for _ in range(empty_samples):
        canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.uint8)

        if rng.random() < 0.7:
            line_count = int(rng.integers(1, 4))
            for _ in range(line_count):
                x1 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                y1 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                x2 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                y2 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                color = int(rng.integers(20, 110))
                thickness = int(rng.integers(1, 3))
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

        noise = rng.normal(0.0, 16.0, size=canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        features.append((canvas.astype(np.float32) / 255.0).reshape(-1))
        labels.append(0)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def train_digit_model():
    training_features, training_labels = generate_printed_digit_dataset()
    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=42,
        verbose=False,
    )
    classifier.fit(training_features, training_labels)
    return classifier


def load_or_train_digit_model():
    if MODEL_CACHE_PATH.exists():
        try:
            with MODEL_CACHE_PATH.open("rb") as model_file:
                return pickle.load(model_file)
        except Exception:
            pass

    trained_model = train_digit_model()
    try:
        with MODEL_CACHE_PATH.open("wb") as model_file:
            pickle.dump(trained_model, model_file)
    except Exception:
        pass
    return trained_model


DIGIT_MODEL = load_or_train_digit_model()


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diffs = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def preprocess_frame(frame, use_gpu):
    if use_gpu:
        try:
            frame_umat = cv2.UMat(frame)
            gray_umat = cv2.cvtColor(frame_umat, cv2.COLOR_BGR2GRAY)
            normalized_gray = CLAHE.apply(gray_umat.get())
            normalized_gray_umat = cv2.UMat(normalized_gray)
            blurred_umat = cv2.GaussianBlur(normalized_gray_umat, BLUR_KERNEL_SIZE, 0)
            binary_umat = cv2.adaptiveThreshold(
                blurred_umat,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                THRESH_BLOCK_SIZE,
                THRESH_C,
            )
            return gray_umat.get(), binary_umat.get(), True
        except Exception:
            pass

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized_gray = CLAHE.apply(gray)
    blurred = cv2.GaussianBlur(normalized_gray, BLUR_KERNEL_SIZE, 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        THRESH_BLOCK_SIZE,
        THRESH_C,
    )
    return gray, binary, False


def find_grid_corners(gray, binary):
    binary_for_grid = cv2.morphologyEx(binary, cv2.MORPH_OPEN, GRID_OPEN_KERNEL)
    contours, _ = cv2.findContours(binary_for_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        return ordered

    return None


def warp_binary_for_grid(binary, perspective_matrix, use_gpu):
    if use_gpu:
        try:
            return cv2.warpPerspective(cv2.UMat(binary), perspective_matrix, (GRID_SIZE, GRID_SIZE)).get(), True
        except Exception:
            pass

    return cv2.warpPerspective(binary, perspective_matrix, (GRID_SIZE, GRID_SIZE)), False


def preprocess_cell(cell_binary):
    margin = int(CELL_SIZE * CELL_MARGIN_RATIO)
    cropped = cell_binary[margin:CELL_SIZE - margin, margin:CELL_SIZE - margin]
    if cropped.size == 0:
        return None

    raw_fill_ratio = float(np.count_nonzero(cropped)) / float(cropped.size)
    if raw_fill_ratio < MIN_EMPTY_PIXEL_RATIO:
        return None

    cropped = cv2.morphologyEx(cropped, cv2.MORPH_CLOSE, CELL_CLEAN_KERNEL)

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
        return 0, 0.0

    sample = prepared.astype(np.float32) / 255.0
    sample = sample.reshape(1, -1)

    probabilities = DIGIT_MODEL.predict_proba(sample)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    prediction = int(sorted_indices[0])
    confidence = float(probabilities[sorted_indices[0]])
    runner_up_confidence = float(probabilities[sorted_indices[1]])

    if prediction == 0:
        return 0, 0.0

    if confidence < MIN_PREDICTION_CONFIDENCE:
        return 0, confidence

    if (confidence - runner_up_confidence) < MIN_CONFIDENCE_MARGIN:
        return 0, confidence

    return prediction, confidence


def smooth_digit_prediction(history_entries):
    weighted_votes = np.zeros(10, dtype=np.float32)
    vote_counts = np.zeros(10, dtype=np.int32)
    history_length = len(history_entries)

    for index, (digit, confidence) in enumerate(history_entries):
        if digit <= 0:
            continue

        adjusted_confidence = float(confidence)
        if adjusted_confidence < MIN_TEMPORAL_CONFIDENCE:
            continue

        age = (history_length - 1) - index
        decay = TEMPORAL_DECAY ** age
        weight = adjusted_confidence * decay

        weighted_votes[digit] += weight
        vote_counts[digit] += 1

    if float(np.sum(weighted_votes)) <= 0.0:
        return 0, 0.0

    best_digit = int(np.argmax(weighted_votes))
    total_weight = float(np.sum(weighted_votes))
    best_share = float(weighted_votes[best_digit] / total_weight)

    if best_share < MIN_DIGIT_VOTE_SHARE:
        return 0, 0.0

    if len(history_entries) >= 3 and vote_counts[best_digit] < 2:
        return 0, 0.0

    return best_digit, best_share


def sanitize_grid_by_confidence(grid, confidence_grid):
    sanitized_grid = grid.copy()
    sanitized_confidence = confidence_grid.copy()

    changed = True
    while changed:
        changed = False

        for row in range(9):
            for digit in range(1, 10):
                matches = np.where(sanitized_grid[row, :] == digit)[0]
                if len(matches) <= 1:
                    continue

                row_confidences = sanitized_confidence[row, matches]
                keep_position = matches[int(np.argmax(row_confidences))]
                for position in matches:
                    if position == keep_position:
                        continue
                    sanitized_grid[row, position] = 0
                    sanitized_confidence[row, position] = 0.0
                    changed = True

        for col in range(9):
            for digit in range(1, 10):
                matches = np.where(sanitized_grid[:, col] == digit)[0]
                if len(matches) <= 1:
                    continue

                col_confidences = sanitized_confidence[matches, col]
                keep_position = matches[int(np.argmax(col_confidences))]
                for position in matches:
                    if position == keep_position:
                        continue
                    sanitized_grid[position, col] = 0
                    sanitized_confidence[position, col] = 0.0
                    changed = True

        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = sanitized_grid[box_row:box_row + 3, box_col:box_col + 3]
                box_conf = sanitized_confidence[box_row:box_row + 3, box_col:box_col + 3]

                for digit in range(1, 10):
                    positions = np.argwhere(box == digit)
                    if len(positions) <= 1:
                        continue

                    confidences = np.array([box_conf[r, c] for r, c in positions], dtype=np.float32)
                    keep_idx = int(np.argmax(confidences))

                    for idx, (r, c) in enumerate(positions):
                        if idx == keep_idx:
                            continue
                        sanitized_grid[box_row + r, box_col + c] = 0
                        sanitized_confidence[box_row + r, box_col + c] = 0.0
                        changed = True

    return sanitized_grid, sanitized_confidence


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
digit_history = [[deque(maxlen=DIGIT_SMOOTHING_WINDOW) for _ in range(9)] for _ in range(9)]
previous_grid = None
cached_solution = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray, binary, gpu_pipeline_active = preprocess_frame(frame, gpu_available)

    corners = find_grid_corners(gray, binary)
    if corners is None:
        corner_history.clear()
        for row in range(9):
            for col in range(9):
                digit_history[row][col].clear()
        previous_grid = None
        cached_solution = None
        cv2.putText(frame, "Grid not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Real-Time Sudoku Solver", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    corner_history.append(corners)
    smoothed_corners = np.mean(np.array(corner_history), axis=0).astype(np.float32)
    smoothed_perspective = cv2.getPerspectiveTransform(smoothed_corners, WARP_DESTINATION)
    inverse_matrix = cv2.getPerspectiveTransform(WARP_DESTINATION, smoothed_corners)
    warped_binary, gpu_warp_active = warp_binary_for_grid(binary, smoothed_perspective, gpu_pipeline_active)

    sudoku_digits = np.zeros((9, 9), dtype=int)
    sudoku_confidence = np.zeros((9, 9), dtype=np.float32)

    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE
            y2 = (row + 1) * CELL_SIZE
            x1 = col * CELL_SIZE
            x2 = (col + 1) * CELL_SIZE
            cell = warped_binary[y1:y2, x1:x2]
            predicted_digit, prediction_confidence = predict_digit(cell)
            digit_history[row][col].append((predicted_digit, prediction_confidence))
            smoothed_digit, smoothed_confidence = smooth_digit_prediction(digit_history[row][col])
            sudoku_digits[row, col] = smoothed_digit
            sudoku_confidence[row, col] = smoothed_confidence

    sudoku_digits, sudoku_confidence = sanitize_grid_by_confidence(sudoku_digits, sudoku_confidence)

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
                    solved_text = str(int(solved_grid[row, col]))
                    font_scale = 0.8
                    thickness = 2
                    text_size, baseline = cv2.getTextSize(solved_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_x = int(frame_point[0] - (text_size[0] / 2))
                    text_y = int(frame_point[1] + (text_size[1] / 2) - baseline)

                    cv2.putText(
                        frame,
                        solved_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                        cv2.LINE_AA,
                    )
    else:
        previous_grid = None
        cached_solution = None

    status_color = (0, 255, 0) if valid_clues else (0, 0, 255)
    status_text = f"Detected: {filled_cells} cells"
    if not valid_clues:
        status_text += " | invalid clues"
    if gpu_warp_active:
        status_text += " | GPU"
    else:
        status_text += " | CPU"

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow("Real-Time Sudoku Solver", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()   
cv2.destroyAllWindows()