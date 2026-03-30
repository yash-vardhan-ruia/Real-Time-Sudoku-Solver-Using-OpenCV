from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

MODEL_IMAGE_SIZE = 28
SAMPLES_PER_DIGIT = 1200
EMPTY_SAMPLES = 1800
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_OUTPUT_PATH = Path(__file__).with_name("digit_cnn.onnx")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_printed_digit_dataset(samples_per_digit=SAMPLES_PER_DIGIT, empty_samples=EMPTY_SAMPLES):
    rng = np.random.default_rng(42)
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]

    images = []
    labels = []

    for digit in range(1, 10):
        text = str(digit)
        for _ in range(samples_per_digit):
            canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.uint8)

            font = fonts[int(rng.integers(0, len(fonts)))]
            font_scale = float(rng.uniform(0.75, 1.30))
            thickness = int(rng.integers(1, 4))

            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size

            jitter_x = int(rng.integers(-4, 5))
            jitter_y = int(rng.integers(-4, 5))
            x = max(0, (MODEL_IMAGE_SIZE - text_w) // 2 + jitter_x)
            y = min(MODEL_IMAGE_SIZE - 1, (MODEL_IMAGE_SIZE + text_h) // 2 + jitter_y)

            cv2.putText(canvas, text, (x, y), font, font_scale, 255, thickness, cv2.LINE_AA)

            angle = float(rng.uniform(-18, 18))
            scale = float(rng.uniform(0.88, 1.14))
            tx = float(rng.uniform(-3.0, 3.0))
            ty = float(rng.uniform(-3.0, 3.0))

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

            noise = rng.normal(0.0, 14.0, size=canvas.shape).astype(np.float32)
            canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            images.append(canvas)
            labels.append(digit)

    for _ in range(empty_samples):
        canvas = np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), dtype=np.uint8)

        if rng.random() < 0.7:
            line_count = int(rng.integers(1, 5))
            for _ in range(line_count):
                x1 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                y1 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                x2 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                y2 = int(rng.integers(0, MODEL_IMAGE_SIZE))
                color = int(rng.integers(20, 130))
                thickness = int(rng.integers(1, 3))
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

        noise = rng.normal(0.0, 18.0, size=canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        images.append(canvas)
        labels.append(0)

    images = np.asarray(images, dtype=np.float32) / 255.0
    labels = np.asarray(labels, dtype=np.int64)
    images = np.expand_dims(images, axis=1)

    return images, labels


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == targets).sum().item())
            total += int(targets.numel())

    if total == 0:
        return 0.0
    return correct / total


def train_model():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    images, labels = generate_printed_digit_dataset()

    tensor_x = torch.from_numpy(images)
    tensor_y = torch.from_numpy(labels)
    dataset = TensorDataset(tensor_x, tensor_y)

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * float(inputs.size(0))

        epoch_loss = running_loss / float(train_size)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    dummy_input = torch.randn(1, 1, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)

    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(MODEL_OUTPUT_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"ONNX model exported to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_model()
