"""Simple TinyNAFNet GUI with before/after preview."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from model import ModelConfig, build_model
from utils import image_files, load_array, save_array_or_image


def array_to_pixmap(
    array: np.ndarray,
    auto_range: bool = False,
) -> QPixmap:
    """Convert grayscale NumPy array to Qt pixmap."""

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D array, "
            f"got {array.shape}"
        )

    if auto_range:
        minimum = float(array.min())
        maximum = float(array.max())

        if maximum > minimum:
            array = (
                array - minimum
            ) / (
                maximum - minimum
            )
    else:
        array = np.clip(
            array,
            0.0,
            1.0,
        )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    array = (
        np.clip(array, 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)

    height, width = array.shape

    image = QImage(
        array.data,
        width,
        height,
        width,
        QImage.Format_Grayscale8,
    ).copy()

    return QPixmap.fromImage(
        image
    )


def tensor_to_pixmap(
    tensor: torch.Tensor,
) -> QPixmap:
    """Convert [1,H,W] tensor to preview."""
    array = (
        tensor.detach()
        .float()
        .squeeze()
        .cpu()
        .numpy()
    )

    return array_to_pixmap(
        array
    )


def load_model(
    weights: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load model using checkpoint configuration."""

    checkpoint = torch.load(
        weights,
        map_location=device,
        weights_only=False,
    )

    state = checkpoint.get(
        "model",
        checkpoint,
    )

    config = checkpoint.get(
        "model_config",
        {},
    )

    model_config = ModelConfig(
        width=int(
            config.get("width", 28)
        ),
        enc_blocks=tuple(
            config.get(
                "enc_blocks",
                (1, 1, 1, 1),
            )
        ),
        dec_blocks=tuple(
            config.get(
                "dec_blocks",
                (1, 1, 1, 1),
            )
        ),
        middle_blocks=int(
            config.get(
                "middle_blocks",
                2,
            )
        ),
        scale=int(
            config.get(
                "scale",
                2,
            )
        ),
        dw_expand=int(
            config.get(
                "dw_expand",
                2,
            )
        ),
        ffn_expand=int(
            config.get(
                "ffn_expand",
                2,
            )
        ),
    )

    model = build_model(
        model_config
    ).to(device)

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    if device.type == "cuda":
        model = model.to(
            memory_format=torch.channels_last
        )

    return model


class Worker(QObject):
    """Background inference worker."""

    progress = Signal(int)
    current_file = Signal(str)
    statistics = Signal(float, float, int)
    input_preview = Signal(QPixmap)
    output_preview = Signal(QPixmap)
    log_message = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        weights: Path,
        input_dir: Path,
        output_dir: Path,
        batch_size: int,
        device_name: str,
    ) -> None:
        super().__init__()

        self.weights = weights
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device_name = device_name

    def run(self) -> None:
        try:
            device = torch.device(
                self.device_name
            )

            if (
                device.type == "cuda"
                and not torch.cuda.is_available()
            ):
                raise RuntimeError(
                    "CUDA is not available."
                )

            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True

            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            files = sorted(
                p
                for p in image_files(
                    self.input_dir
                )
                if p.suffix.lower()
                == ".npy"
            )

            if not files:
                raise RuntimeError(
                    f"No .npy files found in "
                    f"{self.input_dir}"
                )

            self.log_message.emit(
                f"Found {len(files)} .npy files."
            )

            model = load_model(
                self.weights,
                device,
            )

            self.log_message.emit(
                "Model loaded successfully."
            )

            processed = 0
            total = len(files)

            total_forward = 0.0
            overall_start = time.perf_counter()

            first_preview_done = False

            for start in range(
                0,
                total,
                self.batch_size,
            ):
                batch_paths = files[
                    start : start
                    + self.batch_size
                ]

                tensors = []
                bits_list = []

                for path in batch_paths:
                    tensor, _, bits = load_array(
                        path
                    )

                    tensors.append(
                        tensor
                    )

                    bits_list.append(bits)

                x = torch.stack(
                    tensors
                ).to(
                    device,
                    non_blocking=True,
                )

                if device.type == "cuda":
                    x = x.to(
                        memory_format=torch.channels_last
                    )
                    torch.cuda.synchronize()

                start_forward = (
                    time.perf_counter()
                )

                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                        ):
                            y = model(x)
                    else:
                        y = model(x)

                if device.type == "cuda":
                    torch.cuda.synchronize()

                total_forward += (
                    time.perf_counter()
                    - start_forward
                )

                y = y.float().clamp(
                    0.0,
                    1.0,
                )

                if not first_preview_done:
                    self.input_preview.emit(
                        tensor_to_pixmap(
                            x[0].cpu()
                        )
                    )

                    self.output_preview.emit(
                        tensor_to_pixmap(
                            y[0].cpu()
                        )
                    )

                    first_preview_done = True

                for index, path in enumerate(
                    batch_paths
                ):
                    output_path = (
                        self.output_dir
                        / path.name
                    )

                    save_array_or_image(
                        y[index],
                        output_path,
                        bits_list[index],
                    )

                    processed += 1

                    self.current_file.emit(
                        path.name
                    )

                elapsed = (
                    time.perf_counter()
                    - overall_start
                )

                throughput = (
                    processed
                    / max(
                        elapsed,
                        1e-9,
                    )
                )

                ms_per_image = (
                    1000.0
                    * total_forward
                    / max(
                        processed,
                        1,
                    )
                )

                self.statistics.emit(
                    throughput,
                    ms_per_image,
                    processed,
                )

                self.progress.emit(
                    int(
                        processed
                        * 100
                        / total
                    )
                )

            self.log_message.emit(
                "Inference completed."
            )

            self.finished.emit()

        except Exception as exc:
            self.failed.emit(
                f"{type(exc).__name__}: {exc}"
            )


class MainWindow(QMainWindow):
    """Main GUI."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "TinyNAFNet"
        )

        self.resize(
            750,
            550,
        )

        self.thread: QThread | None = None
        self.worker: Worker | None = None

        self.build_ui()

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(
            central
        )

        layout = QVBoxLayout(
            central
        )

        # ---------------------------------------------------------
        # Settings
        # ---------------------------------------------------------

        settings = QGroupBox(
            "Model and Folders"
        )

        form = QFormLayout(
            settings
        )

        self.model_edit = QLineEdit(
            "weights/best.pt"
        )

        model_button = QPushButton(
            "Browse"
        )

        model_button.clicked.connect(
            self.browse_model
        )

        model_row = QHBoxLayout()
        model_row.addWidget(
            self.model_edit
        )
        model_row.addWidget(
            model_button
        )

        form.addRow(
            "Model:",
            model_row,
        )

        self.input_edit = QLineEdit()

        input_button = QPushButton(
            "Browse"
        )

        input_button.clicked.connect(
            self.browse_input
        )

        input_row = QHBoxLayout()
        input_row.addWidget(
            self.input_edit
        )
        input_row.addWidget(
            input_button
        )

        form.addRow(
            "Input:",
            input_row,
        )

        self.output_edit = QLineEdit(
            "results_gui"
        )

        output_button = QPushButton(
            "Browse"
        )

        output_button.clicked.connect(
            self.browse_output
        )

        output_row = QHBoxLayout()
        output_row.addWidget(
            self.output_edit
        )
        output_row.addWidget(
            output_button
        )

        form.addRow(
            "Output:",
            output_row,
        )

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(
            1,
            128,
        )
        self.batch_spin.setValue(
            16
        )

        form.addRow(
            "Batch size:",
            self.batch_spin,
        )

        layout.addWidget(
            settings
        )

        # ---------------------------------------------------------
        # Buttons
        # ---------------------------------------------------------

        buttons = QHBoxLayout()

        self.start_button = QPushButton(
            "Start"
        )

        self.start_button.clicked.connect(
            self.start
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.stop_button.clicked.connect(
            self.stop
        )

        self.stop_button.setEnabled(
            False
        )

        self.open_button = QPushButton(
            "Open Output Folder"
        )

        self.open_button.clicked.connect(
            self.open_output
        )

        buttons.addWidget(
            self.start_button
        )
        buttons.addWidget(
            self.stop_button
        )
        buttons.addWidget(
            self.open_button
        )
        buttons.addStretch()

        layout.addLayout(
            buttons
        )

        # ---------------------------------------------------------
        # Progress
        # ---------------------------------------------------------

        self.progress = QProgressBar()

        layout.addWidget(
            self.progress
        )

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        stats = QGroupBox(
            "Statistics"
        )

        stats_layout = QFormLayout(
            stats
        )

        self.file_label = QLabel("-")
        self.count_label = QLabel("0")
        self.throughput_label = QLabel(
            "0.00 images/sec"
        )
        self.time_label = QLabel(
            "0.00 ms/image"
        )

        stats_layout.addRow(
            "Current file:",
            self.file_label,
        )

        stats_layout.addRow(
            "Processed:",
            self.count_label,
        )

        stats_layout.addRow(
            "Throughput:",
            self.throughput_label,
        )

        stats_layout.addRow(
            "Forward time:",
            self.time_label,
        )

        layout.addWidget(
            stats
        )

        # ---------------------------------------------------------
        # Before / After
        # ---------------------------------------------------------

        preview_group = QGroupBox(
            "Before / After Preview"
        )

        preview_layout = QHBoxLayout(
            preview_group
        )

        # Before
        before_layout = QVBoxLayout()

        before_title = QLabel(
            "Degraded Input"
        )

        before_title.setAlignment(
            Qt.AlignCenter
        )

        self.before_label = QLabel(
            "No preview"
        )

        self.before_label.setAlignment(
            Qt.AlignCenter
        )

        self.before_label.setMinimumSize(
            300,
            250,
        )

        self.before_label.setStyleSheet(
            "border: 1px solid #888;"
        )

        before_layout.addWidget(
            before_title
        )

        before_layout.addWidget(
            self.before_label
        )

        # After
        after_layout = QVBoxLayout()

        after_title = QLabel(
            "Restored Output"
        )

        after_title.setAlignment(
            Qt.AlignCenter
        )

        self.after_label = QLabel(
            "No preview"
        )

        self.after_label.setAlignment(
            Qt.AlignCenter
        )

        self.after_label.setMinimumSize(
            300,
            250,
        )

        self.after_label.setStyleSheet(
            "border: 1px solid #888;"
        )

        after_layout.addWidget(
            after_title
        )

        after_layout.addWidget(
            self.after_label
        )

        preview_layout.addLayout(
            before_layout
        )

        preview_layout.addLayout(
            after_layout
        )

        layout.addWidget(
            preview_group
        )

        # ---------------------------------------------------------
        # Log
        # ---------------------------------------------------------

        log_group = QGroupBox(
            "Log"
        )

        log_layout = QVBoxLayout(
            log_group
        )

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        log_layout.addWidget(
            self.log
        )

        layout.addWidget(
            log_group
        )

    # -------------------------------------------------------------
    # Browsers
    # -------------------------------------------------------------

    def browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "PyTorch Checkpoint (*.pt)",
        )

        if path:
            self.model_edit.setText(
                path
            )

    def browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Folder",
        )

        if path:
            self.input_edit.setText(
                path
            )

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
        )

        if path:
            self.output_edit.setText(
                path
            )

    # -------------------------------------------------------------
    # Start
    # -------------------------------------------------------------

    def start(self) -> None:
        weights = Path(
            self.model_edit.text().strip()
        )

        input_dir = Path(
            self.input_edit.text().strip()
        )

        output_dir = Path(
            self.output_edit.text().strip()
        )

        if not weights.exists():
            QMessageBox.warning(
                self,
                "Error",
                f"Model not found:\n{weights}",
            )
            return

        if not input_dir.exists():
            QMessageBox.warning(
                self,
                "Error",
                f"Input folder not found:\n{input_dir}",
            )
            return

        if not image_files(
            input_dir
        ):
            QMessageBox.warning(
                self,
                "Error",
                "No supported input files found.",
            )
            return

        self.progress.setValue(
            0
        )

        self.log.clear()

        self.thread = QThread()

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.worker = Worker(
            weights,
            input_dir,
            output_dir,
            self.batch_spin.value(),
            device,
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.progress.connect(
            self.progress.setValue
        )

        self.worker.current_file.connect(
            self.file_label.setText
        )

        self.worker.statistics.connect(
            self.update_statistics
        )

        self.worker.input_preview.connect(
            self.show_before
        )

        self.worker.output_preview.connect(
            self.show_after
        )

        self.worker.log_message.connect(
            self.log.appendPlainText
        )

        self.worker.finished.connect(
            self.finished
        )

        self.worker.failed.connect(
            self.failed
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.failed.connect(
            self.thread.quit
        )

        self.thread.start()

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            True
        )

    # -------------------------------------------------------------
    # Stop
    # -------------------------------------------------------------

    def stop(self) -> None:
        self.log.appendPlainText(
            "Stop requested."
        )

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        if self.thread is not None:
            self.thread.requestInterruption()

    # -------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------

    def show_before(
        self,
        pixmap: QPixmap,
    ) -> None:
        self.before_label.setPixmap(
            pixmap.scaled(
                self.before_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def show_after(
        self,
        pixmap: QPixmap,
    ) -> None:
        self.after_label.setPixmap(
            pixmap.scaled(
                self.after_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    def update_statistics(
        self,
        throughput: float,
        ms_per_image: float,
        processed: int,
    ) -> None:
        self.count_label.setText(
            str(processed)
        )

        self.throughput_label.setText(
            f"{throughput:.2f} images/sec"
        )

        self.time_label.setText(
            f"{ms_per_image:.2f} ms/image"
        )

    # -------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------

    def finished(self) -> None:
        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        self.progress.setValue(
            100
        )

        QMessageBox.information(
            self,
            "Complete",
            "Inference completed successfully.",
        )

    def failed(
        self,
        message: str,
    ) -> None:
        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        QMessageBox.critical(
            self,
            "Inference Error",
            message,
        )

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    def open_output(self) -> None:
        import os

        folder = Path(
            self.output_edit.text().strip()
        )

        if not folder.exists():
            QMessageBox.information(
                self,
                "Output",
                "Output folder does not exist yet.",
            )
            return

        os.startfile(
            str(folder)
        )


def main() -> None:
    app = QApplication(
        sys.argv
    )

    window = MainWindow()
    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()