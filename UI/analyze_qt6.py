"""PyQt6 desktop interface for PSD inference and inter-zone gap analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import ImageDraw
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from zone_analysis import STAINS, render_gap_overlay
from mask_pairing import build_mask_pairs, load_mask_key
from gap_band_analysis import make_positive_band_preview, measure_positive_gap_band


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = APP_DIR / "ckpt" / "best_model_ch64_dicepen.pth"


class InferenceWorker(QThread):
    output = pyqtSignal(str)
    completed = pyqtSignal(bool, str)

    def __init__(self, psd_root: Path, output_root: Path, checkpoint: Path) -> None:
        super().__init__()
        self.psd_root = psd_root
        self.output_root = output_root
        self.checkpoint = checkpoint

    def run(self) -> None:
        command = [
            sys.executable,
            str(APP_DIR / "inference.py"),
            "--psd_root", str(self.psd_root),
            "--output_dir", str(self.output_root),
            "--checkpoint", str(self.checkpoint),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=APP_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.output.emit(line.rstrip())
            return_code = process.wait()
            if return_code:
                self.completed.emit(False, f"Inference exited with status {return_code}")
            else:
                self.completed.emit(True, "Prediction completed")
        except Exception as exc:
            self.completed.emit(False, str(exc))


class ClickableImageLabel(QLabel):
    """Image label that reports clicks in unscaled pixmap coordinates."""

    image_clicked = pyqtSignal(int, int)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

    def mousePressEvent(self, event) -> None:
        pixmap = self.pixmap()
        if pixmap is not None:
            x = int(event.position().x())
            y = int(event.position().y())
            if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                self.image_clicked.emit(x, y)
        super().mousePressEvent(event)


class AnalysisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Knee Growth Plate Zone Analysis")
        self.resize(1320, 850)
        self.worker = None
        self.last_overlay = None
        self.last_sample = "analysis"
        self.background_checks = {}
        self.analysis_context = None
        self.selection_points = []
        self.current_roi = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.addWidget(self._build_prediction_box())

        splitter = QSplitter()
        splitter.addWidget(self._build_analysis_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setSizes([390, 900])
        outer.addWidget(splitter, 1)

    def _path_row(self, line_edit: QLineEdit, callback, button_text: str = "Browse…") -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton(button_text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    def _build_prediction_box(self) -> QGroupBox:
        box = QGroupBox("1. Prediction")
        layout = QGridLayout(box)
        self.psd_path = QLineEdit()
        self.psd_path.setPlaceholderText("Layered PSD file, or directory containing layered PSD files")
        self.output_path = QLineEdit()
        self.checkpoint_path = QLineEdit(str(DEFAULT_CHECKPOINT))

        layout.addWidget(QLabel("PSD input"), 0, 0)
        layout.addWidget(self._path_row(self.psd_path, self.choose_psd_directory), 0, 1)
        file_button = QPushButton("Choose PSD…")
        file_button.clicked.connect(self.choose_psd_file)
        layout.addWidget(file_button, 0, 2)
        layout.addWidget(QLabel("Save predictions"), 1, 0)
        layout.addWidget(self._path_row(self.output_path, self.choose_output_directory), 1, 1, 1, 2)
        layout.addWidget(QLabel("Checkpoint"), 2, 0)
        layout.addWidget(self._path_row(self.checkpoint_path, self.choose_checkpoint), 2, 1, 1, 2)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.run_prediction)
        layout.addWidget(self.predict_button, 3, 0)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label, 3, 1, 1, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(105)
        layout.addWidget(self.log, 4, 0, 1, 3)
        return box

    def _build_analysis_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        selection = QGroupBox("2. Compare predicted zones")
        form = QFormLayout(selection)
        self.sample_combo = QComboBox()
        self.sample_combo.currentTextChanged.connect(self._sample_changed)
        self.first_zone = QComboBox()
        self.second_zone = QComboBox()
        self.first_zone.addItems(STAINS)
        self.second_zone.addItems(STAINS)
        self.first_zone.setCurrentText("calcein")
        self.second_zone.setCurrentText("mineral")
        self.first_zone.currentTextChanged.connect(self._refresh_mask_pairs)
        self.second_zone.currentTextChanged.connect(self._refresh_mask_pairs)
        self.pair_combo = QComboBox()
        form.addRow("Sample", self.sample_combo)
        form.addRow("First zone", self.first_zone)
        form.addRow("Second zone", self.second_zone)
        form.addRow("Mask combination", self.pair_combo)
        layout.addWidget(selection)

        backgrounds = QGroupBox("Background signals")
        bg_grid = QGridLayout(backgrounds)
        for index, stain in enumerate(STAINS):
            check = QCheckBox(stain.capitalize())
            check.setChecked(stain == "mineral")
            self.background_checks[stain] = check
            bg_grid.addWidget(check, index // 3, index % 3)
        layout.addWidget(backgrounds)

        buttons = QHBoxLayout()
        analyze = QPushButton("Analyze gap")
        analyze.clicked.connect(self.analyze)
        self.save_button = QPushButton("Save image…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        self.reset_range_button = QPushButton("Reset range")
        self.reset_range_button.clicked.connect(self.reset_range)
        buttons.addWidget(analyze)
        buttons.addWidget(self.reset_range_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        metrics = QGroupBox("Thickness statistics (pixels)")
        metrics_form = QFormLayout(metrics)
        self.trimmed_mean_label = QLabel("—")
        self.std_label = QLabel("—")
        self.max_label = QLabel("—")
        self.range_label = QLabel("—")
        self.vertical_range_label = QLabel("—")
        metrics_form.addRow("10% trimmed mean", self.trimmed_mean_label)
        metrics_form.addRow("Standard deviation", self.std_label)
        metrics_form.addRow("Maximum", self.max_label)
        metrics_form.addRow("Horizontal range", self.range_label)
        metrics_form.addRow("Vertical range", self.vertical_range_label)
        layout.addWidget(metrics)

        note = QLabel(
            "Cyan and magenta show the selected predicted zones. Red shows the "
            "positive in-between gap band. Click two corners on the overlay to set "
            "both horizontal and vertical ranges; 10% is trimmed from each tail."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Overlay preview — click two corners to select a range"))
        self.preview = ClickableImageLabel("Run prediction and analyze two zones to see the overlay.")
        self.preview.image_clicked.connect(self._preview_clicked)
        self.preview.setMinimumSize(600, 500)
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, 3)
        layout.addWidget(QLabel("Subtracted positive band (> 0)"))
        self.band_preview = QLabel()
        self.band_preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        band_scroll = QScrollArea()
        band_scroll.setWidget(self.band_preview)
        band_scroll.setWidgetResizable(False)
        layout.addWidget(band_scroll, 2)
        return panel

    def choose_psd_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose PSD directory")
        if selected:
            self._set_psd_input(Path(selected))

    def choose_psd_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose a PSD file", filter="Photoshop (*.psd *.psb)")
        if selected:
            self._set_psd_input(Path(selected))

    def _set_psd_input(self, source: Path) -> None:
        self.psd_path.setText(str(source))
        parent = source.parent if source.is_file() else source
        self.output_path.setText(str(parent / "predictions"))

    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose prediction output directory")
        if selected:
            self.output_path.setText(selected)

    def choose_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint", filter="PyTorch checkpoint (*.pth *.pt)")
        if selected:
            self.checkpoint_path.setText(selected)

    def run_prediction(self) -> None:
        psd_root = Path(self.psd_path.text()).expanduser()
        output_root = Path(self.output_path.text()).expanduser()
        checkpoint = Path(self.checkpoint_path.text()).expanduser()
        if not psd_root.exists() or (psd_root.is_file() and psd_root.suffix.lower() not in {".psd", ".psb"}):
            self._error("Choose a valid layered PSD/PSB file or directory.")
            return
        if not checkpoint.is_file():
            self._error("Choose a valid model checkpoint.")
            return
        output_root.mkdir(parents=True, exist_ok=True)
        self.predict_button.setEnabled(False)
        self.status_label.setText("Running prediction…")
        self.log.clear()
        self.worker = InferenceWorker(psd_root, output_root, checkpoint)
        self.worker.output.connect(self.log.appendPlainText)
        self.worker.completed.connect(self._prediction_finished)
        self.worker.start()

    def _prediction_finished(self, success: bool, message: str) -> None:
        self.predict_button.setEnabled(True)
        self.status_label.setText(message)
        if success:
            self._populate_samples()
        else:
            self._error(message + "\n\nSee the prediction log for details.")

    def _populate_samples(self) -> None:
        root = Path(self.output_path.text()).expanduser()
        samples = sorted(
            directory.name for directory in root.iterdir()
            if directory.is_dir() and any(directory.glob("[1-9][a-h].png"))
        ) if root.is_dir() else []
        self.sample_combo.clear()
        self.sample_combo.addItems(samples)
        if not samples:
            self._error("Prediction finished, but no sample prediction folders were found.")

    def _sample_changed(self, sample: str) -> None:
        if sample:
            self.last_sample = sample
        self._refresh_mask_pairs()

    def _refresh_mask_pairs(self) -> None:
        self.pair_combo.clear()
        sample = self.sample_combo.currentText()
        if not sample:
            return
        prediction_dir = Path(self.output_path.text()).expanduser() / sample
        pairs = build_mask_pairs(
            prediction_dir, self.first_zone.currentText(), self.second_zone.currentText()
        )
        for first_key, second_key, category in pairs:
            label = f"{category.capitalize()}: {first_key} - {second_key}"
            self.pair_combo.addItem(label, (first_key, second_key))

    def analyze(self) -> None:
        sample = self.sample_combo.currentText()
        first_name = self.first_zone.currentText()
        second_name = self.second_zone.currentText()
        if not sample:
            self._error("Run prediction and choose a sample first.")
            return
        if first_name == second_name:
            self._error("Choose two different predicted zones.")
            return

        pair = self.pair_combo.currentData()
        if not pair:
            self._error("No compatible mask combinations are available for these zones.")
            return
        first_key, second_key = pair

        prediction_dir = Path(self.output_path.text()).expanduser() / sample
        signal_dir = Path(self.output_path.text()).expanduser() / "_formatted_input" / sample
        try:
            first_mask = load_mask_key(prediction_dir, first_key)
            second_mask = load_mask_key(prediction_dir, second_key)
        except Exception as exc:
            self._error(str(exc))
            return

        self.analysis_context = {
            "sample": sample,
            "first_key": first_key,
            "second_key": second_key,
            "first_mask": first_mask,
            "second_mask": second_mask,
            "signal_dir": signal_dir,
        }
        self.selection_points = []
        self.current_roi = None
        self._update_analysis(None)

    def _update_analysis(self, roi) -> bool:
        if self.analysis_context is None:
            return False
        context = self.analysis_context
        try:
            result = measure_positive_gap_band(
                context["first_mask"], context["second_mask"],
                context["first_key"], context["second_key"], roi=roi,
            )
            backgrounds = [
                name for name, check in self.background_checks.items() if check.isChecked()
            ]
            overlay = render_gap_overlay(
                context["signal_dir"], backgrounds, context["first_mask"],
                context["second_mask"], result.gap_mask,
            )
        except Exception as exc:
            self._error(str(exc))
            return False

        height, width = result.gap_mask.shape
        if roi is None:
            selected_x = (0, width - 1)
            selected_y = (0, height - 1)
        else:
            x1, y1, x2, y2 = roi
            selected_x = tuple(sorted((x1, x2)))
            selected_y = tuple(sorted((y1, y2)))
            draw = ImageDraw.Draw(overlay)
            draw.rectangle((selected_x[0], selected_y[0], selected_x[1], selected_y[1]), outline=(255, 255, 0), width=2)

        band = make_positive_band_preview(result.gap_mask)
        self.trimmed_mean_label.setText(f"{result.trimmed_mean:.2f}")
        self.std_label.setText(f"{result.standard_deviation:.2f}")
        self.max_label.setText(f"{result.maximum:.0f}")
        self.range_label.setText(
            f"x = {selected_x[0]}–{selected_x[1]} "
            f"({result.thicknesses.size} positive-band columns)"
        )
        self.vertical_range_label.setText(f"y = {selected_y[0]}–{selected_y[1]}")
        self.last_overlay = overlay
        self.last_sample = f"{context['sample']}_{context['first_key']}_{context['second_key']}"
        self.current_roi = roi
        self.preview.setPixmap(QPixmap.fromImage(ImageQt(overlay)))
        self.preview.adjustSize()
        self.band_preview.setPixmap(QPixmap.fromImage(ImageQt(band)))
        self.band_preview.adjustSize()
        self.save_button.setEnabled(True)
        return True

    def _preview_clicked(self, x: int, y: int) -> None:
        if self.analysis_context is None:
            return
        if not self.selection_points:
            self.selection_points = [(x, y)]
            marker = self.last_overlay.copy()
            draw = ImageDraw.Draw(marker)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 0), width=2)
            self.preview.setPixmap(QPixmap.fromImage(ImageQt(marker)))
            self.status_label.setText(f"First range corner: ({x}, {y}); click the opposite corner")
            return
        x1, y1 = self.selection_points[0]
        roi = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
        self.selection_points = []
        if self._update_analysis(roi):
            self.status_label.setText(f"Selected range: x={roi[0]}–{roi[2]}, y={roi[1]}–{roi[3]}")

    def reset_range(self) -> None:
        if self.analysis_context is None:
            return
        self.selection_points = []
        if self._update_analysis(None):
            self.status_label.setText("Range reset to the full image")

    def save_image(self) -> None:
        if self.last_overlay is None:
            return
        default = Path(self.output_path.text()).expanduser() / f"{self.last_sample}_gap_overlay.png"
        selected, _ = QFileDialog.getSaveFileName(self, "Save overlay", str(default), "PNG image (*.png);;TIFF image (*.tif *.tiff)")
        if selected:
            self.last_overlay.save(selected)
            self.status_label.setText(f"Saved {selected}")

    def _error(self, text: str) -> None:
        QMessageBox.critical(self, "Zone analysis", text)


def main() -> None:
    app = QApplication(sys.argv)
    window = AnalysisWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
