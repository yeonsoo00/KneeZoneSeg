"""PyQt6 desktop interface for PSD inference and inter-zone gap analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QThread, pyqtSignal
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

from zone_analysis import STAINS, load_zone_mask, measure_vertical_gap, render_gap_overlay


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


class AnalysisWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Knee Growth Plate Zone Analysis")
        self.resize(1320, 850)
        self.worker = None
        self.last_overlay = None
        self.last_sample = "analysis"
        self.background_checks = {}

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
        self.psd_path.setPlaceholderText("Directory containing the nine numbered PSD files")
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
        form.addRow("Sample", self.sample_combo)
        form.addRow("First zone", self.first_zone)
        form.addRow("Second zone", self.second_zone)
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
        buttons.addWidget(analyze)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        metrics = QGroupBox("Thickness statistics (pixels)")
        metrics_form = QFormLayout(metrics)
        self.trimmed_mean_label = QLabel("—")
        self.std_label = QLabel("—")
        self.max_label = QLabel("—")
        self.range_label = QLabel("—")
        metrics_form.addRow("10% trimmed mean", self.trimmed_mean_label)
        metrics_form.addRow("Standard deviation", self.std_label)
        metrics_form.addRow("Maximum", self.max_label)
        metrics_form.addRow("Horizontal range", self.range_label)
        layout.addWidget(metrics)

        note = QLabel(
            "Cyan and magenta show the selected predicted zones. Red shows the "
            "in-between gap. Thickness is measured vertically in columns occupied "
            "by both zones; 10% is trimmed from each tail for the mean."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Overlay preview"))
        self.preview = QLabel("Run prediction and analyze two zones to see the overlay.")
        self.preview.setAlignment(self.preview.alignment())
        self.preview.setMinimumSize(600, 500)
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        return panel

    def choose_psd_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose PSD directory")
        if selected:
            self._set_psd_input(Path(selected))

    def choose_psd_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Choose a PSD file", filter="Photoshop (*.psd *.psb)")
        if selected:
            self._set_psd_input(Path(selected).parent)

    def _set_psd_input(self, directory: Path) -> None:
        self.psd_path.setText(str(directory))
        self.output_path.setText(str(directory / "predictions"))

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
        if not psd_root.is_dir():
            self._error("Choose a valid PSD directory.")
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

        prediction_dir = Path(self.output_path.text()).expanduser() / sample
        signal_dir = Path(self.output_path.text()).expanduser() / "_formatted_input" / sample
        try:
            first_mask = load_zone_mask(prediction_dir, first_name)
            second_mask = load_zone_mask(prediction_dir, second_name)
            result = measure_vertical_gap(first_mask, second_mask, first_name, second_name)
            backgrounds = [name for name, check in self.background_checks.items() if check.isChecked()]
            overlay = render_gap_overlay(
                signal_dir, backgrounds, first_mask, second_mask, result.gap_mask
            )
        except Exception as exc:
            self._error(str(exc))
            return

        self.trimmed_mean_label.setText(f"{result.trimmed_mean:.2f}")
        self.std_label.setText(f"{result.standard_deviation:.2f}")
        self.max_label.setText(f"{result.maximum:.0f}")
        self.range_label.setText(
            f"x = {result.x_start}–{result.x_end} "
            f"({result.thicknesses.size} shared columns; {result.upper_zone} above {result.lower_zone})"
        )
        self.last_overlay = overlay
        self.last_sample = sample
        self.preview.setPixmap(QPixmap.fromImage(ImageQt(overlay)))
        self.preview.adjustSize()
        self.save_button.setEnabled(True)

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
