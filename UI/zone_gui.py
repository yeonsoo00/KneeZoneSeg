"""PyQt6 interface for final-model inference and five-zone analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zone_metrics import (
    DEFAULT_PIXEL_SIZE_UM,
    SIDE_EXCLUSION,
    STAINS,
    ZONE_DEFINITIONS,
    analyze_zones,
    export_zone_csv,
)
from zone_visualization import (
    render_all_zones,
    render_zone_measurement,
    save_zone_visualizations,
)


APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR.parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "Prediction"
DEFAULT_OUTPUT_PLACEHOLDER = "KneeZoneSeg/Prediction"
DEFAULT_CHECKPOINT_PLACEHOLDER = "ckpt/best_model_ch48_super.pth"
LAYERED_SUFFIXES = frozenset({".psd", ".psb", ".pxd"})


def default_checkpoint() -> Path:
    candidates = (
        PACKAGE_DIR / "ckpt" / "best_model_ch48_super.pth",
        PACKAGE_DIR.parent
        / "SemiSeparatedUnetFusion"
        / "ckpt"
        / "groupnorm_deepsuper"
        / "best_model_ch48_super.pth",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_CHECKPOINT = default_checkpoint()


def _path_ends_with(path: Path, suffix: Path) -> bool:
    """Return whether path ends with every component in suffix."""
    return (
        len(path.parts) >= len(suffix.parts)
        and path.parts[-len(suffix.parts) :] == suffix.parts
    )


class InferenceWorker(QThread):
    output = pyqtSignal(str)
    completed = pyqtSignal(bool, str)

    def __init__(self, input_root: Path, output_root: Path, checkpoint: Path) -> None:
        super().__init__()
        self.input_root = input_root
        self.output_root = output_root
        self.checkpoint = checkpoint

    def run(self) -> None:
        command = [
            sys.executable,
            str(PACKAGE_DIR / "inference.py"),
            "--input_root",
            str(self.input_root),
            "--output_dir",
            str(self.output_root),
            "--checkpoint",
            str(self.checkpoint),
            "--base_ch",
            "48",
            "--use_pyramid_agg",
            "--use_cross_attention",
            "--norm",
            "group",
            "--group_norm_groups",
            "8",
            "--deep_supervision",
            "--sfo_hsv_channels",
            "1",
            "--img_size",
            "512",
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=PACKAGE_DIR,
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
                self.completed.emit(
                    False, f"Inference exited with status {return_code}"
                )
            else:
                self.completed.emit(True, "Prediction completed")
        except Exception as exc:
            self.completed.emit(False, str(exc))


class AnalysisWindow(QMainWindow):
    """Run final-model inference and analyze the five fixed GP zones."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Knee Growth Plate Five-Zone Analysis")
        self.resize(1480, 900)
        self.worker = None
        self.results = None
        self.signal_path = None
        self.visualizations = {}
        self._preview_pixmap = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.addWidget(self._build_prediction_box())
        splitter = QSplitter()
        splitter.addWidget(self._build_analysis_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setSizes([790, 690])
        outer.addWidget(splitter, 1)

    def _path_row(
        self, line_edit: QLineEdit, callback, text: str = "Browse…"
    ) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        button = QPushButton(text)
        button.setFixedWidth(button.sizeHint().width())
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    def _build_prediction_box(self) -> QGroupBox:
        box = QGroupBox("1. Final-model prediction")
        layout = QGridLayout(box)
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText(
            "PSD/PSB/PXD file, or recursive directory of layered files and JPG/PNG sample folders"
        )
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText(DEFAULT_OUTPUT_PLACEHOLDER)
        self.output_path.editingFinished.connect(self._populate_samples)
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText(DEFAULT_CHECKPOINT_PLACEHOLDER)

        layout.addWidget(QLabel("Input"), 0, 0)
        layout.addWidget(
            self._path_row(self.input_path, self.choose_input_directory), 0, 1
        )
        file_button = QPushButton("Browse…")
        file_button.setFixedWidth(file_button.sizeHint().width())
        file_button.setToolTip("Choose one layered PSD, PSB, or PXD file")
        file_button.clicked.connect(self.choose_input_file)
        layout.addWidget(file_button, 0, 2)
        layout.addWidget(QLabel("Prediction output"), 1, 0)
        layout.addWidget(
            self._path_row(self.output_path, self.choose_output_directory), 1, 1, 1, 2
        )
        layout.addWidget(QLabel("Final checkpoint"), 2, 0)
        layout.addWidget(
            self._path_row(self.checkpoint_path, self.choose_checkpoint), 2, 1, 1, 2
        )

        self.predict_button = QPushButton("Predict with best_model_ch48_super.pth")
        self.predict_button.clicked.connect(self.run_prediction)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.predict_button, 3, 0, 1, 2)
        layout.addWidget(self.status_label, 3, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log, 4, 0, 1, 3)
        return box

    def _build_analysis_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        settings = QGroupBox("2. Five-zone analysis")
        form = QFormLayout(settings)
        self.sample_combo = QComboBox()
        self.sample_combo.currentTextChanged.connect(self._sample_changed)
        self.background_combo = QComboBox()
        self.background_combo.addItems(STAINS)
        self.background_combo.setCurrentText("sfo")
        self.background_combo.currentTextChanged.connect(self._clear_analysis)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.0001, 1000.0)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setValue(DEFAULT_PIXEL_SIZE_UM)
        self.pixel_size.setSuffix(" µm/px")
        form.addRow("Sample", self.sample_combo)
        form.addRow("Analysis background", self.background_combo)
        form.addRow("Pixel size", self.pixel_size)
        layout.addWidget(settings)

        definitions = ", ".join(
            f"{name}: {first}–{second}"
            for name, (first, second) in ZONE_DEFINITIONS.items()
        )
        note = QLabel(
            f"{definitions}. The common support of each boundary pair is trimmed by "
            f"{SIDE_EXCLUSION:.0%} at the left and {SIDE_EXCLUSION:.0%} at the right. "
            "No manual range reset and no gap analysis are used."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        action_row = QHBoxLayout()
        analyze_button = QPushButton("Analyze five zones")
        analyze_button.clicked.connect(self.analyze)
        self.csv_button = QPushButton("Export measurements to CSV…")
        self.csv_button.setEnabled(False)
        self.csv_button.clicked.connect(self.export_csv)
        self.visualization_button = QPushButton("Save zone visualizations…")
        self.visualization_button.setEnabled(False)
        self.visualization_button.clicked.connect(self.save_visualizations)
        action_row.addWidget(analyze_button)
        action_row.addWidget(self.csv_button)
        action_row.addWidget(self.visualization_button)
        layout.addLayout(action_row)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Zone",
                "Boundaries",
                "Area px²",
                "Width px",
                "Mean px",
                "Median px",
                "SD px",
                "Min–max px",
                "Mean signal",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        row = QHBoxLayout()
        row.addWidget(QLabel("Visualization"))
        self.visualization_combo = QComboBox()
        self.visualization_combo.currentTextChanged.connect(self._show_visualization)
        row.addWidget(self.visualization_combo, 1)
        layout.addLayout(row)
        self.preview = QLabel(
            "Analyze a sample to see the total five-zone contour visualization."
        )
        self.preview.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, 1)
        return panel

    def choose_input_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose recursive input directory"
        )
        if selected:
            self._set_input(Path(selected))

    def choose_input_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a layered PSD, PSB, or PXD",
            filter="Layered images (*.psd *.psb *.pxd)",
        )
        if selected:
            self._set_input(Path(selected))

    def _set_input(self, source: Path) -> None:
        self.input_path.setText(str(source))

    def _output_root(self) -> Path:
        text = self.output_path.text().strip()
        return Path(text).expanduser() if text else DEFAULT_OUTPUT_DIR

    def _checkpoint(self) -> Path:
        text = self.checkpoint_path.text().strip()
        return Path(text).expanduser() if text else DEFAULT_CHECKPOINT

    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Choose prediction output directory"
        )
        if selected:
            self.output_path.setText(selected)
            self._populate_samples()

    def choose_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose checkpoint", filter="PyTorch checkpoint (*.pth *.pt)"
        )
        if selected:
            self.checkpoint_path.setText(selected)

    def run_prediction(self) -> None:
        input_text = self.input_path.text().strip()
        input_root = Path(input_text).expanduser()
        output_root = self._output_root()
        checkpoint = self._checkpoint()
        if (
            not input_text
            or not input_root.exists()
            or (
                input_root.is_file()
                and input_root.suffix.casefold() not in LAYERED_SUFFIXES
            )
        ):
            self._error(
                "Choose a valid PSD/PSB/PXD file or a directory containing "
                "layered files and/or JPG/PNG sample folders."
            )
            return
        if not checkpoint.is_file():
            self._error(
                "The default ckpt/best_model_ch48_super.pth checkpoint was not "
                "found. Copy it there or choose another checkpoint."
            )
            return
        output_root.mkdir(parents=True, exist_ok=True)
        self.predict_button.setEnabled(False)
        self.status_label.setText("Running prediction…")
        self.log.clear()
        self.worker = InferenceWorker(input_root, output_root, checkpoint)
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
        root = self._output_root()
        samples = (
            sorted(
                directory.relative_to(root).as_posix()
                for directory in root.rglob("*")
                if directory.is_dir()
                and all(
                    (directory / f"{key}.png").is_file()
                    for key in ("1c", "1d", "6a", "6b", "6c", "6d", "9a", "9d")
                )
            )
            if root.is_dir()
            else []
        )
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItems(samples)
        self.sample_combo.blockSignals(False)
        self._sample_changed(self.sample_combo.currentText())
        self.status_label.setText(f"Loaded {len(samples)} prediction sample(s)")
        if not samples:
            self._error(
                "No complete prediction folders were found in that output directory."
            )

    def _sample_changed(self, _sample: str) -> None:
        self._clear_analysis()

    def _clear_analysis(self, *_args) -> None:
        self.results = None
        self.signal_path = None
        self.visualizations.clear()
        self._preview_pixmap = None
        self.table.setRowCount(0)
        self.visualization_combo.clear()
        self.preview.clear()
        self.preview.setText(
            "Analyze a sample to see the total five-zone contour visualization."
        )
        self.csv_button.setEnabled(False)
        self.visualization_button.setEnabled(False)

    def analyze(self) -> None:
        sample = self.sample_combo.currentText()
        if not sample:
            self._error("Run prediction first.")
            return
        output_root = self._output_root()
        prediction_dir = output_root / sample
        self.signal_path = (
            output_root
            / "_formatted_input"
            / sample
            / f"{self.background_combo.currentText()}.png"
        )
        try:
            self.results = analyze_zones(
                prediction_dir,
                self.signal_path,
                side_exclusion=SIDE_EXCLUSION,
                pixel_size_um=self.pixel_size.value(),
            )
            self.visualizations = {
                "All zones": render_all_zones(self.signal_path, self.results)
            }
            for name, result in self.results.items():
                self.visualizations[name] = render_zone_measurement(
                    self.signal_path, result
                )
        except Exception as exc:
            self._error(str(exc))
            return
        self._fill_table()
        self.visualization_combo.addItems(self.visualizations)
        self.visualization_combo.setCurrentText("All zones")
        self._show_visualization("All zones")
        self.csv_button.setEnabled(True)
        self.visualization_button.setEnabled(True)
        self.status_label.setText("Five-zone analysis completed using the central 90%")

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self.results))
        for row, result in enumerate(self.results.values()):
            values = (
                result.name,
                f"{result.boundary_1}–{result.boundary_2}",
                f"{result.area_px:,}",
                str(result.width_px),
                f"{result.mean_thickness_px:.2f}",
                f"{result.median_thickness_px:.2f}",
                f"{result.std_thickness_px:.2f}",
                f"{result.min_thickness_px:.0f}–{result.max_thickness_px:.0f}",
                f"{result.mean_signal_gray:.2f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))

    def _show_visualization(self, name: str) -> None:
        image = self.visualizations.get(name)
        if image is None:
            return
        rgb = image.convert("RGB")
        raw = rgb.tobytes("raw", "RGB")
        qimage = QImage(
            raw,
            rgb.width,
            rgb.height,
            rgb.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        # Keep a persistent pixmap owned by the window. This prevents the Qt
        # preview from retaining a temporary Pillow buffer after a save dialog.
        self._preview_pixmap = QPixmap.fromImage(qimage)
        self.preview.setPixmap(self._preview_pixmap)
        self.preview.adjustSize()

    def export_csv(self) -> None:
        if not self.results:
            return
        sample = self.sample_combo.currentText()
        default = (
            self._output_root() / "Zone_analysis" / sample / "zone_measurements.csv"
        )
        default.parent.mkdir(parents=True, exist_ok=True)
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export zone measurements", str(default), "CSV file (*.csv)"
        )
        if selected:
            path = export_zone_csv(self.results, selected)
            self.status_label.setText(f"Saved {path}")

    def save_visualizations(self) -> None:
        if not self.results or self.signal_path is None:
            return
        sample = self.sample_combo.currentText()
        analysis_root = self._output_root() / "Zone_analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        selected = QFileDialog.getExistingDirectory(
            self, "Choose zone visualization directory", str(analysis_root)
        )
        if selected:
            selected_path = Path(selected)
            sample_path = Path(sample)
            output_dir = (
                selected_path
                if _path_ends_with(selected_path, sample_path)
                else selected_path / sample_path
            )
            try:
                paths = save_zone_visualizations(
                    self.signal_path, self.results, output_dir
                )
                csv_path = export_zone_csv(
                    self.results, output_dir / "zone_measurements.csv"
                )
            except Exception as exc:
                self._error(str(exc))
                return
            self._show_visualization(self.visualization_combo.currentText())
            self.status_label.setText(
                f"Saved {len(paths)} images/masks and {csv_path.name} in {output_dir}"
            )

    def _error(self, text: str) -> None:
        QMessageBox.critical(self, "Five-zone analysis", text)


def main() -> None:
    app = QApplication(sys.argv)
    window = AnalysisWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
