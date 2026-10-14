"""Launch with python disparity_gui.py (or python -m src.disparity_gui)."""
import itertools
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, QLibraryInfo, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QSplitter, QScrollArea,
    QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)
try:
    from . import disparity_core as core
except ImportError:
    import disparity_core as core


def configure_qt_plugins():
    """Undo cv2 wheel Qt overrides before creating the PyQt application.

    Non-headless OpenCV wheels set these variables during import. Their bundled
    Qt plugins cannot be loaded by the Qt library used by PyQt5.
    """
    cv2_root = Path(cv2.__file__).resolve().parent
    for variable in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        value = os.environ.get(variable)
        if value and Path(value).resolve().is_relative_to(cv2_root):
            if variable == 'QT_QPA_PLATFORM_PLUGIN_PATH':
                os.environ[variable] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
            else:
                os.environ.pop(variable, None)


configure_qt_plugins()


class ImageView(QScrollArea):
    def __init__(self):
        super().__init__()
        self.label = QLabel('Load images and calibration, then run.')
        self.label.setAlignment(Qt.AlignCenter)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.image = None
        self.zoom = 0  # fit

    def set_image(self, image):
        self.image = image
        self.render()

    def render(self):
        if self.image is None:
            return
        rgb = np.ascontiguousarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        pix = QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())
        if self.zoom:
            pix = pix.scaled(max(1, int(w*self.zoom)), max(1, int(h*self.zoom)), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setWidgetResizable(False)
            self.label.resize(pix.size())
        else:
            self.setWidgetResizable(True)
            pix = pix.scaled(self.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.zoom:
            self.render()


class Worker(QThread):
    result = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, request):
        super().__init__()
        self.request = request

    def run(self):
        try:
            r = self.request
            left, right = core.read_pair(r['left'], r['right'], r['layout'])
            if r['swap']:
                left, right = right, left
            c = core.load_calibration(r['calibration'])
            rl, rr, masks, q = core.rectify_pair(left, right, c, r['alpha'], r['model'], r['saved'])
            if r['clahe']:
                clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
                pair = []
                for image in (rl, rr):
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    pair.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
                rl, rr = pair
            rows, best = [], None
            candidates = [r['params']]
            if r['sweep']:
                blocks = (5, 9, 15) if 'SGBM' not in r['algorithm'] else (3, 5, 9)
                if r['algorithm'] == 'CUDA SGM':
                    candidates = [dict(r['params'], P1=p1, P2=p2, uniquenessRatio=u)
                                  for (p1, p2), u in itertools.product(((10, 120), (20, 240), (40, 480)), (5, 10, 15))]
                else:
                    candidates = [dict(r['params'], blockSize=b, uniquenessRatio=u,
                                       P1=8*b*b, P2=32*b*b)
                                  for b, u in itertools.product(blocks, (5, 10, 15))]
            for index, p in enumerate(candidates):
                if self.isInterruptionRequested():
                    return
                self.progress.emit(f'Computing {index+1}/{len(candidates)}: {r["algorithm"]}')
                d, valid, elapsed = core.compute_disparity(rl, rr, r['algorithm'], p, r.get('wls'))
                valid, metrics, error = core.evaluate(rl, rr, d, valid, masks)
                row = dict(params=p, milliseconds=elapsed, **metrics)
                rows.append(row)
                if best is None or metrics['score'] > best['metrics']['score']:
                    best = dict(left=rl, right=rr, disparity=d, valid=valid, error=error,
                                Q=q, metrics=row, params=p, request=r)
            if self.isInterruptionRequested():
                return
            rows.sort(key=lambda row: row['score'], reverse=True)
            best['rows'] = rows
            self.result.emit(best)
        except Exception as exc:
            self.failed.emit(str(exc))


class DisparityWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Stereo disparity parameter tuner')
        self.resize(1450, 920)
        self.worker = None
        self.output = None
        self.pending = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.live_run)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        split = QSplitter()
        outer.addWidget(split)
        controls = QWidget()
        form = QFormLayout(controls)
        self.paths = {}
        for key, title in (('left', 'Left / combined image'), ('right', 'Right image'), ('calibration', 'Calibration YAML')):
            edit = QLineEdit()
            button = QPushButton('Browse…')
            button.clicked.connect(lambda _, k=key: self.browse(k))
            row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(edit); layout.addWidget(button)
            form.addRow(title, row)
            self.paths[key] = edit
            edit.textChanged.connect(self.schedule)
        self.layout_choice = QComboBox()
        self.layout_choice.addItems(['Separate images', 'Side by side', 'Top / bottom'])
        form.addRow('Input layout', self.layout_choice)
        self.layout_choice.currentTextChanged.connect(self.layout_changed)
        self.swap = QCheckBox('Swap input images (YAML still defines left/right)')
        form.addRow(self.swap)
        self.model = QComboBox(); self.model.addItems(['Pinhole', 'Fisheye'])
        form.addRow('Lens model', self.model)
        self.alpha = QDoubleSpinBox(); self.alpha.setRange(0, 1); self.alpha.setSingleStep(.05)
        form.addRow('Rectification alpha / balance', self.alpha)
        self.saved = QCheckBox('Use saved rectification matrices')
        form.addRow(self.saved)
        self.saved.toggled.connect(lambda checked: self.alpha.setEnabled(not checked))
        self.clahe = QCheckBox('CLAHE contrast enhancement')
        form.addRow(self.clahe)
        self.algorithm = QComboBox()
        for name, reason in core.availability().items():
            self.algorithm.addItem(name)
            index = self.algorithm.count()-1
            if reason:
                item = self.algorithm.model().item(index)
                item.setEnabled(False); item.setToolTip(reason)
        self.algorithm.setCurrentText('StereoSGBM')
        form.addRow('Matcher', self.algorithm)
        available = '\n'.join(f'{name}: {reason}' for name, reason in core.availability().items() if reason)
        notice = QLabel(available); notice.setWordWrap(True)
        form.addRow(notice)
        self.wls = QCheckBox('WLS filter with right-camera disparity')
        form.addRow(self.wls)
        self.wls_lambda = QDoubleSpinBox(); self.wls_lambda.setRange(1., 100000.); self.wls_lambda.setValue(8000.)
        self.wls_lambda.setSingleStep(1000.); form.addRow('WLS lambda (strength)', self.wls_lambda)
        self.wls_sigma = QDoubleSpinBox(); self.wls_sigma.setRange(.01, 10.); self.wls_sigma.setValue(1.5)
        self.wls_sigma.setSingleStep(.1); form.addRow('WLS sigmaColor (edges)', self.wls_sigma)
        self.wls_notice = QLabel(); self.wls_notice.setWordWrap(True); form.addRow(self.wls_notice)
        self.wls.toggled.connect(self.update_wls_controls)
        self.wls_lambda.valueChanged.connect(self.schedule); self.wls_sigma.valueChanged.connect(self.schedule)
        self.params = {}
        limits = dict(minDisparity=(-512, 512), numDisparities=(16, 1024), blockSize=(1, 255),
                      P1=(0, 100000), P2=(1, 100000), uniquenessRatio=(0, 100),
                      speckleWindowSize=(0, 1000), speckleRange=(0, 64),
                      disp12MaxDiff=(-1, 64), preFilterCap=(1, 63), textureThreshold=(0, 1000))
        for key, (lo, hi) in limits.items():
            spin = QSpinBox(); spin.setRange(lo, hi); spin.setValue(core.DEFAULTS[key])
            if key == 'numDisparities': spin.setSingleStep(16)
            if key == 'blockSize': spin.setSingleStep(2)
            spin.valueChanged.connect(self.schedule)
            self.params[key] = spin
            form.addRow(key, spin)
        self.mode = QComboBox(); form.addRow('Matching mode', self.mode)
        self.mode.currentIndexChanged.connect(self.schedule)
        penalty = QPushButton('Set P1/P2 from block size')
        penalty.clicked.connect(self.auto_penalties); form.addRow(penalty)
        self.live = QCheckBox('Live update (after 400 ms pause)'); form.addRow(self.live)
        buttons = QWidget(); bl = QHBoxLayout(buttons); bl.setContentsMargins(0, 0, 0, 0)
        self.run_button = QPushButton('Rectify && compute'); self.run_button.clicked.connect(lambda: self.start(False))
        self.sweep_button = QPushButton('Sweep 9 candidates'); self.sweep_button.clicked.connect(lambda: self.start(True))
        bl.addWidget(self.run_button); bl.addWidget(self.sweep_button); form.addRow(buttons)
        self.cancel_button = QPushButton('Cancel'); self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel); form.addRow(self.cancel_button)
        self.export_button = QPushButton('Export result + settings…'); self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export); form.addRow(self.export_button)
        self.load_button = QPushButton('Load saved settings…'); self.load_button.clicked.connect(self.load_settings)
        form.addRow(self.load_button)
        help_label = QLabel('Sweep keeps disparity range fixed and varies block size/uniqueness; SGBM penalties follow block size. CUDA varies penalties/uniqueness. Score = valid coverage × (1 − photometric MAE / 255). Inspect results visually: this heuristic is not depth accuracy.\n\nUse native, uncropped calibration-size images. If YAML has no image dimensions, ensure the resolution matches yourself. Input is an unrectified, decoded image pair; sensor byte dumps need conversion first.')
        help_label.setWordWrap(True); form.addRow(help_label)
        scroll = QScrollArea(); scroll.setWidget(controls); scroll.setWidgetResizable(True); scroll.setMinimumWidth(405)
        split.addWidget(scroll)
        results = QWidget(); rl = QVBoxLayout(results)
        toolbar = QHBoxLayout(); toolbar.addWidget(QLabel('Zoom'))
        zoom = QComboBox(); zoom.addItems(['Fit', '50%', '100%', '200%']); toolbar.addWidget(zoom)
        self.guides = QCheckBox('Epipolar guides'); self.guides.setChecked(True); toolbar.addWidget(self.guides)
        toolbar.addStretch(); rl.addLayout(toolbar)
        self.tabs = QTabWidget(); self.views = {}
        for name in ('Rectified pair', 'Disparity', 'Valid mask', 'Photometric error'):
            view = ImageView(); self.views[name] = view; self.tabs.addTab(view, name)
        rl.addWidget(self.tabs, 1)
        self.metrics = QLabel('No result yet. Disparity values and range are in pixels.'); rl.addWidget(self.metrics)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Score', 'Valid %', 'MAE', 'ms', 'Block', 'Uniqueness', 'P1 / P2'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setMaximumHeight(220)
        self.table.cellDoubleClicked.connect(self.apply_row)
        rl.addWidget(self.table)
        rl.addWidget(QLabel('Double-click a sweep row to apply its parameters and recompute. Black disparity pixels are invalid.'))
        split.addWidget(results); split.setSizes([440, 1000])
        zoom.currentIndexChanged.connect(lambda i: self.set_zoom((0, .5, 1., 2.)[i]))
        self.guides.toggled.connect(self.display)
        self.algorithm.currentTextChanged.connect(self.algorithm_changed)
        for control in (self.swap, self.saved, self.clahe, self.wls, self.live): control.toggled.connect(self.schedule)
        self.model.currentTextChanged.connect(self.schedule); self.alpha.valueChanged.connect(self.schedule)
        self.algorithm_changed()
        self.statusBar().showMessage('Ready')

    def browse(self, key):
        filt = 'YAML (*.yaml *.yml);;All files (*)' if key == 'calibration' else 'Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)'
        path, _ = QFileDialog.getOpenFileName(self, 'Select ' + key, '', filt)
        if path: self.paths[key].setText(path)

    def layout_changed(self):
        self.paths['right'].setEnabled(self.layout_choice.currentIndex() == 0)
        self.schedule()

    def algorithm_changed(self):
        name = self.algorithm.currentText()
        cuda = name == 'CUDA SGM'
        sgbm = 'SGBM' in name
        for key, spin in self.params.items():
            spin.setEnabled(key in ('minDisparity', 'numDisparities', 'P1', 'P2', 'uniquenessRatio') if cuda
                            else (key not in ('P1', 'P2') if not sgbm else key != 'textureThreshold'))
        self.mode.blockSignals(True); self.mode.clear()
        entries = [('HH4', 0), ('HH', 1)] if cuda else ([('SGBM', 0), ('HH', 1)] if name == 'StereoBinarySGBM'
                  else [('SGBM', 0), ('HH', 1), ('SGBM 3-way', 2), ('HH4', 3)])
        for label, value in entries: self.mode.addItem(label, value)
        self.mode.setEnabled(cuda or sgbm); self.mode.blockSignals(False)
        if cuda:
            self.params['numDisparities'].setValue(128)
            self.params['P1'].setValue(10); self.params['P2'].setValue(120)
        self.update_wls_controls()
        self.schedule()

    def update_wls_controls(self, *_):
        reason = core.wls_unavailable_reason(self.algorithm.currentText())
        self.wls.setEnabled(not reason)
        if reason: self.wls.setChecked(False)
        self.wls_notice.setText(reason)
        enabled = self.wls.isChecked() and not reason
        self.wls_lambda.setEnabled(enabled); self.wls_sigma.setEnabled(enabled)

    def auto_penalties(self):
        b = self.params['blockSize'].value()
        self.params['P1'].setValue(8*b*b); self.params['P2'].setValue(32*b*b)

    def request(self, sweep=False):
        return dict(**{k: v.text() for k, v in self.paths.items()},
                    layout=self.layout_choice.currentText(), swap=self.swap.isChecked(),
                    model=self.model.currentText(), alpha=self.alpha.value(), saved=self.saved.isChecked(),
                    wls=dict(enabled=self.wls.isChecked(), **{'lambda': self.wls_lambda.value()}, sigmaColor=self.wls_sigma.value()),
                    clahe=self.clahe.isChecked(), algorithm=self.algorithm.currentText(), sweep=sweep,
                    params=dict(**{k: v.value() for k, v in self.params.items()}, mode=self.mode.currentData()))

    def schedule(self, *_):
        if hasattr(self, 'live') and self.live.isChecked(): self.timer.start(400)

    def live_run(self):
        if not all(self.paths[k].text() for k in ('left', 'calibration')): return
        if self.layout_choice.currentIndex() == 0 and not self.paths['right'].text(): return
        if self.worker is not None:
            self.pending = True
        else: self.start(False)

    def start(self, sweep):
        if self.worker is not None: return
        self.timer.stop(); self.pending = False
        self.run_button.setEnabled(False); self.sweep_button.setEnabled(False)
        self.cancel_button.setEnabled(True); self.export_button.setEnabled(False)
        self.worker = Worker(self.request(sweep))
        self.worker.result.connect(self.receive)
        self.worker.failed.connect(self.failure)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.finished)
        self.worker.start()

    def cancel(self):
        self.pending = False; self.timer.stop()
        if self.worker: self.worker.requestInterruption()
        self.statusBar().showMessage('Cancelling after the current matcher call…')

    def failure(self, message):
        self.statusBar().showMessage(message)
        if not self.live.isChecked(): QMessageBox.warning(self, 'Processing failed', message)

    def finished(self):
        self.worker.deleteLater(); self.worker = None
        self.run_button.setEnabled(True); self.sweep_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.export_button.setEnabled(self.output is not None)
        if self.pending and self.live.isChecked(): self.timer.start(1)

    def receive(self, output):
        self.output = output
        # Show the winning sweep settings in the controls without starting a redundant live run.
        current = self.request(output['request']['sweep'])
        if current == output['request']:
            for key, value in output['params'].items():
                if key in self.params:
                    self.params[key].blockSignals(True); self.params[key].setValue(value); self.params[key].blockSignals(False)
        self.display()
        rows = output['rows']; self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            p = row['params']
            values = [f'{row["score"]:.4f}', f'{100*row["coverage"]:.1f}', f'{row["photometric_mae"]:.2f}',
                      f'{row["milliseconds"]:.1f}', str(p['blockSize']), str(p['uniquenessRatio']), f'{p["P1"]} / {p["P2"]}']
            for j, value in enumerate(values): self.table.setItem(i, j, QTableWidgetItem(value))
        m = output['metrics']; p = output['params']
        filter_label = ' + WLS' if output['request'].get('wls', {}).get('enabled', False) else ''
        self.metrics.setText(f'{output["request"]["algorithm"]}{filter_label} | {100*m["coverage"]:.1f}% valid | warp MAE {m["photometric_mae"]:.2f}/255 | {m["milliseconds"]:.1f} ms compute | color range [{p["minDisparity"]}, {p["minDisparity"]+p["numDisparities"]}] px')
        self.statusBar().showMessage('Complete. Result uses the inputs/settings captured when processing started.')

    def display(self, *_):
        if self.output is None: return
        o = self.output; p = o['params']
        pair = np.hstack((o['left'], o['right'])).copy()
        if self.guides.isChecked():
            for y in range(0, pair.shape[0], max(1, pair.shape[0]//20)):
                cv2.line(pair, (0, y), (pair.shape[1]-1, y), (0, 255, 0), 1)
        self.views['Rectified pair'].set_image(pair)
        self.views['Disparity'].set_image(core.colorize(o['disparity'], o['valid'], p['minDisparity'], p['numDisparities']))
        self.views['Valid mask'].set_image(cv2.cvtColor(o['valid'].astype(np.uint8)*255, cv2.COLOR_GRAY2BGR))
        error = cv2.applyColorMap(np.clip(o['error'], 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        error[~o['valid']] = 0; self.views['Photometric error'].set_image(error)

    def set_zoom(self, value):
        for view in self.views.values(): view.zoom = value; view.render()

    def apply_row(self, row, _):
        if self.worker or self.output is None: return
        for key, value in self.output['rows'][row]['params'].items():
            if key in self.params: self.params[key].setValue(value)
        self.start(False)

    def export(self):
        if self.output is None: return
        folder = QFileDialog.getExistingDirectory(self, 'Export into a directory (creates a unique result subfolder)')
        if not folder: return
        try:
            from datetime import datetime
            target = Path(folder) / ('disparity_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            target.mkdir()
            o = self.output; p = o['params']
            disparity = o['disparity'].copy(); disparity[~o['valid']] = np.nan
            np.save(target/'disparity_pixels.npy', disparity)
            if o['Q'] is not None: np.save(target/'Q.npy', o['Q'])
            images = {'left_rectified.png': o['left'], 'right_rectified.png': o['right'],
                      'valid_mask.png': o['valid'].astype(np.uint8)*255,
                      'disparity_color.png': core.colorize(o['disparity'], o['valid'], p['minDisparity'], p['numDisparities'])}
            for name, image in images.items():
                if not cv2.imwrite(str(target/name), image): raise IOError('Cannot write ' + name)
            settings = dict(o['request'], params=p, sweep=False)
            settings['metrics'] = o['metrics']; settings['sweep_results'] = o['rows']
            settings['opencv_version'] = cv2.__version__
            for key in ('left', 'right', 'calibration'):
                if settings[key]: settings[key] = str(Path(settings[key]).resolve())
            (target/'settings.json').write_text(json.dumps(settings, indent=2))
            self.statusBar().showMessage('Exported to ' + str(target))
        except Exception as exc: QMessageBox.warning(self, 'Export failed', str(exc))

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load settings', '', 'JSON (*.json)')
        if not path: return
        try:
            r = json.loads(Path(path).read_text())
            if r['algorithm'] not in core.availability() or core.availability()[r['algorithm']]:
                raise ValueError('Saved matcher is unavailable in this OpenCV build.')
            self.live.setChecked(False)
            for key, edit in self.paths.items(): edit.setText(r[key])
            self.layout_choice.setCurrentText(r['layout']); self.model.setCurrentText(r['model'])
            self.swap.setChecked(r['swap']); self.saved.setChecked(r['saved']); self.clahe.setChecked(r['clahe'])
            self.alpha.setValue(r['alpha']); self.algorithm.setCurrentText(r['algorithm'])
            wls = r.get('wls', {})
            if wls.get('enabled') and core.wls_unavailable_reason(r['algorithm']):
                raise ValueError(core.wls_unavailable_reason(r['algorithm']))
            self.wls_lambda.setValue(wls.get('lambda', 8000.)); self.wls_sigma.setValue(wls.get('sigmaColor', 1.5))
            self.wls.setChecked(wls.get('enabled', False))
            for key, spin in self.params.items(): spin.setValue(r['params'][key])
            self.mode.setCurrentIndex(self.mode.findData(r['params']['mode']))
            self.start(False)
        except Exception as exc: QMessageBox.warning(self, 'Load failed', str(exc))

    def closeEvent(self, event):
        if self.worker is not None:
            self.cancel()
            self.statusBar().showMessage('Wait for the current computation to stop, then close again.')
            event.ignore()
        else: event.accept()


def launch_disparity_gui():
    app = QApplication.instance() or QApplication(sys.argv)
    window = DisparityWindow(); window.show()
    app.disparity_window = window
    return app.exec_()


if __name__ == '__main__':
    sys.exit(launch_disparity_gui())
