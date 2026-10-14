"""
calibration_gui_calib.py  —  src/
==================================
Interactive PyQt5 GUI for stereo camera calibration:
  • Load ChArUco images, detect corners, initial calibration
  • Per-image reprojection error review; threshold filter; re-calibrate
  • Stereo rectification preview with alpha slider (cv2.stereoRectify)
  • Export — Custom, ROS2 CameraInfo, passive stereo (metres), and ORB-SLAM3 YAML

Launch:
    python -m src.calibration_gui_calib          # from project root
    from src.calibration_gui_calib import launch_calibration_gui; launch_calibration_gui()
"""

from __future__ import annotations


import io
import os
import sys
import traceback as _tb
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QGridLayout, QWidget,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─── constants ────────────────────────────────────────────────────────────────
THUMB = 100
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ARUCO_DICTS: dict[str, int] = {
    "DICT_4X4_50":  cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50":  cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


# ─── utilities ────────────────────────────────────────────────────────────────

def np2pixmap(img: np.ndarray, max_px: int = THUMB) -> QPixmap:
    """Scale a BGR/gray numpy array into a QPixmap thumbnail."""
    if img is None:
        return QPixmap()
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max_px / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    qi = QImage(small.tobytes(), nw, nh, nw * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qi)


def draw_epipolar(img: np.ndarray, n: int = 14, color=(0, 220, 0)) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    step = max(1, h // (n + 1))
    for i in range(1, n + 1):
        cv2.line(out, (0, i * step), (w, i * step), color, 1, cv2.LINE_AA)
    return out


def fmt_mat(m) -> str:
    if m is None:
        return "N/A"
    return "\n".join("  [" + ", ".join(f"{v:.8g}" for v in row) + "]"
                     for row in np.atleast_2d(m))


def fmt_vec(v) -> str:
    if v is None:
        return "N/A"
    return "[" + ", ".join(f"{x:.8g}" for x in np.ravel(v)) + "]"


# ─── YAML builders ────────────────────────────────────────────────────────────

def _yaml_float(value) -> str:
    """Format exported floating-point values with six decimal places."""
    return f"{float(value):.6f}"


def _row_str(row) -> str:
    return "  - [ " + ", ".join(_yaml_float(v) for v in row) + " ]"


def _mat_block(mat, key: str) -> str:
    rows = "\n".join(_row_str(row) for row in np.atleast_2d(mat))
    return f"\n{key}:\n{rows}\n"


def _vec_inline(vec, key: str) -> str:
    vals = ", ".join(_yaml_float(v) for v in np.ravel(vec))
    return f"\n{key}: [ {vals} ]\n"


def _flat(arr) -> str:
    return ", ".join(_yaml_float(v) for v in np.ravel(arr))


def build_custom_yaml(state, include_rect: bool) -> str:
    """Build project-format YAML string matching 20260206_binning.yaml."""
    Rl_mat, _ = cv2.Rodrigues(state.rv_l[0])
    Rr_mat, _ = cv2.Rodrigues(state.rv_r[0])
    buf = io.StringIO()
    buf.write(f"# Camera calibration — exported {datetime.now():%Y-%m-%d %H:%M}\n")
    buf.write(_mat_block(state.Kl,       "camera_matrix_left"))
    buf.write(_mat_block(state.Kr,       "camera_matrix_right"))
    buf.write(_vec_inline(state.Dl,      "dist_coeffs_left"))
    buf.write(_vec_inline(state.Dr,      "dist_coeffs_right"))
    buf.write(_mat_block(Rl_mat,         "rot_matrix_left"))
    buf.write(_mat_block(Rr_mat,         "rot_matrix_right"))
    buf.write(_vec_inline(state.tv_l[0], "t_left"))
    buf.write(_vec_inline(state.tv_r[0], "t_right"))
    buf.write(_mat_block(state.R,        "R"))
    buf.write(_vec_inline(state.T,       "T"))
    if include_rect and state.rectified:
        buf.write(_mat_block(state.Rl, "rect_matrix_left"))
        buf.write(_mat_block(state.Rr, "rect_matrix_right"))
        buf.write(_mat_block(state.Pl, "proj_matrix_left"))
        buf.write(_mat_block(state.Pr, "proj_matrix_right"))
    return buf.getvalue()


def build_passive_stereo_yaml(state) -> str:
    """OpenCV stereo calibration at calibration resolution, with T in metres."""
    if not state.calibrated:
        raise ValueError("Run calibration first.")
    w, h = state.img_size
    if w <= 0 or h <= 0:
        raise ValueError("Calibration image resolution is unavailable.")
    buf = io.StringIO()
    buf.write("%YAML:1.0\n---\n")
    buf.write("# Stereo Calibration — passive_stereo_capture\n")
    buf.write("# OpenCV FileStorage format. Use native, uncropped calibration images.\n")
    buf.write("# image_width/image_height match the calibration images; use binning=1.\n")
    buf.write("# K1/D1: LEFT camera; K2/D2: RIGHT camera.\n")
    buf.write("# R/T are cv::stereoCalibrate outputs (X_right = R * X_left + T).\n")
    buf.write("# T units: METRES (converted from calibration millimetres).\n")
    buf.write("# The capture node computes rectification and remap maps at startup.\n\n")
    buf.write(f"image_width: {w}\nimage_height: {h}\n")
    for key, matrix, description in (
        ("K1", state.Kl, "Left camera intrinsic matrix"),
        ("D1", np.asarray(state.Dl).reshape(1, -1), "Left distortion coefficients"),
        ("K2", state.Kr, "Right camera intrinsic matrix"),
        ("D2", np.asarray(state.Dr).reshape(1, -1), "Right distortion coefficients"),
        ("R", state.R, "Rotation from LEFT to RIGHT camera"),
        ("T", np.asarray(state.T).reshape(3, 1) / 1000.0,
         "Translation from LEFT to RIGHT camera (METRES)"),
    ):
        matrix = np.asarray(matrix, dtype=np.float64)
        values = _flat(matrix)
        buf.write(f"\n# {description}\n{key}: !!opencv-matrix\n")
        buf.write(f"  rows: {matrix.shape[0]}\n  cols: {matrix.shape[1]}\n")
        buf.write(f"  dt: d\n  data: [ {values} ]\n")
    return buf.getvalue()


def build_camerainfo_yaml(state, side: str, cam_name: str,
                           w: int, h: int, use_rect: bool) -> str:
    """Build ROS2 CameraInfo YAML string matching 16378750.yaml."""
    K = state.Kl if side == "left" else state.Kr
    D = state.Dl if side == "left" else state.Dr
    if use_rect and state.rectified:
        Rect = state.Rl if side == "left" else state.Rr
        P    = state.Pl if side == "left" else state.Pr
    else:
        Rect = np.eye(3)
        P    = np.hstack([K, np.zeros((3, 1))])
    nc = int(np.ravel(D).size)
    buf = io.StringIO()
    buf.write(f"image_width: {w}\n")
    buf.write(f"image_height: {h}\n")
    buf.write(f"camera_name: {cam_name}\n")
    buf.write(f"distortion_model: plumb_bob\n")
    buf.write(f"camera_matrix:\n  rows: 3\n  cols: 3\n  data: [{_flat(K)}]\n")
    buf.write(f"distortion_coefficients:\n  rows: 1\n  cols: {nc}\n  data: [{_flat(D)}]\n")
    buf.write(f"rectification_matrix:\n  rows: 3\n  cols: 3\n  data: [{_flat(Rect)}]\n")
    buf.write(f"projection_matrix:\n  rows: 3\n  cols: 4\n  data: [{_flat(P)}]\n")
    return buf.getvalue()



def get_single_pts(board_type, c, i, board, squares, sq_len):
    if board_type == "Chessboard":
        cols, rows = squares
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_len
        return objp, c.reshape(-1, 2)
    elif board_type == "ChArUco":
        bpts = board.getChessboardCorners()
        objp = bpts[i.flatten()]
        return objp, c.reshape(-1, 2)
    elif board_type == "ArUco Grid":
        objp, imgp = board.matchImagePoints(c, i)
        if objp is not None and len(objp) > 0:
            return objp.reshape(-1, 3), imgp.reshape(-1, 2)
        return None, None

def get_stereo_pts(board_type, cl, il, cr, ir, board, squares, sq_len):
    if board_type == "Chessboard":
        cols, rows = squares
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_len
        return objp, cl.reshape(-1, 2), cr.reshape(-1, 2)
    elif board_type == "ChArUco":
        common = np.intersect1d(il.flatten(), ir.flatten())
        if len(common) < 4: return None, None, None
        ml = np.isin(il.flatten(), common)
        mr = np.isin(ir.flatten(), common)
        bpts = board.getChessboardCorners()
        objp = bpts[common]
        return objp, cl[ml].reshape(-1, 2), cr[mr].reshape(-1, 2)
    elif board_type == "ArUco Grid":
        il_flat = il.flatten()
        ir_flat = ir.flatten()
        common = np.intersect1d(il_flat, ir_flat)
        if len(common) == 0: return None, None, None
        cl_filt = [cl[idx] for idx, v in enumerate(il_flat) if v in common]
        il_filt = np.array([v for v in il_flat if v in common], dtype=np.int32).reshape(-1, 1)
        cr_filt = [cr[idx] for idx, v in enumerate(ir_flat) if v in common]
        ir_filt = np.array([v for v in ir_flat if v in common], dtype=np.int32).reshape(-1, 1)
        sort_l = np.argsort(il_filt.flatten())
        sort_r = np.argsort(ir_filt.flatten())
        cl_filt = [cl_filt[idx] for idx in sort_l]
        il_filt = il_filt[sort_l]
        cr_filt = [cr_filt[idx] for idx in sort_r]
        ir_filt = ir_filt[sort_r]
        objp_l, imgp_l = board.matchImagePoints(cl_filt, il_filt)
        objp_r, imgp_r = board.matchImagePoints(cr_filt, ir_filt)
        if objp_l is not None and objp_r is not None and len(objp_l) > 0:
            return objp_l.reshape(-1, 3), imgp_l.reshape(-1, 2), imgp_r.reshape(-1, 2)
        return None, None, None

# ─── background workers ───────────────────────────────────────────────────────

class DetectWorker(QThread):
    sig_progress = pyqtSignal(int, str)
    sig_done     = pyqtSignal(dict)
    sig_error    = pyqtSignal(str)

    def __init__(self, imgs_l, imgs_r, state, parent=None):
        super().__init__(parent)
        self.imgs_l, self.imgs_r = imgs_l, imgs_r
        self.board_type = state.board_type
        self.board = state.board
        self.adict = state.adict
        self.squares = state.squares

    def run(self):
        try:
            L = {"corners": [], "ids": [], "idx": []}
            R = {"corners": [], "ids": [], "idx": []}
            n = len(self.imgs_l)
            for i, (il, ir) in enumerate(zip(self.imgs_l, self.imgs_r)):
                self.sig_progress.emit(int((i + 1) / n * 100), f"Detecting pair {i+1}/{n}")
                for img, store in ((il, L), (ir, R)):
                    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    if self.board_type == "Chessboard":
                        ok, corners = cv2.findChessboardCorners(gray, self.squares, None)
                        if ok:
                            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                            store["corners"].append(corners)
                            store["ids"].append(np.arange(len(corners), dtype=np.int32).reshape(-1, 1))
                            store["idx"].append(i)
                    
                    elif self.board_type == "ChArUco":
                        mc, mi, _ = cv2.aruco.detectMarkers(gray, self.adict)
                        if mi is not None and len(mi) > 0:
                            ok, cc, ci = cv2.aruco.interpolateCornersCharuco(mc, mi, gray, self.board)
                            if ok and cc is not None and len(cc) >= 4:
                                store["corners"].append(cc.astype(np.float32))
                                store["ids"].append(ci.astype(np.int32))
                                store["idx"].append(i)
                                
                    elif self.board_type == "ArUco Grid":
                        mc, mi, _ = cv2.aruco.detectMarkers(gray, self.adict)
                        if mi is not None and len(mi) > 0:
                            store["corners"].append(mc)
                            store["ids"].append(mi.astype(np.int32))
                            store["idx"].append(i)
                            
            self.sig_done.emit({"L": L, "R": R})
        except Exception:
            self.sig_error.emit(_tb.format_exc())


class CalibWorker(QThread):
    sig_status = pyqtSignal(str)
    sig_done   = pyqtSignal(dict)
    sig_error  = pyqtSignal(str)

    def __init__(self, sel_cl, sel_il, sel_cr, sel_ir,
                 all_cl, all_il, all_cr, all_ir, img_size, state, parent=None):
        super().__init__(parent)
        self.sel_cl, self.sel_il = sel_cl, sel_il
        self.sel_cr, self.sel_ir = sel_cr, sel_ir
        self.all_cl, self.all_il = all_cl, all_il
        self.all_cr, self.all_ir = all_cr, all_ir
        self.img_size = img_size
        self.board_type = state.board_type
        self.board = state.board
        self.squares = state.squares
        self.sq_len = state.sq_len
        
        # Keep track of prior intrinsics if we are re-calibrating
        self.Kl_guess = state.Kl.copy() if state.calibrated and state.Kl is not None else None
        self.Dl_guess = state.Dl.copy() if state.calibrated and state.Dl is not None else None
        self.Kr_guess = state.Kr.copy() if state.calibrated and state.Kr is not None else None
        self.Dr_guess = state.Dr.copy() if state.calibrated and state.Dr is not None else None

    def _calib_single(self, clist, ilist, K_guess=None, D_guess=None):
        import cv2
        obj_pts, img_pts = [], []
        for c, i in zip(clist, ilist):
            o, p = get_single_pts(self.board_type, c, i, self.board, self.squares, self.sq_len)
            if o is not None and len(o) >= 4:
                obj_pts.append(o.astype(np.float32))
                img_pts.append(p.astype(np.float32))
        if not obj_pts:
            raise ValueError("No valid points for calibration")
            
        flags = 0
        if K_guess is not None and D_guess is not None:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            K_in = K_guess.copy()
            D_in = D_guess.copy()
        else:
            K_in = None
            D_in = None
            
        return cv2.calibrateCamera(obj_pts, img_pts, self.img_size, K_in, D_in, flags=flags)

    def run(self):
        try:
            res  = {}

            self.sig_status.emit("Calibrating left camera…")
            rms_l, Kl, Dl, rvl, tvl = self._calib_single(self.sel_cl, self.sel_il, self.Kl_guess, self.Dl_guess)
            res["left"] = {"K": Kl, "D": Dl, "rvecs": rvl, "tvecs": tvl, "rms": rms_l}

            self.sig_status.emit("Calibrating right camera…")
            rms_r, Kr, Dr, rvr, tvr = self._calib_single(self.sel_cr, self.sel_ir, self.Kr_guess, self.Dr_guess)
            res["right"] = {"K": Kr, "D": Dr, "rvecs": rvr, "tvecs": tvr, "rms": rms_r}

            self.sig_status.emit("Computing reprojection errors for all images…")
            res["errors_l"] = self._errors_all(self.all_cl, self.all_il, Kl, Dl)
            res["errors_r"] = self._errors_all(self.all_cr, self.all_ir, Kr, Dr)

            self.sig_status.emit("Stereo calibration…")
            obj_pts, pts_l, pts_r = self._match_stereo()
            if obj_pts:
                rms_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    obj_pts, pts_l, pts_r,
                    Kl, Dl, Kr, Dr, self.img_size,
                    flags=cv2.CALIB_FIX_INTRINSIC)
                res["stereo"] = {"R": R, "T": T, "E": E, "F": F, "rms": rms_s}

            self.sig_done.emit(res)
        except Exception:
            self.sig_error.emit(_tb.format_exc())

    def _errors_all(self, all_c, all_i, K, D) -> np.ndarray:
        errs = []
        for c, i in zip(all_c, all_i):
            obj, img = get_single_pts(self.board_type, c, i, self.board, self.squares, self.sq_len)
            if obj is None or len(obj) < 4:
                errs.append(np.nan)
                continue
            ok, rv, tv = cv2.solvePnP(obj.astype(np.float64), img.astype(np.float64), K, D,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                errs.append(np.nan); continue
            proj, _ = cv2.projectPoints(obj.astype(np.float64), rv, tv, K, D)
            errs.append(float(np.mean(
                np.linalg.norm(img.reshape(-1, 2) - proj.reshape(-1, 2), axis=1))))
        return np.array(errs)

    def _match_stereo(self):
        obj_pts, pts_l, pts_r = [], [], []
        for cl, il, cr, ir in zip(self.sel_cl, self.sel_il, self.sel_cr, self.sel_ir):
            o, pl, pr = get_stereo_pts(self.board_type, cl, il, cr, ir, self.board, self.squares, self.sq_len)
            if o is not None and len(o) >= 4:
                obj_pts.append(o.astype(np.float32))
                pts_l.append(pl.astype(np.float32))
                pts_r.append(pr.astype(np.float32))
        return obj_pts, pts_l, pts_r


# ─── shared calibration state ─────────────────────────────────────────────────

class CalibState:
    """Central data container shared by all GUI tabs."""

    def __init__(self):
        self.imgs_l:   list = []
        self.imgs_r:   list = []
        self.names_l:  list = []
        self.names_r:  list = []
        self.img_size: tuple = (0, 0)
        self.folder:   str   = ""

        # Board config
        self.board_type = "ChArUco"
        self.squares = (12, 9)
        self.sq_len  = 30.0
        self.mk_len  = 22.0
        self.dict_id = cv2.aruco.DICT_4X4_100
        self.board   = None
        self.adict   = None

        # Detection results — indexed by detection pair (0..N-1)
        self.det_cl:   list = []   # charuco corners left
        self.det_il:   list = []   # charuco ids left
        self.det_cr:   list = []   # charuco corners right
        self.det_ir:   list = []   # charuco ids right
        self.det_orig: list = []   # original image-pair indices

        # Intrinsic / stereo calibration
        self.Kl = self.Dl = self.Kr = self.Dr = None
        self.rv_l = self.tv_l = self.rv_r = self.tv_r = None
        self.R = self.T = self.E = self.F = None
        self.errs_l: np.ndarray | None = None  # per detection pair
        self.errs_r: np.ndarray | None = None
        self.rms_l = self.rms_r = self.rms_s = None
        self.calibrated: bool = False

        # Rectification
        self.alpha: float = 0.0
        self.Rl = self.Rr = self.Pl = self.Pr = self.Q = None
        self.roi_l = self.roi_r = None
        self.rectified: bool = False

    def build_board(self):
        if self.board_type != "Chessboard":
            self.adict = cv2.aruco.getPredefinedDictionary(self.dict_id)
        else:
            self.adict = None

        if self.board_type == "ChArUco":
            self.board = cv2.aruco.CharucoBoard(
                size=self.squares, squareLength=self.sq_len,
                markerLength=self.mk_len, dictionary=self.adict)
        elif self.board_type == "ArUco Grid":
            # For GridBoard, sq_len represents marker separation
            self.board = cv2.aruco.GridBoard(
                size=self.squares, markerLength=self.mk_len,
                markerSeparation=self.sq_len, dictionary=self.adict)
        else:
            self.board = None

    def load_imgs(self, folder: str) -> int:
        self.folder = folder
        ld = os.path.join(folder, "left")
        rd = os.path.join(folder, "right")
        lf = sorted(f for f in os.listdir(ld) if Path(f).suffix.lower() in IMG_EXTS)
        rf = sorted(f for f in os.listdir(rd) if Path(f).suffix.lower() in IMG_EXTS)
        n = min(len(lf), len(rf))
        self.names_l = lf[:n]; self.names_r = rf[:n]
        self.imgs_l  = [cv2.imread(os.path.join(ld, f)) for f in self.names_l]
        self.imgs_r  = [cv2.imread(os.path.join(rd, f)) for f in self.names_r]
        if self.imgs_l:
            h, w = self.imgs_l[0].shape[:2]
            self.img_size = (w, h)
        return n

    def apply_detection(self, det: dict):
        L, R = det["L"], det["R"]
        ld = {i: (c, ids) for i, c, ids in zip(L["idx"], L["corners"], L["ids"])}
        rd = {i: (c, ids) for i, c, ids in zip(R["idx"], R["corners"], R["ids"])}
        common = sorted(set(ld) & set(rd))
        self.det_orig = common
        self.det_cl   = [ld[i][0] for i in common]
        self.det_il   = [ld[i][1] for i in common]
        self.det_cr   = [rd[i][0] for i in common]
        self.det_ir   = [rd[i][1] for i in common]

    def apply_calib(self, res: dict):
        l = res["left"];  r = res["right"]
        self.Kl, self.Dl = l["K"], l["D"]
        self.rv_l, self.tv_l, self.rms_l = l["rvecs"], l["tvecs"], l["rms"]
        self.Kr, self.Dr = r["K"], r["D"]
        self.rv_r, self.tv_r, self.rms_r = r["rvecs"], r["tvecs"], r["rms"]
        self.errs_l = res.get("errors_l")
        self.errs_r = res.get("errors_r")
        if "stereo" in res:
            s = res["stereo"]
            self.R, self.T, self.E, self.F, self.rms_s = (
                s["R"], s["T"], s["E"], s["F"], s["rms"])
        self.calibrated = True
        self.rectified  = False

    def rectify(self, alpha: float) -> bool:
        if not self.calibrated or self.R is None:
            return False
        self.alpha = alpha
        (self.Rl, self.Rr, self.Pl, self.Pr, self.Q,
         self.roi_l, self.roi_r) = cv2.stereoRectify(
            self.Kl, self.Dl, self.Kr, self.Dr,
            self.img_size, self.R, self.T, alpha=alpha)
        self.rectified = True
        return True

    def get_rect_sample(self, pair_idx: int = 0):
        if not self.rectified or not self.det_orig:
            return None, None
        k    = min(pair_idx, len(self.det_orig) - 1)
        orig = self.det_orig[k]
        il   = self.imgs_l[orig]; ir = self.imgs_r[orig]
        ml1, ml2 = cv2.initUndistortRectifyMap(
            self.Kl, self.Dl, self.Rl, self.Pl, self.img_size, cv2.CV_32FC1)
        mr1, mr2 = cv2.initUndistortRectifyMap(
            self.Kr, self.Dr, self.Rr, self.Pr, self.img_size, cv2.CV_32FC1)
        return (cv2.remap(il, ml1, ml2, cv2.INTER_LINEAR),
                cv2.remap(ir, mr1, mr2, cv2.INTER_LINEAR))


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Image Review & Filtering
# ═══════════════════════════════════════════════════════════════════════════════

class ImageReviewTab(QWidget):
    sig_calib_done = pyqtSignal()

    # Column indices
    COL_IDX, COL_THUMB_L, COL_ERR_L, COL_THUMB_R, COL_ERR_R, COL_KEEP = range(6)

    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state  = state
        self.status = status_fn
        self._wd    = None   # DetectWorker
        self._wc    = None   # CalibWorker
        self._chart_timer = QTimer()
        self._chart_timer.setSingleShot(True)
        self._chart_timer.setInterval(300)
        self._chart_timer.timeout.connect(self._refresh_chart)
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # ── Setup ─────────────────────────────────────────────────────────────
        setup = QGroupBox("Setup")
        sg = QVBoxLayout(setup)

        r0 = QHBoxLayout()
        r0.addWidget(QLabel("Images folder:"))
        self.le_folder = QLineEdit()
        self.le_folder.setPlaceholderText("Folder with left/ and right/ sub-directories")
        r0.addWidget(self.le_folder)
        b = QPushButton("Browse…"); b.setFixedWidth(80); b.clicked.connect(self._browse)
        r0.addWidget(b)
        sg.addLayout(r0)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Board:"))
        self.cb_board_type = QComboBox()
        self.cb_board_type.addItems(["ChArUco", "ArUco Grid", "Chessboard"])
        self.cb_board_type.currentTextChanged.connect(self._on_board_type_changed)
        r1.addWidget(self.cb_board_type)
        r1.addSpacing(12)
        r1.addWidget(QLabel("Cols×Rows:"))
        self.sb_cols = QSpinBox(); self.sb_cols.setRange(3, 30); self.sb_cols.setValue(12)
        self.sb_rows = QSpinBox(); self.sb_rows.setRange(3, 30); self.sb_rows.setValue(9)
        r1.addWidget(self.sb_cols); r1.addWidget(QLabel("×")); r1.addWidget(self.sb_rows)
        r1.addSpacing(12)
        self.lbl_sq = QLabel("Square/Sep (mm):")
        r1.addWidget(self.lbl_sq)
        self.dsb_sq = QDoubleSpinBox(); self.dsb_sq.setRange(0.1, 1000); self.dsb_sq.setValue(30.0)
        r1.addWidget(self.dsb_sq)
        self.lbl_mk = QLabel("Marker len (mm):")
        r1.addWidget(self.lbl_mk)
        self.dsb_mk = QDoubleSpinBox(); self.dsb_mk.setRange(0.1, 1000); self.dsb_mk.setValue(22.0)
        r1.addWidget(self.dsb_mk)
        r1.addSpacing(12)
        self.lbl_dict = QLabel("Dict:")
        r1.addWidget(self.lbl_dict)
        self.cb_dict = QComboBox(); self.cb_dict.addItems(ARUCO_DICTS)
        self.cb_dict.setCurrentText("DICT_4X4_100")
        r1.addWidget(self.cb_dict)
        r1.addStretch()
        sg.addLayout(r1)
        self._on_board_type_changed(self.cb_board_type.currentText())

        self.btn_detect = QPushButton("🔍  Detect & Calibrate")
        self.btn_detect.setFixedHeight(34)
        self.btn_detect.clicked.connect(self._start)
        sg.addWidget(self.btn_detect)

        self.progress = QProgressBar(); self.progress.hide()
        sg.addWidget(self.progress)
        self.lbl_prog = QLabel(""); self.lbl_prog.setAlignment(Qt.AlignCenter)
        sg.addWidget(self.lbl_prog)
        root.addWidget(setup)

        # ── Filter bar ────────────────────────────────────────────────────────
        fc = QHBoxLayout()
        fc.addWidget(QLabel("Error threshold (px):"))
        self.dsb_thr = QDoubleSpinBox()
        self.dsb_thr.setRange(0.01, 50.0); self.dsb_thr.setSingleStep(0.05)
        self.dsb_thr.setValue(1.0)
        self.dsb_thr.valueChanged.connect(self._apply_threshold)
        fc.addWidget(self.dsb_thr)
        fc.addSpacing(12)
        for txt, fn in [("Select All", self._sel_all),
                        ("Deselect All", self._sel_none),
                        ("Invert", self._sel_inv)]:
            b2 = QPushButton(txt); b2.clicked.connect(fn); fc.addWidget(b2)
        fc.addStretch()
        self.lbl_sel = QLabel("—"); fc.addWidget(self.lbl_sel)
        root.addLayout(fc)

        # ── Table + chart ──────────────────────────────────────────────────────
        spl = QSplitter(Qt.Horizontal)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Left", "L.Err(px)", "Right", "R.Err(px)", "Keep"])
        hh = self.table.horizontalHeader()
        for col, mode in [(0, QHeaderView.ResizeToContents),
                          (1, QHeaderView.Fixed),
                          (2, QHeaderView.ResizeToContents),
                          (3, QHeaderView.Fixed),
                          (4, QHeaderView.ResizeToContents),
                          (5, QHeaderView.ResizeToContents)]:
            hh.setSectionResizeMode(col, mode)
        self.table.setColumnWidth(1, THUMB + 8)
        self.table.setColumnWidth(3, THUMB + 8)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setDefaultSectionSize(THUMB + 10)
        # Connect itemChanged to track checkbox changes
        self.table.itemChanged.connect(self._on_item_changed)
        spl.addWidget(self.table)

        chart_w = QWidget()
        cl = QVBoxLayout(chart_w); cl.setContentsMargins(0, 0, 0, 0)
        self.fig    = Figure(figsize=(5, 3), dpi=90, tight_layout=True)
        self.ax     = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        cl.addWidget(self.canvas)
        spl.addWidget(chart_w)
        spl.setSizes([620, 380])
        root.addWidget(spl, 1)

        # ── Re-calibrate ──────────────────────────────────────────────────────
        bot = QHBoxLayout()
        bot.addStretch()
        self.btn_recal = QPushButton("🔄  Re-Calibrate with Selected Images")
        self.btn_recal.setFixedHeight(34); self.btn_recal.setEnabled(False)
        self.btn_recal.clicked.connect(self._recalibrate)
        bot.addWidget(self.btn_recal)
        root.addLayout(bot)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _on_board_type_changed(self, text):
        if text == "Chessboard":
            self.lbl_sq.setText("Square len (mm):")
            self.lbl_mk.hide(); self.dsb_mk.hide()
            self.lbl_dict.hide(); self.cb_dict.hide()
        elif text == "ArUco Grid":
            self.lbl_sq.setText("Marker Sep (mm):")
            self.lbl_mk.show(); self.dsb_mk.show()
            self.lbl_dict.show(); self.cb_dict.show()
        else: # ChArUco
            self.lbl_sq.setText("Square len (mm):")
            self.lbl_mk.show(); self.dsb_mk.show()
            self.lbl_dict.show(); self.cb_dict.show()

    def _set_busy(self, busy: bool):
        self.btn_detect.setEnabled(not busy)
        self.progress.setVisible(busy)
        if not busy:
            self.lbl_prog.setText("")
            self.btn_recal.setEnabled(self.state.calibrated)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select images folder")
        if d:
            self.le_folder.setText(d)

    def _start(self):
        folder = self.le_folder.text().strip()
        if not folder or not os.path.isdir(os.path.join(folder, "left")):
            QMessageBox.warning(self, "Error",
                "Select a valid folder containing 'left/' and 'right/' sub-directories.")
            return
        self.state.board_type = self.cb_board_type.currentText()
        self.state.squares = (self.sb_cols.value(), self.sb_rows.value())
        self.state.sq_len  = self.dsb_sq.value()
        self.state.mk_len  = self.dsb_mk.value()
        self.state.dict_id = ARUCO_DICTS[self.cb_dict.currentText()]
        
        try:
            self.state.build_board()
        except Exception as e:
            QMessageBox.critical(self, "Board Configuration Error", str(e))
            return
            
        n = self.state.load_imgs(folder)
        self.status(f"Loaded {n} image pairs from {folder}")
        self._set_busy(True)
        self._wd = DetectWorker(self.state.imgs_l, self.state.imgs_r, self.state)
        self._wd.sig_progress.connect(
            lambda p, m: (self.progress.setValue(p), self.lbl_prog.setText(m)))
        self._wd.sig_done.connect(self._on_detect_done)
        self._wd.sig_error.connect(self._on_error)
        self._wd.start()

    def _on_detect_done(self, det: dict):
        self.state.apply_detection(det)
        n = len(self.state.det_orig)
        self.lbl_prog.setText(f"Detected {n} valid pairs — calibrating…")
        if n < 4:
            QMessageBox.warning(self, "Too few detections",
                f"Only {n} valid stereo pairs found. Need ≥ 4.")
            self._set_busy(False); return
        self._run_calib(list(range(n)))

    def _run_calib(self, sel: list[int]):
        sc_l = [self.state.det_cl[i] for i in sel]
        si_l = [self.state.det_il[i] for i in sel]
        sc_r = [self.state.det_cr[i] for i in sel]
        si_r = [self.state.det_ir[i] for i in sel]
        self._wc = CalibWorker(
            sc_l, si_l, sc_r, si_r,
            self.state.det_cl, self.state.det_il,
            self.state.det_cr, self.state.det_ir,
            self.state.img_size, self.state)
        self._wc.sig_status.connect(self.lbl_prog.setText)
        self._wc.sig_done.connect(self._on_calib_done)
        self._wc.sig_error.connect(self._on_error)
        self._wc.start()

    def _on_calib_done(self, res: dict):
        self.state.apply_calib(res)
        self._set_busy(False)
        self._populate_table()
        self._refresh_chart()
        self.sig_calib_done.emit()
        rms = f"L={self.state.rms_l:.3f}  R={self.state.rms_r:.3f}"
        if self.state.rms_s:
            rms += f"  Stereo={self.state.rms_s:.3f}"
        self.status(f"Calibration done — RMS {rms} px")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self._set_busy(False)

    def _populate_table(self):
        n  = len(self.state.det_orig)
        el = self.state.errs_l
        er = self.state.errs_r

        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(n)

        for k in range(n):
            orig = self.state.det_orig[k]

            # Col 0: original index; UserRole = detection array index k
            it0 = QTableWidgetItem(str(orig))
            it0.setData(Qt.UserRole, k)
            it0.setFlags(it0.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, self.COL_IDX, it0)

            # Col 1: left thumbnail (icon follows item on sort)
            it1 = QTableWidgetItem()
            it1.setData(Qt.DecorationRole, np2pixmap(self.state.imgs_l[orig], THUMB))
            it1.setFlags(it1.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, self.COL_THUMB_L, it1)

            # Col 2: left error
            e_l = float(el[k]) if el is not None and k < len(el) and not np.isnan(el[k]) else np.nan
            it2 = QTableWidgetItem("N/A" if np.isnan(e_l) else f"{e_l:.4f}")
            it2.setData(Qt.UserRole, e_l if not np.isnan(e_l) else 9999.0)
            it2.setFlags(it2.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, self.COL_ERR_L, it2)

            # Col 3: right thumbnail
            it3 = QTableWidgetItem()
            it3.setData(Qt.DecorationRole, np2pixmap(self.state.imgs_r[orig], THUMB))
            it3.setFlags(it3.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, self.COL_THUMB_R, it3)

            # Col 4: right error
            e_r = float(er[k]) if er is not None and k < len(er) and not np.isnan(er[k]) else np.nan
            it4 = QTableWidgetItem("N/A" if np.isnan(e_r) else f"{e_r:.4f}")
            it4.setData(Qt.UserRole, e_r if not np.isnan(e_r) else 9999.0)
            it4.setFlags(it4.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(k, self.COL_ERR_R, it4)

            # Col 5: checkbox via CheckState (follows item on sort)
            it5 = QTableWidgetItem()
            it5.setCheckState(Qt.Checked)
            it5.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            self.table.setItem(k, self.COL_KEEP, it5)

        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self._apply_threshold()

    def _on_item_changed(self, item):
        if item.column() == self.COL_KEEP:
            self._update_sel_label()
            self._chart_timer.start()

    def _apply_threshold(self):
        thr = self.dsb_thr.value()
        el  = self.state.errs_l
        er  = self.state.errs_r
        if el is None or er is None:
            return
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            it0 = self.table.item(row, self.COL_IDX)
            it5 = self.table.item(row, self.COL_KEEP)
            if it0 is None or it5 is None:
                continue
            k   = int(it0.data(Qt.UserRole))
            e_l = float(el[k]) if k < len(el) and not np.isnan(el[k]) else 9999.0
            e_r = float(er[k]) if k < len(er) and not np.isnan(er[k]) else 9999.0
            it5.setCheckState(Qt.Checked if max(e_l, e_r) <= thr else Qt.Unchecked)
        self.table.blockSignals(False)
        self._update_sel_label()
        self._chart_timer.start()

    def _set_all_checked(self, v: Qt.CheckState):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            it5 = self.table.item(row, self.COL_KEEP)
            if it5:
                it5.setCheckState(v)
        self.table.blockSignals(False)
        self._update_sel_label(); self._chart_timer.start()

    def _sel_all(self):  self._set_all_checked(Qt.Checked)
    def _sel_none(self): self._set_all_checked(Qt.Unchecked)

    def _sel_inv(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            it5 = self.table.item(row, self.COL_KEEP)
            if it5:
                it5.setCheckState(
                    Qt.Unchecked if it5.checkState() == Qt.Checked else Qt.Checked)
        self.table.blockSignals(False)
        self._update_sel_label(); self._chart_timer.start()

    def _update_sel_label(self):
        total = self.table.rowCount()
        sel   = sum(1 for r in range(total)
                    if (it := self.table.item(r, self.COL_KEEP))
                    and it.checkState() == Qt.Checked)
        self.lbl_sel.setText(f"{sel}/{total} selected")
        self.btn_recal.setText(f"🔄  Re-Calibrate  ({sel}/{total} images)")

    def _get_sel_det_indices(self) -> list[int]:
        sel = []
        for row in range(self.table.rowCount()):
            it0 = self.table.item(row, self.COL_IDX)
            it5 = self.table.item(row, self.COL_KEEP)
            if it0 and it5 and it5.checkState() == Qt.Checked:
                sel.append(int(it0.data(Qt.UserRole)))
        return sorted(sel)

    def _recalibrate(self):
        sel = self._get_sel_det_indices()
        if len(sel) < 4:
            QMessageBox.warning(self, "Too few images",
                "Select at least 4 image pairs for calibration.")
            return
        self._set_busy(True)
        self.lbl_prog.setText(f"Re-calibrating with {len(sel)} images…")
        self._run_calib(sel)

    def _refresh_chart(self):
        self.ax.clear()
        n  = len(self.state.det_orig)
        el = self.state.errs_l
        er = self.state.errs_r
        if n == 0 or el is None:
            self.canvas.draw(); return

        # Build selection map: det_idx → bool
        sel_map: dict[int, bool] = {}
        for row in range(self.table.rowCount()):
            it0 = self.table.item(row, self.COL_IDX)
            it5 = self.table.item(row, self.COL_KEEP)
            if it0:
                k = int(it0.data(Qt.UserRole))
                sel_map[k] = it5 is not None and it5.checkState() == Qt.Checked

        xs = np.arange(n)
        cl = ["#4CAF50" if sel_map.get(i, True) else "#F44336" for i in range(n)]
        cr = ["#2196F3" if sel_map.get(i, True) else "#FF9800" for i in range(n)]
        bw = 0.4
        self.ax.bar(xs - bw / 2, el[:n], width=bw, color=cl, alpha=0.85, label="Left")
        self.ax.bar(xs + bw / 2, er[:n], width=bw, color=cr, alpha=0.85, label="Right")
        thr = self.dsb_thr.value()
        self.ax.axhline(thr, color="red", lw=1.2, ls="--",
                        label=f"Threshold ({thr:.2f} px)")
        self.ax.set_xlabel("Detection pair index", fontsize=8)
        self.ax.set_ylabel("Reprojection error (px)", fontsize=8)
        self.ax.set_title("Per-image Reprojection Error", fontsize=9)
        self.ax.legend(fontsize=7); self.ax.tick_params(labelsize=7)
        self.fig.tight_layout(); self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Calibration Results
# ═══════════════════════════════════════════════════════════════════════════════

class ResultsTab(QWidget):
    def __init__(self, state: CalibState):
        super().__init__()
        self.state = state
        self._build()

    def _te(self, h=90) -> QTextEdit:
        te = QTextEdit(); te.setReadOnly(True)
        te.setFont(QFont("Monospace", 9)); te.setFixedHeight(h)
        return te

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        self.lbl_rms = QLabel("No calibration yet.")
        self.lbl_rms.setFont(QFont("Monospace", 10))
        self.lbl_rms.setAlignment(Qt.AlignCenter)
        root.addWidget(self.lbl_rms)

        spl = QSplitter(Qt.Horizontal)

        gl = QGroupBox("Left Camera"); vl = QVBoxLayout(gl)
        vl.addWidget(QLabel("K — camera matrix")); self.te_Kl = self._te(); vl.addWidget(self.te_Kl)
        vl.addWidget(QLabel("D — distortion"));    self.te_Dl = self._te(50); vl.addWidget(self.te_Dl)
        vl.addWidget(QLabel("rot_matrix  (camera → 1st pattern)")); self.te_Rl = self._te(); vl.addWidget(self.te_Rl)
        vl.addWidget(QLabel("t_left  (camera → 1st pattern)"));     self.te_tl = self._te(50); vl.addWidget(self.te_tl)
        spl.addWidget(gl)

        gr = QGroupBox("Right Camera"); vr = QVBoxLayout(gr)
        vr.addWidget(QLabel("K — camera matrix")); self.te_Kr = self._te(); vr.addWidget(self.te_Kr)
        vr.addWidget(QLabel("D — distortion"));    self.te_Dr = self._te(50); vr.addWidget(self.te_Dr)
        vr.addWidget(QLabel("rot_matrix  (camera → 1st pattern)")); self.te_Rr = self._te(); vr.addWidget(self.te_Rr)
        vr.addWidget(QLabel("t_right  (camera → 1st pattern)"));    self.te_tr = self._te(50); vr.addWidget(self.te_tr)
        spl.addWidget(gr)
        root.addWidget(spl, 1)

        gs = QGroupBox("Stereo Parameters"); hs = QHBoxLayout(gs)
        vR = QVBoxLayout(); vR.addWidget(QLabel("R — stereo rotation"))
        self.te_R = self._te(); vR.addWidget(self.te_R)
        vT = QVBoxLayout(); vT.addWidget(QLabel("T — stereo translation"))
        self.te_T = self._te(50); vT.addWidget(self.te_T)
        hs.addLayout(vR); hs.addLayout(vT)
        root.addWidget(gs)

        ge = QGroupBox("Per-Image Reprojection Errors"); ve = QVBoxLayout(ge)
        self.etable = QTableWidget(0, 4)
        self.etable.setHorizontalHeaderLabels(
            ["Det.#", "Orig.#", "Left Err (px)", "Right Err (px)"])
        self.etable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.etable.setMaximumHeight(180)
        ve.addWidget(self.etable)
        root.addWidget(ge)

    def refresh(self):
        s = self.state
        if not s.calibrated:
            return
        rms = f"RMS —  Left: {s.rms_l:.4f} px    Right: {s.rms_r:.4f} px"
        if s.rms_s:
            rms += f"    Stereo: {s.rms_s:.4f} px"
        self.lbl_rms.setText(rms)

        Rl_m, _ = cv2.Rodrigues(s.rv_l[0])
        Rr_m, _ = cv2.Rodrigues(s.rv_r[0])
        self.te_Kl.setText(fmt_mat(s.Kl)); self.te_Dl.setText(fmt_vec(s.Dl))
        self.te_Rl.setText(fmt_mat(Rl_m)); self.te_tl.setText(fmt_vec(s.tv_l[0]))
        self.te_Kr.setText(fmt_mat(s.Kr)); self.te_Dr.setText(fmt_vec(s.Dr))
        self.te_Rr.setText(fmt_mat(Rr_m)); self.te_tr.setText(fmt_vec(s.tv_r[0]))
        self.te_R.setText(fmt_mat(s.R) if s.R is not None else "N/A")
        self.te_T.setText(fmt_vec(s.T) if s.T is not None else "N/A")

        n = len(s.det_orig)
        self.etable.setRowCount(n)
        el = s.errs_l; er = s.errs_r
        for i in range(n):
            orig = s.det_orig[i]
            e_l  = float(el[i]) if el is not None and i < len(el) and not np.isnan(el[i]) else np.nan
            e_r  = float(er[i]) if er is not None and i < len(er) and not np.isnan(er[i]) else np.nan
            for col, val in enumerate([str(i), str(orig),
                                       f"{e_l:.4f}" if not np.isnan(e_l) else "N/A",
                                       f"{e_r:.4f}" if not np.isnan(e_r) else "N/A"]):
                self.etable.setItem(i, col, QTableWidgetItem(val))
            if not np.isnan(e_l) and not np.isnan(e_r):
                mx = max(e_l, e_r)
                c  = QColor("#ffcdd2" if mx > 1.5 else "#fff9c4" if mx > 0.8 else "#c8e6c9")
                for col in range(4):
                    it = self.etable.item(i, col)
                    if it:
                        it.setBackground(QBrush(c))


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Rectification
# ═══════════════════════════════════════════════════════════════════════════════

class RectTab(QWidget):
    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state   = state
        self.status  = status_fn
        self._sample = 0
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # Alpha
        ac = QHBoxLayout()
        ac.addWidget(QLabel("Alpha:"))
        self.cb_alpha = QComboBox()
        self.cb_alpha.addItems(["-1.0 (Default scaling)", "0.0 (Crop invalid pixels)", "1.0 (Preserve all pixels)"])
        self.cb_alpha.setCurrentIndex(1) # Default to 0.0
        ac.addWidget(self.cb_alpha)
        ac.addSpacing(16)
        self.btn_rect = QPushButton("⟳  Compute Rectification")
        self.btn_rect.setFixedHeight(34); self.btn_rect.clicked.connect(self._compute)
        ac.addWidget(self.btn_rect); ac.addStretch()
        root.addLayout(ac)

        # Sample nav
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev"); self.btn_prev.clicked.connect(self._prev)
        self.btn_next = QPushButton("Next ▶"); self.btn_next.clicked.connect(self._next)
        self.lbl_samp = QLabel("—")
        nav.addWidget(self.btn_prev); nav.addWidget(self.lbl_samp)
        nav.addWidget(self.btn_next); nav.addStretch()
        root.addLayout(nav)

        # Preview
        spl = QSplitter(Qt.Horizontal)
        self.lbl_L = self._preview_lbl("Left  (not computed)")
        self.lbl_R = self._preview_lbl("Right (not computed)")
        spl.addWidget(self.lbl_L); spl.addWidget(self.lbl_R)
        root.addWidget(spl, 1)

        self.lbl_roi = QLabel("ROI: —")
        self.lbl_roi.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl_roi)

    def _preview_lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter); lbl.setMinimumSize(400, 280)
        lbl.setStyleSheet("background: #111; color: #555; font-size: 13px;")
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return lbl

    def _compute(self):
        if not self.state.calibrated:
            QMessageBox.warning(self, "Not calibrated", "Run calibration first.")
            return
            
        idx = self.cb_alpha.currentIndex()
        if idx == 0: alpha = -1.0
        elif idx == 1: alpha = 0.0
        else: alpha = 1.0
        
        if self.state.rectify(alpha):
            self._show()
            self.status(f"Rectification computed  (α = {alpha:.1f})")
        else:
            QMessageBox.warning(self, "Failed",
                "Rectification failed — ensure stereo calibration succeeded.")

    def _show(self):
        n = len(self.state.det_orig)
        if not self.state.rectified or n == 0:
            return
        self._sample = max(0, min(self._sample, n - 1))
        self.lbl_samp.setText(f"Sample {self._sample + 1} / {n}")
        rl, rr = self.state.get_rect_sample(self._sample)
        if rl is None:
            return
        rl_ep = draw_epipolar(rl); rr_ep = draw_epipolar(rr)
        for roi, ep in ((self.state.roi_l, rl_ep), (self.state.roi_r, rr_ep)):
            if roi and any(roi):
                x, y, w, h = roi
                cv2.rectangle(ep, (x, y), (x + w, y + h), (255, 160, 0), 2)
        pw = max(self.lbl_L.width() - 4, 400)
        self.lbl_L.setPixmap(np2pixmap(rl_ep, pw))
        self.lbl_R.setPixmap(np2pixmap(rr_ep, pw))
        self.lbl_roi.setText(
            f"ROI Left: {self.state.roi_l}    ROI Right: {self.state.roi_r}")

    def _prev(self): self._sample = max(0, self._sample - 1); self._show()
    def _next(self):
        n = len(self.state.det_orig)
        self._sample = min(n - 1, self._sample + 1); self._show()

    def refresh(self):
        if self.state.rectified:
            self._show()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Export
# ═══════════════════════════════════════════════════════════════════════════════

def build_orbslam_yaml(state: CalibState, cam_type: str, scale: float) -> str:
    if not state.calibrated: return "Not calibrated."
    
    w = int(state.img_size[0] * scale)
    h = int(state.img_size[1] * scale)
    
    if cam_type == "PinHole":
        Kl, Dl = state.Kl.copy(), state.Dl.copy()
        Kr, Dr = state.Kr.copy(), state.Dr.copy()
        Kl[:2, :] *= scale
        Kr[:2, :] *= scale
        
        fx1, fy1, cx1, cy1 = Kl[0,0], Kl[1,1], Kl[0,2], Kl[1,2]
        fx2, fy2, cx2, cy2 = Kr[0,0], Kr[1,1], Kr[0,2], Kr[1,2]
        
        k1_1, k2_1, p1_1, p2_1, k3_1 = Dl.flatten()[:5] if Dl is not None and len(Dl.flatten()) >= 5 else (0,0,0,0,0)
        k1_2, k2_2, p1_2, p2_2, k3_2 = Dr.flatten()[:5] if Dr is not None and len(Dr.flatten()) >= 5 else (0,0,0,0,0)
        
        T_c1_c2 = np.eye(4)
        if state.R is not None:
            T_c1_c2[:3, :3] = state.R
            # Convert translation from mm to meters
            T_c1_c2[:3, 3] = state.T.flatten() / 1000.0
            
        # baseline in meters
        b = (np.linalg.norm(state.T) / 1000.0) * fx1 if state.T is not None else 0.0
    else:
        if not state.rectified:
            return "Rectification not computed yet (go to Rectification tab and set Alpha)."
        
        Pl, Pr = state.Pl.copy(), state.Pr.copy()
        Pl[:2, :] *= scale
        Pr[:2, :] *= scale
        
        fx1, fy1, cx1, cy1 = Pl[0,0], Pl[1,1], Pl[0,2], Pl[1,2]
        fx2, fy2, cx2, cy2 = Pr[0,0], Pr[1,1], Pr[0,2], Pr[1,2]
        
        k1_1 = k2_1 = p1_1 = p2_1 = k3_1 = 0.0
        k1_2 = k2_2 = p1_2 = p2_2 = k3_2 = 0.0
        
        T_c1_c2 = np.eye(4)
        # Pr[0,3] is -fx * baseline_in_mm, so Tx is in mm. Convert to meters.
        Tx = (Pr[0, 3] / Pr[0, 0]) / 1000.0 if Pr[0, 0] != 0 else 0.0
        T_c1_c2[0, 3] = Tx
        
        # baseline in meters * fx
        b = abs(Tx) * fx1

    def _fmt(val): return _yaml_float(val)
    
    return f"""%YAML:1.0
---
File.version: "1.0"

Camera.type: "{cam_type}"

# --- Câmera 1 ---
Camera1.fx: {_fmt(fx1)}
Camera1.fy: {_fmt(fy1)}
Camera1.cx: {_fmt(cx1)}
Camera1.cy: {_fmt(cy1)}
Camera1.k1: {_fmt(k1_1)}
Camera1.k2: {_fmt(k2_1)}
Camera1.k3: {_fmt(k3_1)}
Camera1.p1: {_fmt(p1_1)}
Camera1.p2: {_fmt(p2_1)}

# --- Câmera 2 ---
Camera2.fx: {_fmt(fx2)}
Camera2.fy: {_fmt(fy2)}
Camera2.cx: {_fmt(cx2)}
Camera2.cy: {_fmt(cy2)}
Camera2.k1: {_fmt(k1_2)}
Camera2.k2: {_fmt(k2_2)}
Camera2.k3: {_fmt(k3_2)}
Camera2.p1: {_fmt(p1_2)}
Camera2.p2: {_fmt(p2_2)}

Camera.width: {w}
Camera.height: {h}

Camera.fps: 30
Camera.RGB: 1

Stereo.ThDepth: 40.0
Stereo.b: {_fmt(b)}

Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{_flat(T_c1_c2)}]

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1500

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# FAST thresholds: try iniThFAST in each grid cell, then minThFAST if needed.
# Lower these values for low-contrast images.
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""

class ExportTab(QWidget):
    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state  = state
        self.status = status_fn
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # Format
        gf = QGroupBox("Export Format"); vf = QVBoxLayout(gf)
        self.chk_custom = QCheckBox("Custom YAML  (project format — both cameras in one file)")
        self.chk_passive = QCheckBox("Passive stereo capture YAML (OpenCV — translation in metres)")
        self.chk_passive.setToolTip("Exports K1/D1 left, K2/D2 right, R and T at calibration resolution. Board dimensions are entered in mm; T is exported in metres.")
        self.chk_ci_l   = QCheckBox("CameraInfo YAML — Left camera  (ROS2 format)")
        self.chk_ci_r   = QCheckBox("CameraInfo YAML — Right camera  (ROS2 format)")
        self.chk_orb_pinhole = QCheckBox("ORB-SLAM3 YAML (PinHole)")
        self.chk_orb_rect    = QCheckBox("ORB-SLAM3 YAML (Rectified)")
        self.chk_rect   = QCheckBox("Include rectification matrices for Custom/CameraInfo (requires Rectification tab)")
        
        for c in (self.chk_custom, self.chk_passive, self.chk_ci_l, self.chk_ci_r, self.chk_orb_pinhole, self.chk_orb_rect, self.chk_rect):
            if c in (self.chk_custom, self.chk_passive, self.chk_ci_l, self.chk_ci_r, self.chk_rect):
                c.setChecked(True)
            else:
                c.setChecked(False)
            vf.addWidget(c)
            c.stateChanged.connect(self._update_preview)
            
        row = QHBoxLayout()
        row.addWidget(QLabel("ORB-SLAM3 Rescale Factor:"))
        self.dsb_rescale = QDoubleSpinBox()
        self.dsb_rescale.setRange(0.1, 10.0); self.dsb_rescale.setValue(1.0); self.dsb_rescale.setSingleStep(0.1)
        self.dsb_rescale.valueChanged.connect(self._update_preview)
        row.addWidget(self.dsb_rescale); row.addStretch()
        vf.addLayout(row)
        
        note = QLabel("Passive stereo: calibration image resolution, unscaled intrinsics, T in metres.\nCalibrate with native images for native-resolution capture; ORB rescale applies only to ORB-SLAM3.")
        note.setWordWrap(True)
        vf.addWidget(note)
        root.addWidget(gf)

        # Metadata
        gm = QGroupBox("Metadata"); mg = QGridLayout(gm)
        mg.addWidget(QLabel("Image width (px):"),  0, 0)
        self.sb_w = QSpinBox(); self.sb_w.setRange(1, 20000); self.sb_w.setValue(1600)
        mg.addWidget(self.sb_w, 0, 1)
        mg.addWidget(QLabel("Image height (px):"), 0, 2)
        self.sb_h = QSpinBox(); self.sb_h.setRange(1, 20000); self.sb_h.setValue(1200)
        mg.addWidget(self.sb_h, 0, 3)
        mg.addWidget(QLabel("Camera name left:"),  1, 0)
        self.le_nl = QLineEdit("left");  mg.addWidget(self.le_nl, 1, 1)
        mg.addWidget(QLabel("Camera name right:"), 1, 2)
        self.le_nr = QLineEdit("right"); mg.addWidget(self.le_nr, 1, 3)
        mg.addWidget(QLabel("Output directory:"),  2, 0)
        self.le_out = QLineEdit()
        self.le_out.setPlaceholderText("Select output directory…")
        mg.addWidget(self.le_out, 2, 1, 1, 2)
        bd = QPushButton("Browse…"); bd.clicked.connect(self._browse_dir)
        mg.addWidget(bd, 2, 3)
        mg.addWidget(QLabel("Filename prefix:"), 3, 0)
        self.le_pfx = QLineEdit(datetime.now().strftime("%Y%m%d"))
        mg.addWidget(self.le_pfx, 3, 1)
        root.addWidget(gm)

        for w in (self.sb_w, self.sb_h):
            w.valueChanged.connect(self._update_preview)
        for w in (self.le_nl, self.le_nr, self.le_pfx):
            w.textChanged.connect(self._update_preview)

        # Preview
        gp = QGroupBox("YAML Preview"); vp = QVBoxLayout(gp)
        self.cb_prev = QComboBox()
        self.cb_prev.addItems(["Custom YAML", "CameraInfo Left", "CameraInfo Right", "ORB-SLAM3 PinHole", "ORB-SLAM3 Rectified", "Passive stereo capture (metres)"])
        self.cb_prev.currentIndexChanged.connect(self._update_preview)
        vp.addWidget(self.cb_prev)
        self.te_prev = QTextEdit()
        self.te_prev.setReadOnly(True); self.te_prev.setFont(QFont("Monospace", 9))
        vp.addWidget(self.te_prev)
        root.addWidget(gp, 1)

        btn_exp = QPushButton("💾  Export")
        btn_exp.setFixedHeight(38); btn_exp.clicked.connect(self._export)
        root.addWidget(btn_exp)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.le_out.setText(d)

    def refresh(self):
        s = self.state
        if s.img_size != (0, 0):
            self.sb_w.setValue(s.img_size[0]); self.sb_h.setValue(s.img_size[1])
        self._update_preview()

    def _update_preview(self):
        if not self.state.calibrated:
            self.te_prev.setText("# Calibrate first to see preview"); return
        w, h  = self.sb_w.value(), self.sb_h.value()
        inc   = self.chk_rect.isChecked()
        which = self.cb_prev.currentText()
        try:
            if which == "Custom YAML":
                txt = build_custom_yaml(self.state, inc)
            elif which == "CameraInfo Left":
                txt = build_camerainfo_yaml(self.state, "left",  self.le_nl.text(), w, h, inc)
            elif which == "CameraInfo Right":
                txt = build_camerainfo_yaml(self.state, "right", self.le_nr.text(), w, h, inc)
            elif which == "ORB-SLAM3 PinHole":
                txt = build_orbslam_yaml(self.state, "PinHole", self.dsb_rescale.value())
            elif which == "ORB-SLAM3 Rectified":
                txt = build_orbslam_yaml(self.state, "Rectified", self.dsb_rescale.value())
            elif which == "Passive stereo capture (metres)":
                txt = build_passive_stereo_yaml(self.state)
            else:
                txt = "Unknown format."
            self.te_prev.setText(txt)
        except Exception as e:
            self.te_prev.setText(f"# Preview error: {e}")

    def _export(self):
        s = self.state
        if not s.calibrated:
            QMessageBox.warning(self, "Not calibrated", "Run calibration first."); return
        out = self.le_out.text().strip()
        if not out:
            QMessageBox.warning(self, "No directory", "Select an output directory."); return
        os.makedirs(out, exist_ok=True)
        pfx  = self.le_pfx.text().strip() or "calib"
        w, h = self.sb_w.value(), self.sb_h.value()
        inc  = self.chk_rect.isChecked()
        saved = []
        try:
            if self.chk_custom.isChecked():
                p = os.path.join(out, f"{pfx}_stereo.yaml")
                Path(p).write_text(build_custom_yaml(s, inc)); saved.append(p)
            if self.chk_passive.isChecked():
                p = os.path.join(out, f"{pfx}_passive_stereo_capture.yaml")
                Path(p).write_text(build_passive_stereo_yaml(s)); saved.append(p)
            if self.chk_ci_l.isChecked():
                p = os.path.join(out, f"{pfx}_{self.le_nl.text()}.yaml")
                Path(p).write_text(build_camerainfo_yaml(s, "left",  self.le_nl.text(), w, h, inc))
                saved.append(p)
            if self.chk_ci_r.isChecked():
                p = os.path.join(out, f"{pfx}_{self.le_nr.text()}.yaml")
                Path(p).write_text(build_camerainfo_yaml(s, "right", self.le_nr.text(), w, h, inc))
                saved.append(p)
            if self.chk_orb_pinhole.isChecked():
                p = os.path.join(out, f"{pfx}_orbslam3_pinhole.yaml")
                Path(p).write_text(build_orbslam_yaml(s, "PinHole", self.dsb_rescale.value()))
                saved.append(p)
            if self.chk_orb_rect.isChecked():
                p = os.path.join(out, f"{pfx}_orbslam3_rectified.yaml")
                Path(p).write_text(build_orbslam_yaml(s, "Rectified", self.dsb_rescale.value()))
                saved.append(p)
            if not saved:
                QMessageBox.warning(self, "Nothing selected", "Select at least one format.")
                return
            self.status("Exported: " + ", ".join(saved))
            QMessageBox.information(self, "Done", "Saved:\n" + "\n".join(saved))
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Distortion Visualization
# ═══════════════════════════════════════════════════════════════════════════════

class DistortionTab(QWidget):
    """Visualises radial & tangential distortion fields from calibration data."""

    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state  = state
        self.status = status_fn
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # Controls
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Camera:"))
        self.cb_cam = QComboBox(); self.cb_cam.addItems(["Left", "Right"])
        ctl.addWidget(self.cb_cam)
        ctl.addSpacing(16)
        ctl.addWidget(QLabel("Component:"))
        self.cb_mode = QComboBox(); self.cb_mode.addItems(["All", "Radial", "Tangential"])
        ctl.addWidget(self.cb_mode)
        ctl.addSpacing(16)
        ctl.addWidget(QLabel("Grid step (px):"))
        self.sb_step = QSpinBox(); self.sb_step.setRange(10, 200); self.sb_step.setValue(40)
        ctl.addWidget(self.sb_step)
        ctl.addSpacing(16)
        self.btn_plot = QPushButton("⟳  Plot Distortion")
        self.btn_plot.setFixedHeight(34); self.btn_plot.clicked.connect(self._plot)
        ctl.addWidget(self.btn_plot); ctl.addStretch()
        root.addLayout(ctl)

        # Matplotlib canvas
        self.fig = Figure(figsize=(14, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        # Info label
        self.lbl_info = QLabel("Calibrate first, then plot distortion.")
        self.lbl_info.setStyleSheet("color: #555; font-size: 11px;")
        root.addWidget(self.lbl_info)

    def _compute_field(self, K, D, w, h, mode, step):
        """Compute distortion displacement field."""
        x = np.arange(0, w + step, step)
        y = np.arange(0, h + step, step)
        u, v = np.meshgrid(x, y)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        d = D.flatten()
        k1 = d[0] if len(d) > 0 else 0
        k2 = d[1] if len(d) > 1 else 0
        p1 = d[2] if len(d) > 2 else 0
        p2 = d[3] if len(d) > 3 else 0
        k3 = d[4] if len(d) > 4 else 0

        xn = (u - cx) / fx
        yn = (v - cy) / fy
        r2 = xn**2 + yn**2

        if mode == "Radial":
            rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            u_d = fx * (xn * rad) + cx
            v_d = fy * (yn * rad) + cy
        elif mode == "Tangential":
            dx_t = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
            dy_t = p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
            u_d = fx * (xn + dx_t) + cx
            v_d = fy * (yn + dy_t) + cy
        else:  # All
            rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            dx_t = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn**2)
            dy_t = p1 * (r2 + 2 * yn**2) + 2 * p2 * xn * yn
            u_d = fx * (xn * rad + dx_t) + cx
            v_d = fy * (yn * rad + dy_t) + cy

        return u, v, u_d - u, v_d - v

    def _plot(self):
        s = self.state
        if not s.calibrated:
            QMessageBox.warning(self, "Not calibrated", "Run calibration first.")
            return

        cam = self.cb_cam.currentText().lower()
        mode = self.cb_mode.currentText()
        step = self.sb_step.value()
        K = s.Kl if cam == "left" else s.Kr
        D = s.Dl if cam == "left" else s.Dr
        w, h = s.img_size

        self.fig.clear()

        if mode == "All":
            # Show radial and tangential side-by-side
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)
            for ax, m, title in [(ax1, "Radial", "Radial Component"),
                                  (ax2, "Tangential", "Tangential Component")]:
                u, v, du, dv = self._compute_field(K, D, w, h, m, step)
                mag = np.sqrt(du**2 + dv**2)
                cs = ax.contourf(u, v, mag, 20, cmap='Greys', alpha=0.15)
                cont = ax.contour(u, v, mag, levels=10, colors='black', linewidths=0.6)
                ax.clabel(cont, inline=True, fontsize=8, fmt='%.1f')
                ax.quiver(u, v, du, dv, color='royalblue', alpha=0.8,
                          scale_units='xy', scale=0.2, width=0.002, headwidth=3)
                ax.set_title(f"{cam.upper()} — {title}", fontsize=11, fontweight='bold')
                ax.set_aspect('equal'); ax.invert_yaxis()
                ax.set_xlim(0, w); ax.set_ylim(h, 0)
                ax.grid(True, linestyle=':', alpha=0.5)
        else:
            ax = self.fig.add_subplot(1, 1, 1)
            u, v, du, dv = self._compute_field(K, D, w, h, mode, step)
            mag = np.sqrt(du**2 + dv**2)
            q = ax.quiver(u, v, du, dv, mag, cmap='viridis',
                          scale_units='xy', scale=0.1, alpha=0.9)
            self.fig.colorbar(q, ax=ax, label='Distortion (px)')
            ax.set_title(f"{cam.upper()} — {mode} Distortion", fontsize=12, fontweight='bold')
            ax.invert_yaxis(); ax.set_aspect('equal')
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
            ax.grid(True, linestyle=':', alpha=0.3)

        d = D.flatten()
        info = (f"Focal: ({K[0,0]:.2f}, {K[1,1]:.2f})  "
                f"PP: ({K[0,2]:.2f}, {K[1,2]:.2f})  "
                f"k1={d[0]:.5f}  k2={d[1]:.5f}  p1={d[2]:.6f}  p2={d[3]:.6f}"
                + (f"  k3={d[4]:.5f}" if len(d) > 4 else ""))
        self.lbl_info.setText(info)
        self.canvas.draw()
        self.status(f"Distortion plotted — {cam} camera, {mode} mode")

    def refresh(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Color Histogram
# ═══════════════════════════════════════════════════════════════════════════════

class HistogramTab(QWidget):
    """Interactive per-channel histogram with switchable color spaces."""

    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state  = state
        self.status = status_fn
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # Controls row
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Camera:"))
        self.cb_cam = QComboBox(); self.cb_cam.addItems(["Left", "Right"])
        ctl.addWidget(self.cb_cam)
        ctl.addSpacing(12)

        ctl.addWidget(QLabel("Image:"))
        self.sb_idx = QSpinBox(); self.sb_idx.setRange(0, 0)
        ctl.addWidget(self.sb_idx)
        ctl.addSpacing(12)

        ctl.addWidget(QLabel("Color Space:"))
        self.cb_cs = QComboBox()
        self.cb_cs.addItems(["BGR", "RGB", "HSV", "LAB", "YCrCb", "Grayscale"])
        ctl.addWidget(self.cb_cs)
        ctl.addSpacing(12)

        self.btn_plot = QPushButton("⟳  Plot")
        self.btn_plot.setFixedHeight(34); self.btn_plot.clicked.connect(self._plot)
        ctl.addWidget(self.btn_plot); ctl.addStretch()
        root.addLayout(ctl)

        # Layout: image preview on left, histogram on right
        spl = QSplitter(Qt.Horizontal)

        # Image preview
        self.lbl_img = QLabel("Load images first"); self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setMinimumSize(400, 300)
        spl.addWidget(self.lbl_img)

        # Histogram canvas
        self.fig = Figure(figsize=(7, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        spl.addWidget(self.canvas)

        spl.setStretchFactor(0, 1); spl.setStretchFactor(1, 2)
        root.addWidget(spl, 1)

    def _get_img(self):
        """Return the selected image (BGR)."""
        cam = self.cb_cam.currentText().lower()
        imgs = self.state.imgs_l if cam == "left" else self.state.imgs_r
        if not imgs:
            return None
        idx = min(self.sb_idx.value(), len(imgs) - 1)
        return imgs[idx]

    def _plot(self):
        img = self._get_img()
        if img is None:
            QMessageBox.warning(self, "No images", "Load images first."); return

        cs = self.cb_cs.currentText()

        # Convert color space
        if cs == "RGB":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ch_names = ["Red", "Green", "Blue"]
            ch_colors = ["red", "green", "blue"]
        elif cs == "HSV":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ch_names = ["Hue", "Saturation", "Value"]
            ch_colors = ["orange", "cyan", "gray"]
        elif cs == "LAB":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            ch_names = ["L (Lightness)", "A (Green-Red)", "B (Blue-Yellow)"]
            ch_colors = ["black", "green", "blue"]
        elif cs == "YCrCb":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ch_names = ["Y (Luma)", "Cr (Red-diff)", "Cb (Blue-diff)"]
            ch_colors = ["gray", "red", "blue"]
        elif cs == "Grayscale":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch_names = ["Intensity"]
            ch_colors = ["black"]
        else:  # BGR
            conv = img
            ch_names = ["Blue", "Green", "Red"]
            ch_colors = ["blue", "green", "red"]

        # Show converted image as preview
        if conv.ndim == 2:
            disp = cv2.cvtColor(conv, cv2.COLOR_GRAY2BGR)
        elif cs == "RGB":
            disp = cv2.cvtColor(conv, cv2.COLOR_RGB2BGR)
        elif cs in ("HSV", "LAB", "YCrCb"):
            # show original for non-RGB spaces (converting back would be misleading for display)
            disp = img
        else:
            disp = conv

        pw = max(self.lbl_img.width() - 4, 400)
        self.lbl_img.setPixmap(np2pixmap(disp, pw))

        # Plot histograms
        self.fig.clear()
        n_ch = len(ch_names)
        axes = [self.fig.add_subplot(n_ch, 1, i + 1) for i in range(n_ch)]

        for i, (ax, name, color) in enumerate(zip(axes, ch_names, ch_colors)):
            if conv.ndim == 2:
                ch = conv
            else:
                ch = conv[:, :, i]
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
            ax.fill_between(range(256), hist.flatten(), alpha=0.5, color=color)
            ax.plot(hist, color=color, linewidth=0.8)
            ax.set_xlim(0, 255)
            ax.set_ylabel(name, fontsize=9)
            ax.tick_params(labelsize=8)
            if i < n_ch - 1:
                ax.set_xticklabels([])

        axes[-1].set_xlabel("Pixel value", fontsize=9)
        self.fig.suptitle(f"{cs} Histogram", fontsize=11, fontweight='bold')
        self.canvas.draw()
        self.status(f"Histogram — {cs} space, {self.cb_cam.currentText()} camera, image #{self.sb_idx.value()}")

    def refresh(self):
        n = max(len(self.state.imgs_l), len(self.state.imgs_r))
        self.sb_idx.setRange(0, max(0, n - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 7 — CLAHE Enhancement
# ═══════════════════════════════════════════════════════════════════════════════

class CLAHETab(QWidget):
    """Interactive CLAHE with Grayscale and LAB (L-channel) modes."""

    def __init__(self, state: CalibState, status_fn):
        super().__init__()
        self.state  = state
        self.status = status_fn
        self._build()

    def _build(self):
        root = QVBoxLayout(self); root.setSpacing(6)

        # Controls
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Camera:"))
        self.cb_cam = QComboBox(); self.cb_cam.addItems(["Left", "Right"])
        ctl.addWidget(self.cb_cam)
        ctl.addSpacing(12)

        ctl.addWidget(QLabel("Image:"))
        self.sb_idx = QSpinBox(); self.sb_idx.setRange(0, 0)
        ctl.addWidget(self.sb_idx)
        ctl.addSpacing(12)

        ctl.addWidget(QLabel("Clip Limit:"))
        self.dsb_clip = QDoubleSpinBox()
        self.dsb_clip.setRange(0.1, 40.0); self.dsb_clip.setValue(2.0)
        self.dsb_clip.setSingleStep(0.5)
        ctl.addWidget(self.dsb_clip)
        ctl.addSpacing(12)

        ctl.addWidget(QLabel("Grid Size:"))
        self.sb_grid = QSpinBox(); self.sb_grid.setRange(1, 64); self.sb_grid.setValue(8)
        ctl.addWidget(self.sb_grid)
        ctl.addSpacing(12)

        self.btn_apply = QPushButton("⟳  Apply CLAHE")
        self.btn_apply.setFixedHeight(34); self.btn_apply.clicked.connect(self._apply)
        ctl.addWidget(self.btn_apply); ctl.addStretch()
        root.addLayout(ctl)

        # Auto-update on parameter change
        self.dsb_clip.valueChanged.connect(self._apply)
        self.sb_grid.valueChanged.connect(self._apply)
        self.cb_cam.currentIndexChanged.connect(self._apply)
        self.sb_idx.valueChanged.connect(self._apply)

        # Labels for the 4 images: Original | Grayscale CLAHE | LAB Original | LAB CLAHE
        grid = QGridLayout()

        self.lbl_title_orig = QLabel("Original"); self.lbl_title_orig.setAlignment(Qt.AlignCenter)
        self.lbl_title_orig.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_title_gray = QLabel("Grayscale CLAHE"); self.lbl_title_gray.setAlignment(Qt.AlignCenter)
        self.lbl_title_gray.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_title_lab_orig = QLabel("LAB (Original)"); self.lbl_title_lab_orig.setAlignment(Qt.AlignCenter)
        self.lbl_title_lab_orig.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.lbl_title_lab_clahe = QLabel("LAB CLAHE (L channel)"); self.lbl_title_lab_clahe.setAlignment(Qt.AlignCenter)
        self.lbl_title_lab_clahe.setStyleSheet("font-weight: bold; font-size: 12px;")

        grid.addWidget(self.lbl_title_orig, 0, 0)
        grid.addWidget(self.lbl_title_gray, 0, 1)
        grid.addWidget(self.lbl_title_lab_orig, 2, 0)
        grid.addWidget(self.lbl_title_lab_clahe, 2, 1)

        self.lbl_orig = QLabel(); self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_gray = QLabel(); self.lbl_gray.setAlignment(Qt.AlignCenter)
        self.lbl_lab  = QLabel(); self.lbl_lab.setAlignment(Qt.AlignCenter)
        self.lbl_lab_c = QLabel(); self.lbl_lab_c.setAlignment(Qt.AlignCenter)

        for lbl in (self.lbl_orig, self.lbl_gray, self.lbl_lab, self.lbl_lab_c):
            lbl.setMinimumSize(300, 220)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid.addWidget(self.lbl_orig, 1, 0)
        grid.addWidget(self.lbl_gray, 1, 1)
        grid.addWidget(self.lbl_lab,  3, 0)
        grid.addWidget(self.lbl_lab_c, 3, 1)

        root.addLayout(grid, 1)

    def _get_img(self):
        cam = self.cb_cam.currentText().lower()
        imgs = self.state.imgs_l if cam == "left" else self.state.imgs_r
        if not imgs:
            return None
        idx = min(self.sb_idx.value(), len(imgs) - 1)
        return imgs[idx]

    def _apply(self):
        img = self._get_img()
        if img is None:
            return

        clip = self.dsb_clip.value()
        grid = self.sb_grid.value()
        tile = (grid, grid)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)

        # Compute target pixel width from label size
        pw = max(min(self.lbl_orig.width(), self.lbl_orig.height()) - 4, 300)

        # 1) Original (as-is)
        self.lbl_orig.setPixmap(np2pixmap(img, pw))

        # 2) Grayscale CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_clahe = clahe.apply(gray)
        self.lbl_gray.setPixmap(np2pixmap(gray_clahe, pw))

        # 3) LAB original (show back as BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.lbl_lab.setPixmap(np2pixmap(img, pw))  # show original for comparison

        # 4) LAB CLAHE on L channel, merged back to BGR
        L, A, B = cv2.split(lab)
        L_clahe = clahe.apply(L)
        lab_enhanced = cv2.merge([L_clahe, A, B])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        self.lbl_lab_c.setPixmap(np2pixmap(result, pw))

        self.status(f"CLAHE — clip={clip:.1f}, grid={grid}x{grid}, "
                    f"{self.cb_cam.currentText()} camera, image #{self.sb_idx.value()}")

    def refresh(self):
        n = max(len(self.state.imgs_l), len(self.state.imgs_r))
        self.sb_idx.setRange(0, max(0, n - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Main window
# ═══════════════════════════════════════════════════════════════════════════════

class CalibGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration GUI")
        self.resize(1380, 840)

        state = CalibState()
        self._tabs = QTabWidget()

        self.t_review  = ImageReviewTab(state, self._status)
        self.t_results = ResultsTab(state)
        self.t_rect    = RectTab(state, self._status)
        self.t_export  = ExportTab(state, self._status)
        self.t_distort = DistortionTab(state, self._status)
        self.t_hist    = HistogramTab(state, self._status)
        self.t_clahe   = CLAHETab(state, self._status)

        self._tabs.addTab(self.t_review,  "📷  Image Review")
        self._tabs.addTab(self.t_results, "📊  Results")
        self._tabs.addTab(self.t_rect,    "↔  Rectification")
        self._tabs.addTab(self.t_export,  "💾  Export")
        self._tabs.addTab(self.t_distort, "🔍  Distortion")
        self._tabs.addTab(self.t_hist,    "📊  Histogram")
        self._tabs.addTab(self.t_clahe,   "🎨  CLAHE")

        self.t_review.sig_calib_done.connect(self._on_calib_done)
        self._tabs.currentChanged.connect(self._on_tab)

        self.setCentralWidget(self._tabs)
        self.statusBar().showMessage(
            "Ready — select a folder with left/ and right/ sub-dirs, then click Detect & Calibrate")

    def _status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _on_calib_done(self):
        self.t_results.refresh()
        self.t_export.refresh()
        self.t_hist.refresh()
        self.t_clahe.refresh()

    def _on_tab(self, idx: int):
        {1: self.t_results.refresh,
         2: self.t_rect.refresh,
         3: self.t_export.refresh,
         4: self.t_distort.refresh,
         5: self.t_hist.refresh,
         6: self.t_clahe.refresh}.get(idx, lambda: None)()


# ─── entry points ─────────────────────────────────────────────────────────────

def launch_calibration_gui():
    # ── Fix Qt platform plugin conflict ──────────────────────────────────────
    # cv2 (opencv-python) overrides QT_QPA_PLATFORM_PLUGIN_PATH to its own
    # bundled Qt plugins on import. Those libs are ABI-incompatible with PyQt5,
    # crashing xcb. Force-override to PyQt5's own platforms dir here, AFTER
    # cv2 has already been imported, so it takes effect before QApplication().
    import PyQt5 as _pyqt5
    _qt_platforms = os.path.join(
        os.path.dirname(_pyqt5.__file__), "Qt5", "plugins", "platforms")
    if os.path.isdir(_qt_platforms):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_platforms
    # ─────────────────────────────────────────────────────────────────────────
    app = QApplication.instance() or QApplication(sys.argv)
    win = CalibGUI()
    win.show()
    app.exec_()


if __name__ == "__main__":
    launch_calibration_gui()
