"""
GUI Interativo de Calibração de Cor Macbeth para Câmeras FLIR.

Fluxo:
  1. Abrir imagem (Ctrl+O ou botão).
  2. Selecionar patch na lista lateral (clique, Tab/Shift+Tab, ↑/↓).
  3. Desenhar bounding box na imagem (clicar e arrastar).
  4. Repetir para todos os 24 patches (mínimo 8 para calibrar).
  5. Clicar em "Calibrar" ou pressionar Space.
  6. Visualizar plots embutidos na aba de resultados.
  7. Salvar YAML + script PySpin (Ctrl+S ou botão).

Atalhos de teclado:
  Tab / Shift+Tab ou ↓ / ↑  : próximo / anterior patch
  Delete                      : limpar ROI do patch atual
  Ctrl+O                      : abrir imagem
  Ctrl+S                      : salvar resultados
  Ctrl+Z                      : desfazer última ROI
  Space                       : executar calibração
  Escape                      : cancelar desenho em andamento
"""

import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

if __package__:
    from .macbeth_generator import MACBETH_PATCHES
    from .color_calibration import ColorCalibration
else:
    from macbeth_generator import MACBETH_PATCHES
    from color_calibration import ColorCalibration


# ─────────────────────────── colour helpers ───────────────────────────────────

def srgb_tuple(r, g, b) -> str:
    """Returns a #RRGGBB hex string clamped to [0, 255]."""
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
    )

def patch_reference_rgb(idx: int) -> Tuple[int, int, int]:
    return MACBETH_PATCHES[idx]["srgb"]

def luminance(r, g, b) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def enhance_dark_preview(image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """Brighten the display in stops without modifying calibration pixels."""
    lut = np.clip(np.arange(256, dtype=np.float64) * 2.0 ** exposure, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


# ─────────────────────────── ImageCanvas ──────────────────────────────────────

class ImageCanvas(tk.Frame):
    """
    Scrollable, zoomable canvas that lets the user draw one bounding box at a time.
    The canvas always works in *image space*; zoom only affects rendering.
    """

    ZOOM_STEP = 1.2
    ZOOM_MIN  = 0.1
    ZOOM_MAX  = 10.0

    # visual config
    COLOR_UNSET   = "#888888"
    COLOR_ACTIVE  = "#f0c040"
    COLOR_DONE    = "#44cc66"
    COLOR_HOVER   = "#ff5533"
    TEXT_UNSET    = "white"
    TEXT_ACTIVE   = "black"
    TEXT_DONE     = "black"

    def __init__(self, parent, on_roi_drawn, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_roi_drawn = on_roi_drawn   # callback(patch_idx, x, y, w, h)

        # image state
        self._cv_image: Optional[np.ndarray] = None  # BGR original
        self._pil_image: Optional[Image.Image] = None
        self._zoom: float = 1.0
        self._tk_image = None  # keep reference

        # ROI drawing state
        self._draw_start: Optional[Tuple[int, int]] = None  # canvas coords
        self._draw_rect_id = None
        self._active_patch: int = 0

        # stored rois: list of 24, each None or (x, y, w, h) in image space
        self._rois: List[Optional[Tuple[int, int, int, int]]] = [None] * 24
        self._rect_ids: Dict[int, int] = {}   # patch_idx -> canvas rect item id
        self._label_ids: Dict[int, int] = {}  # patch_idx -> canvas text item id

        # widgets
        self._hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self._vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self._canvas = tk.Canvas(
            self,
            bg="#2b2b2b",
            cursor="crosshair",
            xscrollcommand=self._hbar.set,
            yscrollcommand=self._vbar.set,
        )
        self._hbar.config(command=self._canvas.xview)
        self._vbar.config(command=self._canvas.yview)

        self._hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._vbar.pack(side=tk.RIGHT,  fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # bindings
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_drag)
        self._canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._canvas.bind("<MouseWheel>",       self._on_mousewheel)
        self._canvas.bind("<Button-4>",         self._on_mousewheel)   # Linux scroll up
        self._canvas.bind("<Button-5>",         self._on_mousewheel)   # Linux scroll down
        self._canvas.bind("<Escape>",           self._cancel_draw)

    # ── public API ────────────────────────────────────────────────────────────

    def load_image(self, cv_img: np.ndarray):
        """Load a new BGR image and reset all ROIs."""
        self._cv_image = cv_img
        self._rois = [None] * 24
        self._rect_ids.clear()
        self._label_ids.clear()
        self._zoom = 1.0
        self._fit_image()
        self._redraw_canvas()

    def set_active_patch(self, idx: int):
        self._active_patch = idx
        self._redraw_all_rois()

    def clear_roi(self, idx: int):
        self._rois[idx] = None
        self._redraw_all_rois()

    def get_rois(self) -> List[Optional[Tuple[int, int, int, int]]]:
        return list(self._rois)

    def set_roi(self, idx: int, roi: Optional[Tuple[int, int, int, int]]):
        """Set ROI from external source (e.g. undo)."""
        self._rois[idx] = roi
        self._redraw_all_rois()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fit_image(self):
        if self._cv_image is None:
            return
        canvas_w = self._canvas.winfo_width()  or 800
        canvas_h = self._canvas.winfo_height() or 600
        img_h, img_w = self._cv_image.shape[:2]
        self._zoom = min(canvas_w / img_w, canvas_h / img_h, 1.0)

    def _img_to_canvas(self, x, y) -> Tuple[float, float]:
        return x * self._zoom, y * self._zoom

    def _canvas_to_img(self, cx, cy) -> Tuple[float, float]:
        # account for scroll offset
        ox = self._canvas.canvasx(0)
        oy = self._canvas.canvasy(0)
        return (cx + ox) / self._zoom, (cy + oy) / self._zoom

    def _redraw_canvas(self):
        self._canvas.delete("all")
        if self._cv_image is None:
            return
        img_h, img_w = self._cv_image.shape[:2]
        dw = max(1, int(img_w * self._zoom))
        dh = max(1, int(img_h * self._zoom))
        rgb = cv2.cvtColor(self._cv_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(pil)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image, tags="img")
        self._canvas.config(scrollregion=(0, 0, dw, dh))
        self._redraw_all_rois()

    def _redraw_all_rois(self):
        # remove old rect/label canvas items
        for item_id in list(self._rect_ids.values()) + list(self._label_ids.values()):
            self._canvas.delete(item_id)
        self._rect_ids.clear()
        self._label_ids.clear()

        for idx in range(24):
            roi = self._rois[idx]
            if roi is None:
                continue
            x, y, w, h = roi
            cx0, cy0 = self._img_to_canvas(x, y)
            cx1, cy1 = self._img_to_canvas(x + w, y + h)

            if idx == self._active_patch:
                outline = self.COLOR_ACTIVE
                fill_alpha = ""
                width = 2
                text_col = self.TEXT_ACTIVE
            else:
                outline = self.COLOR_DONE
                fill_alpha = ""
                width = 1
                text_col = self.TEXT_DONE

            rid = self._canvas.create_rectangle(
                cx0, cy0, cx1, cy1,
                outline=outline, width=width, tags=f"roi_{idx}"
            )
            # reference color swatch inside the rect corner
            ref_r, ref_g, ref_b = patch_reference_rgb(idx)
            lid = self._canvas.create_text(
                cx0 + 2, cy0 + 2,
                text=f"{idx+1}", fill=outline,
                anchor=tk.NW, font=("Helvetica", 8, "bold"),
                tags=f"label_{idx}"
            )
            self._rect_ids[idx]  = rid
            self._label_ids[idx] = lid

    # ── mouse events ──────────────────────────────────────────────────────────

    def _on_press(self, event):
        if self._cv_image is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        self._draw_start = (ix, iy)
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def _on_drag(self, event):
        if self._draw_start is None or self._cv_image is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        sx, sy = self._draw_start
        cx0, cy0 = self._img_to_canvas(min(sx, ix), min(sy, iy))
        cx1, cy1 = self._img_to_canvas(max(sx, ix), max(sy, iy))
        if self._draw_rect_id:
            self._canvas.coords(self._draw_rect_id, cx0, cy0, cx1, cy1)
        else:
            self._draw_rect_id = self._canvas.create_rectangle(
                cx0, cy0, cx1, cy1,
                outline=self.COLOR_ACTIVE, width=2, dash=(4, 2), tags="drawing"
            )

    def _on_release(self, event):
        if self._draw_start is None or self._cv_image is None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        sx, sy = self._draw_start
        self._draw_start = None
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

        x = int(min(sx, ix))
        y = int(min(sy, iy))
        w = int(abs(ix - sx))
        h = int(abs(iy - sy))
        if w < 3 or h < 3:
            return  # too small, ignore

        img_h, img_w = self._cv_image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        self._previous_roi = self._rois[self._active_patch]
        self._rois[self._active_patch] = (x, y, w, h)
        self._redraw_all_rois()
        self._on_roi_drawn(self._active_patch, x, y, w, h)

    def _cancel_draw(self, event=None):
        self._draw_start = None
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def _on_mousewheel(self, event):
        # determine direction
        if event.num == 4 or event.delta > 0:
            factor = self.ZOOM_STEP
        else:
            factor = 1.0 / self.ZOOM_STEP

        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        self._zoom = new_zoom
        self._redraw_canvas()

    def zoom_in(self):
        self._zoom = min(self.ZOOM_MAX, self._zoom * self.ZOOM_STEP)
        self._redraw_canvas()

    def zoom_out(self):
        self._zoom = max(self.ZOOM_MIN, self._zoom / self.ZOOM_STEP)
        self._redraw_canvas()

    def zoom_fit(self):
        self._fit_image()
        self._redraw_canvas()


# ─────────────────────────── PatchListPanel ───────────────────────────────────

class PatchListPanel(tk.Frame):
    """Scrollable list of 24 Macbeth patches with status indicators."""

    ICON_NONE   = "⬜"
    ICON_DONE   = "✅"

    def __init__(self, parent, on_select, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_select = on_select

        tk.Label(self, text="Patches Macbeth", font=("Helvetica", 10, "bold"),
                 bg=self["bg"]).pack(pady=(4, 2))

        container = tk.Frame(self, bg=self["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        vbar = tk.Scrollbar(container, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            container,
            yscrollcommand=vbar.set,
            font=("Helvetica", 9),
            selectbackground="#f0c040",
            selectforeground="black",
            activestyle="none",
            height=24,
            width=22,
        )
        vbar.config(command=self._listbox.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for p in MACBETH_PATCHES:
            self._listbox.insert(tk.END, f"{self.ICON_NONE} {p['id']:2d}. {p['name']}")

        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._listbox.selection_set(0)

    def _on_listbox_select(self, event):
        sel = self._listbox.curselection()
        if sel:
            self._on_select(sel[0])

    def update_status(self, rois: List):
        current_sel = self._listbox.curselection()
        sel_idx = current_sel[0] if current_sel else 0

        self._listbox.delete(0, tk.END)
        for i, p in enumerate(MACBETH_PATCHES):
            icon = self.ICON_DONE if rois[i] is not None else self.ICON_NONE
            self._listbox.insert(tk.END, f"{icon} {p['id']:2d}. {p['name']}")

        self._listbox.selection_set(sel_idx)
        self._listbox.see(sel_idx)

    def select(self, idx: int):
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(idx)
        self._listbox.see(idx)

    def get_selected(self) -> int:
        sel = self._listbox.curselection()
        return sel[0] if sel else 0


# ─────────────────────────── PatchPreviewPanel ────────────────────────────────

class PatchPreviewPanel(tk.Frame):
    """Shows captured colour vs. reference for the active patch."""

    SWATCH_W = 80
    SWATCH_H = 50

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        bg = self["bg"]

        tk.Label(self, text="Prévia do Patch", font=("Helvetica", 9, "bold"),
                 bg=bg).pack(pady=(4, 2))

        row = tk.Frame(self, bg=bg)
        row.pack(fill=tk.X, padx=6)

        # captured swatch
        col_cap = tk.Frame(row, bg=bg)
        col_cap.pack(side=tk.LEFT, expand=True)
        tk.Label(col_cap, text="Capturado", font=("Helvetica", 8), bg=bg).pack()
        self._cap_canvas = tk.Canvas(col_cap, width=self.SWATCH_W, height=self.SWATCH_H,
                                     bg="#444", bd=1, relief=tk.SUNKEN)
        self._cap_canvas.pack()
        self._cap_lbl = tk.Label(col_cap, text="—", font=("Courier", 7), bg=bg)
        self._cap_lbl.pack()

        # reference swatch
        col_ref = tk.Frame(row, bg=bg)
        col_ref.pack(side=tk.RIGHT, expand=True)
        tk.Label(col_ref, text="Referência", font=("Helvetica", 8), bg=bg).pack()
        self._ref_canvas = tk.Canvas(col_ref, width=self.SWATCH_W, height=self.SWATCH_H,
                                     bg="#444", bd=1, relief=tk.SUNKEN)
        self._ref_canvas.pack()
        self._ref_lbl = tk.Label(col_ref, text="—", font=("Courier", 7), bg=bg)
        self._ref_lbl.pack()

        # patch name label
        self._name_lbl = tk.Label(self, text="", font=("Helvetica", 8, "italic"), bg=bg)
        self._name_lbl.pack(pady=(4, 0))

    def update(self, patch_idx: int, captured_rgb: Optional[Tuple[float, float, float]]):
        p = MACBETH_PATCHES[patch_idx]
        ref_r, ref_g, ref_b = p["srgb"]
        self._name_lbl.config(text=f"{p['id']}. {p['name']}")

        # reference swatch
        ref_hex = srgb_tuple(ref_r, ref_g, ref_b)
        self._ref_canvas.config(bg=ref_hex)
        self._ref_lbl.config(text=f"R:{ref_r:3d} G:{ref_g:3d} B:{ref_b:3d}")

        # captured swatch
        if captured_rgb is not None:
            cr, cg, cb = captured_rgb
            cap_hex = srgb_tuple(cr, cg, cb)
            self._cap_canvas.config(bg=cap_hex)
            self._cap_lbl.config(text=f"R:{int(cr):3d} G:{int(cg):3d} B:{int(cb):3d}")
        else:
            self._cap_canvas.config(bg="#444")
            self._cap_lbl.config(text="—")


# ─────────────────────────── CalibrationResultsPanel ─────────────────────────

class CalibrationResultsPanel(tk.Frame):
    """Embedded matplotlib figure showing the calibration diagnostic plots."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._fig = None
        self._canvas_widget = None
        self._placeholder = tk.Label(
            self,
            text="Execute a calibração para ver os resultados aqui.",
            font=("Helvetica", 10), fg="#888", bg=self["bg"],
            wraplength=280, justify=tk.CENTER,
        )
        self._placeholder.pack(expand=True)

    def show_results(self, calib: ColorCalibration):
        if self._canvas_widget:
            self._canvas_widget.get_tk_widget().destroy()
        self._placeholder.pack_forget()

        S_corrected = calib.apply_ccm(calib.S)
        metrics = calib.evaluate_metrics()

        fig, axes = plt.subplots(2, 1, figsize=(5.5, 7),
                                 gridspec_kw={"height_ratios": [1.6, 1]})
        fig.patch.set_facecolor("#2b2b2b")
        for ax in axes:
            ax.set_facecolor("#2b2b2b")
            ax.tick_params(colors="#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")

        # ── top: colour swatches ──────────────────────────────────────────────
        ax0 = axes[0]
        ax0.set_xlim(0, 24)
        ax0.set_ylim(0, 3)
        ax0.set_title("Alvo / Capturado / Corrigido", color="#eeeeee", fontsize=9)
        ax0.set_xticks(np.arange(24) + 0.5)
        ax0.set_xticklabels([str(i + 1) for i in range(24)], fontsize=7)
        ax0.set_yticks([0.5, 1.5, 2.5])
        ax0.set_yticklabels(["Corrigido", "Capturado", "Alvo"], fontsize=8,
                             fontweight="bold", color="#cccccc")

        for idx in range(24):
            t_col = np.clip(calib.T[:, idx] / calib.max_val, 0.0, 1.0)
            s_col = np.clip(calib.S[:, idx] / calib.max_val, 0.0, 1.0)
            c_col = np.clip(S_corrected[:, idx] / calib.max_val, 0.0, 1.0)
            ax0.add_patch(mpatches.Rectangle((idx, 2), 1, 1, facecolor=t_col, edgecolor="none"))
            ax0.add_patch(mpatches.Rectangle((idx, 1), 1, 1, facecolor=s_col, edgecolor="none"))
            ax0.add_patch(mpatches.Rectangle((idx, 0), 1, 1, facecolor=c_col, edgecolor="none"))
        ax0.axhline(1, color="#888", lw=1)
        ax0.axhline(2, color="#888", lw=1)

        # ── bottom: ΔE bar chart ──────────────────────────────────────────────
        ax1 = axes[1]
        indices = np.arange(24)
        w = 0.38
        de_b = metrics["per_patch_de_before"]
        de_a = metrics["per_patch_de_after"]
        ax1.bar(indices - w/2, de_b, w, label="Antes", color="#e74c3c", alpha=0.9)
        ax1.bar(indices + w/2, de_a, w, label="Depois (CCM)", color="#2ecc71", alpha=0.9)
        ax1.set_xlabel("Patch #", color="#cccccc", fontsize=8)
        ax1.set_ylabel("ΔE (CIE76)", color="#cccccc", fontsize=8)
        ax1.set_xticks(indices)
        ax1.set_xticklabels([str(i + 1) for i in range(24)], fontsize=7)
        ax1.legend(fontsize=7, facecolor="#3a3a3a", labelcolor="#cccccc", framealpha=0.8)
        rmse_b = metrics["rmse_before"]
        rmse_a = metrics["rmse_after"]
        pct    = metrics["improvement_pct"]
        ax1.set_title(
            f"ΔE médio: {metrics['delta_e_mean_before']:.2f} → {metrics['delta_e_mean_after']:.2f}  |  "
            f"RMSE: {rmse_b:.1f} → {rmse_a:.1f}  ({pct:.0f}% melhoria)",
            color="#eeeeee", fontsize=8
        )

        plt.tight_layout(pad=1.2)
        self._canvas_widget = FigureCanvasTkAgg(fig, master=self)
        self._canvas_widget.draw()
        self._canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)


# ─────────────────────────── CCM Info Panel ───────────────────────────────────

class CcmInfoPanel(tk.Frame):
    """Shows the computed CCM values and FLIR gain assignments."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        bg = self["bg"]
        tk.Label(self, text="Matriz CCM (Ganhos FLIR)", font=("Helvetica", 9, "bold"),
                 bg=bg).pack(pady=(6, 2))
        self._text = tk.Text(
            self, font=("Courier", 8), bg="#1e1e1e", fg="#ccffcc",
            relief=tk.FLAT, state=tk.DISABLED, height=14, width=36,
        )
        self._text.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

    def update(self, calib: ColorCalibration):
        if calib.CCM is None:
            return
        gains = calib.get_flir_gains()
        m = calib.CCM
        metrics = calib.evaluate_metrics()
        lines = [
            "  CCM  |   R      G      B  ",
            f"  R    | {m[0,0]:+.4f} {m[0,1]:+.4f} {m[0,2]:+.4f}",
            f"  G    | {m[1,0]:+.4f} {m[1,1]:+.4f} {m[1,2]:+.4f}",
            f"  B    | {m[2,0]:+.4f} {m[2,1]:+.4f} {m[2,2]:+.4f}",
            "",
            "  Parâmetros FLIR:",
            f"  Gain00={gains['Gain00']:+.5f}  Gain01={gains['Gain01']:+.5f}",
            f"  Gain02={gains['Gain02']:+.5f}",
            f"  Gain10={gains['Gain10']:+.5f}  Gain11={gains['Gain11']:+.5f}",
            f"  Gain12={gains['Gain12']:+.5f}",
            f"  Gain20={gains['Gain20']:+.5f}  Gain21={gains['Gain21']:+.5f}",
            f"  Gain22={gains['Gain22']:+.5f}",
            "",
            f"  RMSE: {metrics['rmse_before']:.2f}→{metrics['rmse_after']:.2f}"
            f" ({metrics['improvement_pct']:.0f}% ↑)",
            f"  ΔE médio: {metrics['delta_e_mean_before']:.2f}→"
            f"{metrics['delta_e_mean_after']:.2f}",
        ]
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.insert(tk.END, "\n".join(lines))
        self._text.config(state=tk.DISABLED)


# ─────────────────────────── Main Application ────────────────────────────────

class ColorCalibrationApp(tk.Tk):
    """Main application window."""

    BG_DARK = "#2b2b2b"
    BG_SIDE = "#1e1e1e"

    def __init__(self):
        super().__init__()
        self.title("Calibração de Cor Macbeth — FLIR")
        self.geometry("1280x780")
        self.minsize(900, 600)
        self.configure(bg=self.BG_DARK)

        # state
        self._cv_image: Optional[np.ndarray] = None
        self._image_path: str = ""
        self._calib: Optional[ColorCalibration] = None
        self._active_patch: int = 0
        self._undo_stack: List[Tuple[int, Optional[Tuple]]] = []  # (patch_idx, prev_roi)
        self._camera = "Esquerda"
        self._sessions = {}
        self._camera_var = tk.StringVar(value=self._camera)
        self._exposure = tk.DoubleVar(value=0.0)
        self._ccm_method = tk.StringVar(value="white_point_preserving")

        self._build_ui()
        self._bind_keys()
        self._update_status()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg="#3c3c3c", pady=3)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_cfg = dict(bg="#555", fg="white", relief=tk.FLAT, padx=8, pady=3,
                       font=("Helvetica", 9), activebackground="#777",
                       activeforeground="white", cursor="hand2")

        tk.Button(toolbar, text="📂 Abrir Imagem", command=self._open_image, **btn_cfg).pack(side=tk.LEFT, padx=(4, 2))
        tk.Button(toolbar, text="🗑 Limpar Tudo",  command=self._clear_all,  **btn_cfg).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="↩ Desfazer",     command=self._undo,       **btn_cfg).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        tk.Label(toolbar, text="Método:", bg="#3c3c3c", fg="#ccc",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        method_menu = ttk.Combobox(
            toolbar, textvariable=self._ccm_method, width=22, state="readonly",
            values=["white_point_preserving", "flir_least_squares", "weighted_least_squares"],
            font=("Helvetica", 9),
        )
        method_menu.pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self._btn_calibrate = tk.Button(
            toolbar, text="▶ Calibrar", command=self._run_calibration,
            bg="#225522", fg="white", relief=tk.FLAT, padx=10, pady=3,
            font=("Helvetica", 9, "bold"), activebackground="#448844",
            activeforeground="white", cursor="hand2", state=tk.DISABLED,
        )
        self._btn_calibrate.pack(side=tk.LEFT, padx=2)

        self._btn_save = tk.Button(
            toolbar, text="💾 Salvar", command=self._save_results,
            bg="#225577", fg="white", relief=tk.FLAT, padx=8, pady=3,
            font=("Helvetica", 9), activebackground="#4477aa",
            activeforeground="white", cursor="hand2", state=tk.DISABLED,
        )
        self._btn_save.pack(side=tk.LEFT, padx=2)

        # zoom buttons
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=6)
        tk.Button(toolbar, text="🔍+", command=lambda: self._img_canvas.zoom_in(),  **btn_cfg).pack(side=tk.RIGHT, padx=2)
        tk.Button(toolbar, text="🔍−", command=lambda: self._img_canvas.zoom_out(), **btn_cfg).pack(side=tk.RIGHT, padx=2)
        tk.Button(toolbar, text="⊡",   command=lambda: self._img_canvas.zoom_fit(), **btn_cfg).pack(side=tk.RIGHT, padx=2)

        controls = tk.Frame(self, bg="#3c3c3c")
        controls.pack(fill=tk.X)
        tk.Button(controls, text="Abrir par estéreo", command=self._open_stereo).pack(side=tk.LEFT, padx=4)
        camera_menu = ttk.Combobox(controls, textvariable=self._camera_var,
                                  values=["Esquerda", "Direita"], state="readonly", width=12)
        camera_menu.pack(side=tk.LEFT, padx=4)
        camera_menu.bind("<<ComboboxSelected>>", lambda e: self._switch_camera())
        tk.Label(controls, text="Luz da prévia (EV):", bg="#3c3c3c", fg="white").pack(side=tk.LEFT)
        tk.Scale(controls, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL,
                 variable=self._exposure, command=lambda value: self._refresh_preview(),
                 length=180).pack(side=tk.LEFT)
        tk.Label(controls, text="Calibração usa a imagem original. ≥8 patches por câmera.",
                 bg="#3c3c3c", fg="#ccc").pack(side=tk.LEFT, padx=8)

        # ── main area: image + sidebar ────────────────────────────────────────
        main = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=self.BG_DARK,
                              sashwidth=4, sashrelief=tk.FLAT)
        main.pack(fill=tk.BOTH, expand=True)

        # left: image canvas
        left = tk.Frame(main, bg=self.BG_DARK)
        self._img_canvas = ImageCanvas(left, on_roi_drawn=self._on_roi_drawn,
                                       bg=self.BG_DARK)
        self._img_canvas.pack(fill=tk.BOTH, expand=True)

        # instruction label below image
        self._instr_lbl = tk.Label(
            left, text="Abra uma imagem para começar.",
            bg="#333", fg="#aaa", font=("Helvetica", 8), anchor=tk.W,
        )
        self._instr_lbl.pack(fill=tk.X, padx=2, pady=1)
        main.add(left, minsize=400)

        # right: sidebar notebook
        right = tk.Frame(main, bg=self.BG_SIDE, width=320)
        right.pack_propagate(False)

        sidebar_nb = ttk.Notebook(right)
        sidebar_nb.pack(fill=tk.BOTH, expand=True)

        # tab 1: patch list + preview
        tab_patches = tk.Frame(sidebar_nb, bg=self.BG_SIDE)
        sidebar_nb.add(tab_patches, text="Patches")

        self._patch_list = PatchListPanel(
            tab_patches, on_select=self._select_patch, bg=self.BG_SIDE
        )
        self._patch_list.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        ttk.Separator(tab_patches, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        self._preview = PatchPreviewPanel(tab_patches, bg=self.BG_SIDE)
        self._preview.pack(fill=tk.X, padx=2, pady=2)

        # nav buttons
        nav_row = tk.Frame(tab_patches, bg=self.BG_SIDE)
        nav_row.pack(fill=tk.X, pady=4)
        tk.Button(nav_row, text="◀ Anterior", command=self._prev_patch,
                  bg="#444", fg="white", relief=tk.FLAT, padx=6, pady=2,
                  font=("Helvetica", 8), cursor="hand2").pack(side=tk.LEFT, padx=4)
        tk.Button(nav_row, text="Próximo ▶", command=self._next_patch,
                  bg="#444", fg="white", relief=tk.FLAT, padx=6, pady=2,
                  font=("Helvetica", 8), cursor="hand2").pack(side=tk.RIGHT, padx=4)
        tk.Button(nav_row, text="🗑 Limpar ROI", command=self._clear_current_roi,
                  bg="#662222", fg="white", relief=tk.FLAT, padx=6, pady=2,
                  font=("Helvetica", 8), cursor="hand2").pack(side=tk.LEFT, padx=4)

        # tab 2: calibration results
        tab_results = tk.Frame(sidebar_nb, bg=self.BG_SIDE)
        sidebar_nb.add(tab_results, text="Resultados")

        self._results_panel = CalibrationResultsPanel(tab_results, bg=self.BG_SIDE)
        self._results_panel.pack(fill=tk.BOTH, expand=True)

        self._ccm_panel = CcmInfoPanel(tab_results, bg=self.BG_SIDE)
        self._ccm_panel.pack(fill=tk.X, padx=2, pady=2)

        self._sidebar_nb = sidebar_nb
        main.add(right, minsize=300)

        # ── status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Pronto.")
        status = tk.Label(self, textvariable=self._status_var,
                          bg="#1a1a1a", fg="#aaa", anchor=tk.W,
                          font=("Helvetica", 8), padx=6)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # initialise preview with patch 0
        self._preview.update(0, None)

    # ── keyboard bindings ─────────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind("<Control-o>",      lambda e: self._open_image())
        self.bind("<Control-s>",      lambda e: self._save_results())
        self.bind("<Control-z>",      lambda e: self._undo())
        self.bind("<space>",          lambda e: self._run_calibration())
        self.bind("<Tab>",            lambda e: (self._next_patch(), "break"))
        self.bind("<Shift-Tab>",      lambda e: (self._prev_patch(), "break"))
        self.bind("<Down>",           lambda e: self._next_patch())
        self.bind("<Up>",             lambda e: self._prev_patch())
        self.bind("<Delete>",         lambda e: self._clear_current_roi())
        self.bind("<Return>",         lambda e: self._next_unset_patch())

    # ── event handlers ────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Abrir Imagem do Gráfico Macbeth",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                       ("Todos os arquivos", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{path}")
            return
        self._sessions = {}
        self._load_camera(self._camera, path, img)

    def _remember_camera(self):
        if self._cv_image is not None:
            self._sessions[self._camera] = dict(image=self._cv_image, path=self._image_path,
                rois=self._img_canvas.get_rois(), undo=list(self._undo_stack),
                calib=self._calib, exposure=self._exposure.get())

    def _load_camera(self, camera, path, image):
        self._camera = camera
        self._camera_var.set(camera)
        self._cv_image, self._image_path = image, path
        self._calib = None
        self._undo_stack = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        level = float(np.percentile(gray, 90))
        self._exposure.set(min(5.0, max(0.0, np.log2(160.0 / max(level, 1.0)))))
        self._img_canvas.load_image(enhance_dark_preview(image, self._exposure.get()))
        self._patch_list.update_status([None] * 24)
        self._sidebar_nb.select(0)
        self._select_patch(0)
        self._remember_camera()
        self._update_status()

    def _open_stereo(self):
        pair = []
        for camera in ("Esquerda", "Direita"):
            path = filedialog.askopenfilename(title=f"Imagem da câmera {camera}",
                filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")])
            if not path:
                return
            image = cv2.imread(path)
            if image is None:
                messagebox.showerror("Erro", f"Não foi possível abrir: {path}")
                return
            pair.append((camera, path, image))
        self._sessions = {}
        for camera, path, image in pair:
            self._load_camera(camera, path, image)
        self._camera_var.set("Esquerda")
        self._switch_camera()

    def _switch_camera(self):
        target = self._camera_var.get()
        if target == self._camera:
            return
        self._remember_camera()
        if target not in self._sessions:
            self._camera_var.set(self._camera)
            messagebox.showinfo("Câmera", "Use Abrir par estéreo para carregar as duas câmeras.")
            return
        state = self._sessions[target]
        self._camera = target
        self._cv_image, self._image_path = state["image"], state["path"]
        self._calib, self._undo_stack = state["calib"], list(state["undo"])
        self._exposure.set(state["exposure"])
        self._img_canvas.load_image(enhance_dark_preview(self._cv_image, self._exposure.get()))
        self._img_canvas._rois = list(state["rois"])
        self._img_canvas._redraw_all_rois()
        self._patch_list.update_status(state["rois"])
        self._select_patch(0)
        if self._calib is not None:
            self._results_panel.show_results(self._calib)
            self._ccm_panel.update(self._calib)
        else:
            self._sidebar_nb.select(0)
        self._update_status()

    def _refresh_preview(self):
        if self._cv_image is not None:
            self._img_canvas._cancel_draw()
            self._img_canvas._cv_image = enhance_dark_preview(self._cv_image, self._exposure.get())
            self._img_canvas._redraw_canvas()

    def _invalidate_calibration(self):
        self._calib = None
        if self._camera in self._sessions:
            self._sessions[self._camera]["calib"] = None
        self._btn_save.config(state=tk.DISABLED)

    def _on_roi_drawn(self, patch_idx: int, x: int, y: int, w: int, h: int):
        """Called by ImageCanvas whenever a new box is drawn."""
        self._invalidate_calibration()
        # push undo
        old_roi = self._img_canvas._previous_roi
        self._undo_stack.append((patch_idx, old_roi))

        # update patch list status
        self._patch_list.update_status(self._img_canvas.get_rois())

        # sample colour and update preview
        if self._cv_image is not None:
            roi_crop = self._cv_image[y:y+h, x:x+w]
            mean_bgr = np.mean(roi_crop, axis=(0, 1))
            cap_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
            self._preview.update(patch_idx, cap_rgb)

        self._update_status()
        # auto-advance to next unset patch
        self.after(120, self._next_unset_patch)

    def _select_patch(self, idx: int):
        self._active_patch = idx
        self._img_canvas.set_active_patch(idx)
        self._patch_list.select(idx)

        # update preview
        rois = self._img_canvas.get_rois()
        cap_rgb = None
        if rois[idx] is not None and self._cv_image is not None:
            x, y, w, h = rois[idx]
            roi_crop = self._cv_image[y:y+h, x:x+w]
            mean_bgr = np.mean(roi_crop, axis=(0, 1))
            cap_rgb = (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
        self._preview.update(idx, cap_rgb)

        name = MACBETH_PATCHES[idx]["name"]
        if self._cv_image is not None:
            status = "já definida" if rois[idx] is not None else "não definida"
            self._instr_lbl.config(
                text=f"Patch {idx+1}/24: {name} — {status}. Arraste para desenhar a ROI."
            )
        self._update_status()

    def _next_patch(self):
        self._select_patch((self._active_patch + 1) % 24)

    def _prev_patch(self):
        self._select_patch((self._active_patch - 1) % 24)

    def _next_unset_patch(self):
        """Jump to the next patch that has no ROI yet."""
        rois = self._img_canvas.get_rois()
        for offset in range(1, 25):
            idx = (self._active_patch + offset) % 24
            if rois[idx] is None:
                self._select_patch(idx)
                return
        # all set — stay on current

    def _clear_current_roi(self):
        self._invalidate_calibration()
        idx = self._active_patch
        old_roi = self._img_canvas.get_rois()[idx]
        self._undo_stack.append((idx, old_roi))
        self._img_canvas.clear_roi(idx)
        self._patch_list.update_status(self._img_canvas.get_rois())
        self._preview.update(idx, None)
        self._update_status()

    def _clear_all(self):
        if not messagebox.askyesno("Limpar", "Limpar todas as ROIs selecionadas?"):
            return
        for i in range(24):
            self._img_canvas.clear_roi(i)
        self._patch_list.update_status([None] * 24)
        self._undo_stack.clear()
        self._calib = None
        self._btn_save.config(state=tk.DISABLED)
        self._preview.update(self._active_patch, None)
        self._update_status()

    def _undo(self):
        if not self._undo_stack:
            self._status_var.set("Nada para desfazer.")
            return
        self._invalidate_calibration()
        patch_idx, prev_roi = self._undo_stack.pop()
        self._img_canvas.set_roi(patch_idx, prev_roi)
        self._patch_list.update_status(self._img_canvas.get_rois())
        self._select_patch(patch_idx)
        self._update_status()

    def _run_calibration(self):
        if self._cv_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        self._remember_camera()
        for camera, state in self._sessions.items():
            count = sum(r is not None for r in state["rois"])
            if count < 8:
                messagebox.showwarning("Patches insuficientes",
                    f"{camera}: {count}/24 patches. Defina ao menos 8 patches por câmera.")
                return
        self._status_var.set("Calculando CCM…")
        self.update_idletasks()
        try:
            results = {}
            for camera, state in self._sessions.items():
                calib = ColorCalibration(target_type="standard_srgb")
                calib.extract_patches_from_rois(state["image"], state["rois"])
                calib.solve_ccm(method=self._ccm_method.get())
                results[camera] = calib
            for camera, calib in results.items():
                self._sessions[camera]["calib"] = calib
            self._calib = results[self._camera]
            self._results_panel.show_results(self._calib)
            self._ccm_panel.update(self._calib)
            self._sidebar_nb.select(1)
            self._btn_save.config(state=tk.NORMAL)
            self._status_var.set("Calibração concluída: " + ", ".join(results) +
                ". Troque a câmera para ver seus resultados.")
        except Exception as exc:
            messagebox.showerror("Erro na calibração", str(exc))
            self._status_var.set(f"Erro: {exc}")

    def _save_results(self):
        if self._calib is None:
            messagebox.showwarning("Aviso", "Execute a calibração antes de salvar.")
            return

        self._remember_camera()
        if any(state["calib"] is None for state in self._sessions.values()):
            messagebox.showwarning("Aviso", "Calibre todas as câmeras antes de salvar.")
            return
        out_dir = filedialog.askdirectory(title="Selecione o diretório de saída",
                                          initialdir=os.path.dirname(self._image_path) or ".")
        if not out_dir:
            return
        try:
            saved = []
            stereo = len(self._sessions) == 2
            if stereo:
                yaml_name = "stereo_color_ccm.yaml"
                with open(os.path.join(out_dir, yaml_name), "w", encoding="utf-8") as f:
                    for camera, side in (("Esquerda", "left"), ("Direita", "right")):
                        matrix = self._sessions[camera]["calib"].CCM
                        values = np.asarray(matrix, dtype=float).reshape(9).tolist()
                        f.write(f"color_ccm_{side}: {json.dumps(values, allow_nan=False)}\n")
                saved.append(yaml_name)
            for camera, state in self._sessions.items():
                base = os.path.splitext(os.path.basename(state["path"]))[0]
                if len(self._sessions) == 2:
                    base += "_left" if camera == "Esquerda" else "_right"
                calib = state["calib"]
                if not stereo:
                    calib.export_flir_yaml(os.path.join(out_dir, f"{base}_flir_ccm.yaml"))
                calib.plot_calibration_results(os.path.join(out_dir, f"{base}_calibration_results.png"))
                with open(os.path.join(out_dir, f"{base}_apply_flir_ccm.py"), "w", encoding="utf-8") as f:
                    f.write(calib.generate_pyspin_snippet())
                saved.append(base)
            messagebox.showinfo("Salvo com sucesso", f"YAML, plot e script PySpin salvos em:\n{out_dir}\n" + "\n".join(saved))
            self._status_var.set(f"Resultados salvos em: {out_dir}")
        except Exception as exc:
            messagebox.showerror("Erro ao salvar", str(exc))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_status(self):
        rois = self._img_canvas.get_rois()
        n_set = sum(1 for r in rois if r is not None)
        self._remember_camera()
        ready = bool(self._sessions) and all(sum(r is not None for r in state["rois"]) >= 8
                                            for state in self._sessions.values())
        self._btn_calibrate.config(state=tk.NORMAL if ready else tk.DISABLED)
        self._btn_save.config(state=tk.NORMAL if self._sessions and all(
            state["calib"] is not None for state in self._sessions.values()) else tk.DISABLED)
        self.title(f"Calibração de Cor Macbeth — {self._camera} — {os.path.basename(self._image_path)}")
        if self._cv_image is not None:
            self._status_var.set(
                f"{self._camera}: {n_set}/24 patches definidos.  "
                f"Patch ativo: {self._active_patch + 1} ({MACBETH_PATCHES[self._active_patch]['name']}).  "
                f"{'Pronto para calibrar!' if ready else 'Defina ≥8 patches em cada câmera.'}"
            )


# ──────────────────────────────────────────────────────────────────────────────

def launch_gui():
    app = ColorCalibrationApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
