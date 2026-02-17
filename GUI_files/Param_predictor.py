"""
Interface for alsi10mg SLM Process Parameter Predictor
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QMessageBox, QGroupBox, QGridLayout, QCheckBox,
                             QTextEdit, QSlider, QFrame, QSizePolicy, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import (QFont, QColor, QPainter, QPen, QBrush, QLinearGradient,
                          QRadialGradient, QPainterPath, QFontDatabase)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# FIX for PyTorch 2.10+ weights_only issue
import torch.serialization
try:
    from numpy._core.multiarray import scalar as numpy_scalar
    torch.serialization.add_safe_globals([numpy_scalar])
except ImportError:
    try:
        from numpy.core.multiarray import scalar as numpy_scalar
        torch.serialization.add_safe_globals([numpy_scalar])
    except ImportError:
        pass

PROJECT_ROOT = r"c:\MLAM\DUI_KIMI\alsi10mg_optimization_framework"

# ─── DARK THEME PALETTE ────────────────────────────────────────────────────────
DARK = {
    "bg_deep":       "#0A0D14",
    "bg_panel":      "#0F1420",
    "bg_card":       "#141B2D",
    "bg_elevated":   "#1A2238",
    "border":        "#1F2B45",
    "border_glow":   "#2A3F6F",
    "text_primary":  "#E8EDF7",
    "text_secondary":"#8899BB",
    "text_muted":    "#4A5A7A",
    "accent_blue":   "#00BFFF",
    "accent_cyan":   "#00FFEA",
    "accent_orange": "#FF6B35",
    "accent_purple": "#A855F7",
    "accent_green":  "#00FF88",
    "accent_red":    "#FF3860",
    "neon_blue":     "#0080FF",
    "neon_teal":     "#00D4AA",
}

# ─── COLOR SCHEMES PER GAUGE ───────────────────────────────────────────────────
GAUGE_COLORS = {
    "orange": {
        "primary": QColor(255, 107, 53),
        "glow":    QColor(255, 107, 53, 60),
        "track":   QColor(255, 107, 53, 30),
        "text":    QColor(255, 140, 80),
    },
    "blue": {
        "primary": QColor(0, 191, 255),
        "glow":    QColor(0, 191, 255, 60),
        "track":   QColor(0, 191, 255, 30),
        "text":    QColor(100, 210, 255),
    },
    "purple": {
        "primary": QColor(168, 85, 247),
        "glow":    QColor(168, 85, 247, 60),
        "track":   QColor(168, 85, 247, 30),
        "text":    QColor(196, 130, 255),
    },
    "green": {
        "primary": QColor(0, 255, 136),
        "glow":    QColor(0, 255, 136, 60),
        "track":   QColor(0, 255, 136, 30),
        "text":    QColor(80, 255, 170),
    },
}

SLIDER_COLORS = {
    "density":   "#00BFFF",
    "roughness": "#A855F7",
    "hardness":  "#00FF88",
}


# ─── GAUGE WIDGET ──────────────────────────────────────────────────────────────
class GaugeWidget(QWidget):
    """Custom dark neon gauge widget"""
    valueChanged = pyqtSignal(float)

    def __init__(self, title, min_val, max_val, unit,
                 color_scheme="blue", decimals=1, parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.color_scheme = color_scheme
        self.decimals = decimals
        self._value = min_val
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = max(self.min_val, min(self.max_val, val))
        self.update()
        self.valueChanged.emit(self._value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        size = min(w, h)
        cx = w // 2
        cy = h // 2 + 12
        r  = size // 2 - 28

        colors = GAUGE_COLORS.get(self.color_scheme, GAUGE_COLORS["blue"])
        pct = (self._value - self.min_val) / max(self.max_val - self.min_val, 1e-9)

        # Background card
        card_path = QPainterPath()
        card_path.addRoundedRect(QRectF(2, 2, w - 4, h - 4), 16, 16)
        painter.fillPath(card_path, QBrush(QColor(DARK["bg_card"])))
        painter.setPen(QPen(QColor(DARK["border_glow"]), 1.2))
        painter.drawPath(card_path)

        # Outer glow halo
        halo = QRadialGradient(cx, cy, r + 20)
        cg = colors["glow"]
        halo.setColorAt(0.0,  QColor(cg.red(), cg.green(), cg.blue(), 0))
        halo.setColorAt(0.72, QColor(cg.red(), cg.green(), cg.blue(), 0))
        halo.setColorAt(0.87, QColor(cg.red(), cg.green(), cg.blue(), 26))
        halo.setColorAt(1.0,  QColor(cg.red(), cg.green(), cg.blue(), 0))
        painter.setBrush(QBrush(halo))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - r - 20, cy - r - 20, (r + 20) * 2, (r + 20) * 2)

        # Track arc
        pen_track = QPen(colors["track"], 10)
        pen_track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_track)
        painter.drawArc(cx - r, cy - r, r * 2, r * 2, 225 * 16, -270 * 16)

        # Active arc
        if pct > 0:
            pen_arc = QPen(colors["primary"], 10)
            pen_arc.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_arc)
            span = int(-270 * 16 * pct)
            painter.drawArc(cx - r, cy - r, r * 2, r * 2, 225 * 16, span)
            pen_shine = QPen(QColor(255, 255, 255, 50), 2)
            pen_shine.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_shine)
            painter.drawArc(cx - r, cy - r, r * 2, r * 2, 225 * 16, span)

        # Centre circle
        inner_r = 36
        radgrad = QRadialGradient(cx, cy, inner_r)
        radgrad.setColorAt(0.0, QColor(DARK["bg_elevated"]))
        radgrad.setColorAt(1.0, QColor(DARK["bg_card"]))
        painter.setBrush(QBrush(radgrad))
        painter.setPen(QPen(colors["primary"], 1.5))
        painter.drawEllipse(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2)

        # Value text
        painter.setPen(QPen(colors["text"]))
        font_val = QFont("Consolas", 14, QFont.Weight.Bold)
        painter.setFont(font_val)
        val_str  = f"{self._value:.{self.decimals}f}"
        val_rect = painter.fontMetrics().boundingRect(val_str)
        painter.drawText(cx - val_rect.width() // 2, cy + 5, val_str)

        # Unit text
        font_unit = QFont("Consolas", 7)
        painter.setFont(font_unit)
        painter.setPen(QPen(QColor(DARK["text_muted"])))
        unit_rect = painter.fontMetrics().boundingRect(self.unit)
        painter.drawText(cx - unit_rect.width() // 2, cy + 18, self.unit)

        # Title text
        font_title = QFont("Segoe UI", 14, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QPen(QColor(DARK["text_secondary"])))
        title_rect = painter.fontMetrics().boundingRect(self.title)
        painter.drawText(cx - title_rect.width() // 2, 20, self.title)

        # Min / max labels
        font_mm = QFont("Consolas", 7)
        painter.setFont(font_mm)
        painter.setPen(QPen(QColor(DARK["text_muted"])))
        if self.max_val < 1:
            min_t = f"{self.min_val:.2f}"
            max_t = f"{self.max_val:.2f}"
        else:
            min_t = str(int(self.min_val))
            max_t = str(int(self.max_val))
        painter.drawText(cx - r - 2, cy + 8, min_t)
        painter.drawText(cx + r - 22, cy + 8, max_t)


# ─── SLIDER GROUP ──────────────────────────────────────────────────────────────
class SliderGroup(QWidget):
    """Dark-themed combined slider + input field"""
    valueChanged = pyqtSignal(float)

    def __init__(self, title, min_val, max_val, unit, color, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.unit    = unit
        self.color   = color

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {DARK['text_primary']}; letter-spacing: 0.5px;")
        layout.addWidget(title_lbl)

        row = QHBoxLayout()
        row.setSpacing(12)

        min_lbl = QLabel(f"{min_val:.0f}")
        min_lbl.setFont(QFont("Consolas", 10))
        min_lbl.setStyleSheet(f"color: {DARK['text_muted']}; font-weight: bold;")
        row.addWidget(min_lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setFixedHeight(28)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {DARK['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5,
                    stop:0 white, stop:0.45 white,
                    stop:0.55 {color}, stop:1 {color});
                border: 2px solid {color};
                width: 20px;
                height: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {color};
                border: 3px solid white;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {color}88, stop:1 {color});
                border-radius: 3px;
            }}
        """)
        row.addWidget(self.slider)

        max_lbl = QLabel(f"{max_val:.0f}")
        max_lbl.setFont(QFont("Consolas", 10))
        max_lbl.setStyleSheet(f"color: {DARK['text_muted']}; font-weight: bold;")
        row.addWidget(max_lbl)

        row.addSpacing(12)

        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_field.setFixedSize(96, 38)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid {color}88;
                border-radius: 8px;
                padding: 4px 8px;
                background-color: {DARK['bg_deep']};
                color: {color};
                selection-background-color: {color}55;
            }}
            QLineEdit:focus {{
                border: 2px solid {color};
                background-color: {DARK['bg_elevated']};
            }}
        """)
        row.addWidget(self.input_field)

        unit_lbl = QLabel(unit)
        unit_lbl.setFont(QFont("Segoe UI", 11))
        unit_lbl.setStyleSheet(f"color: {DARK['text_secondary']};")
        row.addWidget(unit_lbl)

        layout.addLayout(row)

        self.slider.valueChanged.connect(self._slider_changed)
        self.input_field.editingFinished.connect(self._input_changed)
        self.set_value((min_val + max_val) / 2)

    def _slider_changed(self, value):
        real_value = self.min_val + (value / 1000.0) * (self.max_val - self.min_val)
        self.input_field.setText(f"{real_value:.1f}")
        self.valueChanged.emit(real_value)

    def _input_changed(self):
        try:
            value = float(self.input_field.text())
            value = max(self.min_val, min(self.max_val, value))
            self.set_value(value)
            self.valueChanged.emit(value)
        except ValueError:
            pass

    def set_value(self, value):
        value = max(self.min_val, min(self.max_val, value))
        self.input_field.setText(f"{value:.1f}")
        slider_val = int(((value - self.min_val) / (self.max_val - self.min_val)) * 1000)
        self.slider.setValue(slider_val)

    def get_value(self):
        try:
            return float(self.input_field.text())
        except ValueError:
            return (self.min_val + self.max_val) / 2


# ─── HELPERS ───────────────────────────────────────────────────────────────────
class SectionCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            SectionCard {{
                background-color: {DARK['bg_card']};
                border: 1px solid {DARK['border_glow']};
                border-radius: 14px;
            }}
        """)


class DividerLine(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(1)
        self.setStyleSheet(f"background: {DARK['border_glow']}; border: none;")


# ─── MODEL ─────────────────────────────────────────────────────────────────────
class ReverseANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(64, 32),  nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.network(x)


# ─── MAIN WINDOW ───────────────────────────────────────────────────────────────
class ModernProcessPredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Parameter Predictor - AlSi10Mg SLM")
        self.showMaximized()
        self.model_loaded = False
        self.load_model()
        self.init_ui()

    def load_model(self):
        try:
            norm_path  = os.path.join(PROJECT_ROOT, "models", "normalization_params.npz")
            model_path = os.path.join(PROJECT_ROOT, "models", "reverse_ann_complete.pth")
            if not os.path.exists(norm_path) or not os.path.exists(model_path):
                print("Model files not found")
                return
            norm_params  = np.load(norm_path)
            self.y_mean  = norm_params['y_mean']
            self.y_scale = norm_params['y_scale']
            self.X_mean  = norm_params['X_mean']
            self.X_scale = norm_params['X_scale']
            self.model   = ReverseANN()
            checkpoint   = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── TOP BAR ───────────────────────────────────────────────────────────
        topbar = QWidget()
        topbar.setFixedHeight(62)
        topbar.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {DARK['bg_deep']}, stop:0.5 #0D1527, stop:1 {DARK['bg_deep']});
            border-bottom: 1px solid {DARK['border_glow']};
        """)
        tb = QHBoxLayout(topbar)
        tb.setContentsMargins(32, 0, 32, 0)

        brand = QLabel("⬡  AlSi10Mg  SLM")
        brand.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        brand.setStyleSheet(f"""
            color: {DARK['accent_cyan']};
            background: {DARK['accent_cyan']}18;
            border: 1px solid {DARK['accent_cyan']}55;
            border-radius: 6px;
            padding: 4px 14px;
        """)
        tb.addWidget(brand)
        tb.addSpacing(24)

        title_lbl = QLabel("Process Parameter Predictor")
        title_lbl.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title_lbl.setStyleSheet(f"color: {DARK['text_primary']}; letter-spacing: 1px;")
        tb.addWidget(title_lbl)
        tb.addStretch()

        root.addWidget(topbar)

        # ── MAIN CONTENT ──────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet(f"background: {DARK['bg_deep']};")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 24)
        content_layout.setSpacing(24)

        # ════════════════ LEFT PANEL ═════════════════════════════════════════
        left_panel = QWidget()
        left_panel.setStyleSheet("background: transparent;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(0, 0, 0, 0)

        lh = QLabel("Target Mechanical Properties")
        lh.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lh.setStyleSheet(f"color: {DARK['accent_blue']}; letter-spacing: 0.8px; padding-bottom: 2px;")
        left_layout.addWidget(lh)

        left_layout.addWidget(DividerLine())

        self.sliders = {}
        slider_defs = [
            ("density",   "Relative Density",       95.0,  100.0, "%",  SLIDER_COLORS["density"],    95.0),
            ("roughness", "Surface Roughness (Sa)",   5.0,   30.0, "μm", SLIDER_COLORS["roughness"],   5.0),
            ("hardness",  "Hardness",               100.0,  150.0, "HV", SLIDER_COLORS["hardness"],  100.0),
        ]
        for key, title, lo, hi, unit, color, default in slider_defs:
            card = SectionCard()
            cl = QVBoxLayout(card)
            cl.setContentsMargins(22, 16, 22, 16)
            self.sliders[key] = SliderGroup(title, lo, hi, unit, color)
            self.sliders[key].set_value(default)
            cl.addWidget(self.sliders[key])
            left_layout.addWidget(card)

        left_layout.addSpacing(14)

        reset_btn = QPushButton("Reset")
        reset_btn.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        reset_btn.setFixedHeight(52)
        reset_btn.setFixedWidth(150)
        reset_btn.setParent(reset_btn.parent()) 
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #44444422, stop:1 #44444411);
                color: #AAAAAA;
                border: 1.5px solid #88888888;
                border-radius: 10px;
                padding: 0px 20px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: #44444433;
                border: 1.5px solid #888888;
            }}
            QPushButton:pressed {{
                background: #44444455;
            }}
        """)
        reset_btn.clicked.connect(self.reset)
        left_layout.addWidget(reset_btn)

        left_layout.addStretch()

        content_layout.addWidget(left_panel, 1)

        # ════════════════ RIGHT PANEL ════════════════════════════════════════
        right_panel = QWidget()
        right_panel.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        right_layout.setContentsMargins(0, 0, 0, 0)

        rh = QLabel("Predicted Process Parameters")
        rh.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        rh.setStyleSheet(f"color: {DARK['accent_green']}; letter-spacing: 0.8px; padding-bottom: 2px;")
        right_layout.addWidget(rh)

        right_layout.addWidget(DividerLine())

        # ── Gauge grid ────────────────────────────────────────────────────────
        # The card itself expands; equal stretch on rows + cols prevents overlap
        gauges_card = SectionCard()
        gauges_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        gauges_grid = QGridLayout(gauges_card)
        gauges_grid.setSpacing(10)
        gauges_grid.setContentsMargins(12, 12, 12, 12)
        gauges_grid.setRowStretch(0, 1)
        gauges_grid.setRowStretch(1, 1)
        gauges_grid.setColumnStretch(0, 1)
        gauges_grid.setColumnStretch(1, 1)

        self.gauge_power = GaugeWidget("Laser Power",     0, 1000, "W",    "orange", decimals=1)
        self.gauge_speed = GaugeWidget("Scan Speed",      0, 3000, "mm/s", "blue",   decimals=1)
        self.gauge_hatch = GaugeWidget("Hatch Distance",  0, 0.2,  "mm",   "purple", decimals=2)
        self.gauge_layer = GaugeWidget("Layer Thickness", 0, 0.1,  "mm",   "green",  decimals=2)

        gauges_grid.addWidget(self.gauge_power, 0, 0)
        gauges_grid.addWidget(self.gauge_speed, 0, 1)
        gauges_grid.addWidget(self.gauge_hatch, 1, 0)
        gauges_grid.addWidget(self.gauge_layer, 1, 1)

        right_layout.addWidget(gauges_card, stretch=3)

        # ── Energy Density card ───────────────────────────────────────────────
        energy_card = SectionCard()
        energy_main = QVBoxLayout(energy_card)
        energy_main.setContentsMargins(28, 16, 28, 16)
        energy_main.setSpacing(6)

        ed_header = QLabel("Laser Energy Density")
        ed_header.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        ed_header.setStyleSheet(f"color: {DARK['text_secondary']}; letter-spacing: 0.6px;")
        ed_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        energy_main.addWidget(ed_header)

        val_row = QHBoxLayout()
        val_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_row.setSpacing(14)

        self.energy_label = QLabel("--")
        self.energy_label.setFont(QFont("Consolas", 52, QFont.Weight.Bold))
        self.energy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_label.setStyleSheet(f"color: {DARK['accent_blue']};")
        val_row.addWidget(self.energy_label)

        unit_col = QVBoxLayout()
        unit_col.setAlignment(Qt.AlignmentFlag.AlignBottom)
        eu = QLabel("J / mm³")
        eu.setFont(QFont("Segoe UI", 16))
        eu.setStyleSheet(f"color: {DARK['text_secondary']};")
        unit_col.addWidget(eu)
        val_row.addLayout(unit_col)

        energy_main.addLayout(val_row)

        badge_row = QHBoxLayout()
        badge_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_status = QLabel("Waiting for prediction...")
        self.energy_status.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.energy_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_status.setStyleSheet(f"""
            color: {DARK['text_muted']};
            background: {DARK['bg_elevated']};
            border: 1px solid {DARK['border']};
            border-radius: 20px;
            padding: 7px 24px;
        """)
        badge_row.addWidget(self.energy_status)
        energy_main.addLayout(badge_row)

        right_layout.addWidget(energy_card, stretch=1)

        content_layout.addWidget(right_panel, 2)
        root.addWidget(content)

        # ── FOOTER ────────────────────────────────────────────────────────────
        footer = QWidget()
        footer.setFixedHeight(32)
        footer.setStyleSheet(f"""
            background: {DARK['bg_panel']};
            border-top: 1px solid {DARK['border']};
        """)
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(32, 0, 32, 0)

        fl_left = QLabel("AlSi10Mg Optimization Framework")
        fl_left.setFont(QFont("Consolas", 9))
        fl_left.setStyleSheet(f"color: {DARK['text_muted']};")
        fl.addWidget(fl_left)

        root.addWidget(footer)

        # ── Connect & fire initial prediction ─────────────────────────────────
        self.sliders['density'].valueChanged.connect(self.predict)
        self.sliders['roughness'].valueChanged.connect(self.predict)
        self.sliders['hardness'].valueChanged.connect(self.predict)
        self.predict()

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self):
        if not self.model_loaded:
            return
        try:
            density   = self.sliders['density'].get_value()
            roughness = self.sliders['roughness'].get_value()
            hardness  = self.sliders['hardness'].get_value()

            target        = np.array([[density, roughness, hardness]])
            target_scaled = (target - self.y_mean) / self.y_scale
            target_tensor = torch.FloatTensor(target_scaled)

            with torch.no_grad():
                pred_scaled = self.model(target_tensor).numpy()

            pred = pred_scaled * self.X_scale + self.X_mean

            self.gauge_power.value = pred[0, 0]
            self.gauge_speed.value = pred[0, 1]
            self.gauge_hatch.value = pred[0, 2]
            self.gauge_layer.value = pred[0, 3]

            energy = pred[0, 0] / (pred[0, 1] * pred[0, 2] * pred[0, 3])
            self.energy_label.setText(f"{energy:.2f}")

            if 50 <= energy <= 100:
                val_color  = DARK["accent_green"]
                status_txt = "●  OPTIMAL — Process Stable"
                bg         = DARK["accent_green"]
            elif 20 <= energy < 50:
                val_color  = DARK["accent_orange"]
                status_txt = "▲  LOW — Risk of Lack of Fusion"
                bg         = DARK["accent_orange"]
            elif 100 < energy <= 200:
                val_color  = DARK["accent_orange"]
                status_txt = "▲  HIGH — Risk of Keyholing"
                bg         = DARK["accent_orange"]
            else:
                val_color  = DARK["accent_red"]
                status_txt = "✕  CRITICAL — Process Unstable"
                bg         = DARK["accent_red"]

            self.energy_label.setStyleSheet(f"color: {val_color};")
            self.energy_status.setText(status_txt)
            self.energy_status.setStyleSheet(f"""
                color: {bg};
                background: {bg}22;
                border: 1.5px solid {bg}88;
                border-radius: 20px;
                padding: 7px 24px;
                font-family: Consolas;
                font-size: 12pt;
                font-weight: bold;
            """)
        except Exception as e:
            print(f"Prediction error: {e}")

    def reset(self):
        self.sliders['density'].set_value(95.0)
        self.sliders['roughness'].set_value(5.0)
        self.sliders['hardness'].set_value(100.0)


# ─── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(DARK["bg_deep"]))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(DARK["text_primary"]))
    palette.setColor(QPalette.ColorRole.Base,            QColor(DARK["bg_panel"]))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(DARK["bg_card"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(DARK["bg_elevated"]))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(DARK["text_primary"]))
    palette.setColor(QPalette.ColorRole.Text,            QColor(DARK["text_primary"]))
    palette.setColor(QPalette.ColorRole.Button,          QColor(DARK["bg_elevated"]))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(DARK["text_primary"]))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(DARK["accent_cyan"]))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(DARK["neon_blue"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)

    app.setFont(QFont("Segoe UI", 10))

    window = ModernProcessPredictor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()