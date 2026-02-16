"""
AlSi10Mg SLM Process Parameter Predictor
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QMessageBox, QGroupBox, QGridLayout, QSlider, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

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

# Get the folder where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class GaugeWidget(QWidget):
    """Custom gauge widget for displaying values"""
    valueChanged = pyqtSignal(float)
    
    def __init__(self, title, min_val, max_val, unit, color_scheme="blue", decimals=1, parent=None):
        super().__init__(parent)
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.color_scheme = color_scheme
        self.decimals = decimals
        self._value = min_val
        self.setMinimumSize(180, 180)
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
        
        width = self.width()
        height = self.height()
        size = min(width, height) - 20
        center_x = width // 2
        center_y = height // 2 + 10
        radius = size // 2 - 20
        
        # Draw background arc
        pen_bg = QPen(QColor(220, 220, 220), 12)
        pen_bg.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(center_x - radius, center_y - radius, 
                       radius * 2, radius * 2, 225 * 16, -270 * 16)
        
        # Color schemes
        colors = {
            "blue": (QColor(52, 152, 219), QColor(41, 128, 185)),
            "green": (QColor(46, 204, 113), QColor(39, 174, 96)),
            "orange": (QColor(230, 126, 34), QColor(211, 84, 0)),
            "purple": (QColor(155, 89, 182), QColor(142, 68, 173)),
            "red": (QColor(231, 76, 60), QColor(192, 57, 43))
        }
        
        color1, color2 = colors.get(self.color_scheme, colors["blue"])
        
        # Calculate angle for current value
        percentage = (self._value - self.min_val) / (self.max_val - self.min_val)
        
        # Draw value arc
        gradient_pen = QPen(color1, 12)
        gradient_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(gradient_pen)
        
        span = int(-270 * 16 * percentage)
        painter.drawArc(center_x - radius, center_y - radius,
                       radius * 2, radius * 2, 225 * 16, span)
        
        # Draw center circle
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(color2, 2))
        painter.drawEllipse(center_x - 35, center_y - 35, 70, 70)
        
        # Draw value text with variable decimals
        painter.setPen(QPen(QColor(44, 62, 80)))
        font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        painter.setFont(font)
        text = f"{self._value:.{self.decimals}f}"
        text_rect = painter.boundingRect(0, 0, 0, 0, Qt.AlignmentFlag.AlignCenter, text)
        painter.drawText(center_x - text_rect.width()//2, center_y - 5, text)
        
        # Draw unit
        font_unit = QFont("Segoe UI", 8)
        painter.setFont(font_unit)
        painter.setPen(QPen(QColor(127, 140, 141)))
        painter.drawText(center_x - 15, center_y + 12, self.unit)
        
        # Draw title
        font_title = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font_title)
        painter.setPen(QPen(QColor(52, 73, 94)))
        title_rect = painter.boundingRect(0, 0, 0, 0, Qt.AlignmentFlag.AlignCenter, self.title)
        painter.drawText(center_x - title_rect.width()//2, 20, self.title)
        
        # Draw min/max labels
        font_small = QFont("Segoe UI", 7)
        painter.setFont(font_small)
        painter.setPen(QPen(QColor(149, 165, 166)))
        
        # Format min/max labels based on value range
        if self.max_val < 1:
            min_text = f"{self.min_val:.2f}"
            max_text = f"{self.max_val:.2f}"
        else:
            min_text = str(int(self.min_val))
            max_text = str(int(self.max_val))
        
        painter.drawText(center_x - radius - 5, center_y + 5, min_text)
        painter.drawText(center_x + radius - 20, center_y + 5, max_text)

class SliderGroup(QWidget):
    """Combined slider and input field"""
    valueChanged = pyqtSignal(float)
    
    def __init__(self, title, min_val, max_val, unit, color, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self.color = color
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Title row
        title_layout = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # Input and slider row
        input_slider_layout = QHBoxLayout()
        
        # Min label at the start
        self.min_label = QLabel(f"{min_val:.0f}")
        self.min_label.setFont(QFont("Segoe UI", 10))
        self.min_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
        input_slider_layout.addWidget(self.min_label)
        
        input_slider_layout.addSpacing(10)
        
        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid #bdc3c7;
                height: 6px;
                background: #ecf0f1;
                margin: 0px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {color};
                border: 2px solid white;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border-radius: 3px;
            }}
        """)
        input_slider_layout.addWidget(self.slider)
        
        input_slider_layout.addSpacing(10)
        
        # Max label at the end
        self.max_label = QLabel(f"{max_val:.0f}")
        self.max_label.setFont(QFont("Segoe UI", 10))
        self.max_label.setStyleSheet("color: #7f8c8d; font-weight: bold;")
        input_slider_layout.addWidget(self.max_label)
        
        input_slider_layout.addSpacing(20)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setFont(QFont("Segoe UI", 12))
        self.input_field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_field.setFixedWidth(90)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                border: 2px solid {color};
                border-radius: 6px;
                padding: 6px;
                background-color: white;
                color: #2c3e50;
            }}
            QLineEdit:focus {{
                border: 3px solid {color};
            }}
        """)
        input_slider_layout.addWidget(self.input_field)
        
        # Unit label
        unit_label = QLabel(unit)
        unit_label.setFont(QFont("Segoe UI", 10))
        input_slider_layout.addWidget(unit_label)
        
        layout.addLayout(input_slider_layout)
        
        # Connect signals
        self.slider.valueChanged.connect(self._slider_changed)
        self.input_field.editingFinished.connect(self._input_changed)
        
        # Set default value
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

class ReverseANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.network(x)

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
            # Look for model files in the same folder as this script
            norm_path = os.path.join(SCRIPT_DIR, "normalization_params.npz")
            model_path = os.path.join(SCRIPT_DIR, "reverse_ann_complete.pth")
            
            print(f"Looking for model files in: {SCRIPT_DIR}")
            
            if not os.path.exists(norm_path):
                print(f"ERROR: normalization_params.npz not found!")
                print(f"Expected location: {norm_path}")
                return
                
            if not os.path.exists(model_path):
                print(f"ERROR: reverse_ann_complete.pth not found!")
                print(f"Expected location: {model_path}")
                return
            
            norm_params = np.load(norm_path)
            self.y_mean = norm_params['y_mean']
            self.y_scale = norm_params['y_scale']
            self.X_mean = norm_params['X_mean']
            self.X_scale = norm_params['X_scale']
            
            self.model = ReverseANN()
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Left panel - Inputs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # Header
        header = QLabel("Target Mechanical Properties")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setStyleSheet("color: #2c3e50; padding: 15px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(header)
        
        # Status indicator
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if self.model_loaded:
            status_text = "Model Active"
            status_color = "#27ae60"
        else:
            status_text = "Model Error | Check Console"
            status_color = "#e74c3c"
        
        status_label = QLabel(status_text)
        status_label.setFont(QFont("Segoe UI", 11))
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        status_layout.addWidget(status_label)
        left_layout.addWidget(status_widget)
        
        # Input sliders
        self.sliders = {}
        
        # Relative Density (95-100)
        density_container = QGroupBox()
        density_layout = QVBoxLayout(density_container)
        self.sliders['density'] = SliderGroup(
            "Relative Density", 95.0, 100.0, "%", "#3498db"
        )
        self.sliders['density'].set_value(95.0)
        density_layout.addWidget(self.sliders['density'])
        left_layout.addWidget(density_container)
        
        # Surface Roughness (5-30)
        roughness_container = QGroupBox()
        roughness_layout = QVBoxLayout(roughness_container)
        self.sliders['roughness'] = SliderGroup(
            "Surface Roughness (Sa)", 5.0, 30.0, "um", "#9b59b6"
        )
        self.sliders['roughness'].set_value(5.0)
        roughness_layout.addWidget(self.sliders['roughness'])
        left_layout.addWidget(roughness_container)
        
        # Hardness (100-150)
        hardness_container = QGroupBox()
        hardness_layout = QVBoxLayout(hardness_container)
        self.sliders['hardness'] = SliderGroup(
            "Hardness", 100.0, 150.0, "HV", "#1abc9c"
        )
        self.sliders['hardness'].set_value(100.0)
        hardness_layout.addWidget(self.sliders['hardness'])
        left_layout.addWidget(hardness_container)
        
        left_layout.addStretch()
        
        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        reset_btn = QPushButton("Reset")
        reset_btn.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        reset_btn.setMinimumHeight(55)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border-radius: 10px;
                padding: 15px;
                border: none;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        reset_btn.clicked.connect(self.reset)
        btn_layout.addWidget(reset_btn)
        
        left_layout.addLayout(btn_layout)
        
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Outputs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        
        # Output header
        out_header = QLabel("Predicted Process Parameters")
        out_header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        out_header.setStyleSheet("color: #27ae60; padding: 15px;")
        out_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(out_header)
        
        # Gauges container
        gauges_widget = QWidget()
        gauges_layout = QGridLayout(gauges_widget)
        gauges_layout.setSpacing(20)
        
        # Laser Power Gauge
        self.gauge_power = GaugeWidget("Laser Power", 0, 1000, "W", "orange", decimals=1)
        gauges_layout.addWidget(self.gauge_power, 0, 0)
        
        # Scan Speed Gauge
        self.gauge_speed = GaugeWidget("Scan Speed", 0, 3000, "mm/s", "blue", decimals=1)
        gauges_layout.addWidget(self.gauge_speed, 0, 1)
        
        # Hatch Distance Gauge (2 decimals)
        self.gauge_hatch = GaugeWidget("Hatch Distance", 0, 0.2, "mm", "purple", decimals=2)
        gauges_layout.addWidget(self.gauge_hatch, 1, 0)
        
        # Layer Thickness Gauge (2 decimals)
        self.gauge_layer = GaugeWidget("Layer Thickness", 0, 0.1, "mm", "green", decimals=2)
        gauges_layout.addWidget(self.gauge_layer, 1, 1)
        
        right_layout.addWidget(gauges_widget)
        
        # Energy Density Display
        energy_frame = QGroupBox("Laser Energy Density")
        energy_frame.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        energy_layout = QVBoxLayout(energy_frame)
        
        self.energy_label = QLabel("--")
        self.energy_label.setFont(QFont("Segoe UI", 48, QFont.Weight.Bold))
        self.energy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_label.setStyleSheet("color: #2c3e50;")
        energy_layout.addWidget(self.energy_label)
        
        self.energy_unit = QLabel("J/mm^3")
        self.energy_unit.setFont(QFont("Segoe UI", 16))
        self.energy_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_unit.setStyleSheet("color: #7f8c8d;")
        energy_layout.addWidget(self.energy_unit)
        
        self.energy_status = QLabel("Waiting for prediction...")
        self.energy_status.setFont(QFont("Segoe UI", 12))
        self.energy_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.energy_status.setStyleSheet("color: #95a5a6; padding: 15px;")
        energy_layout.addWidget(self.energy_status)
        
        right_layout.addWidget(energy_frame)
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel, 2)
        
        # Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QWidget {
                background-color: #f0f2f5;
            }
            QGroupBox {
                background-color: white;
                border: 2px solid #e1e4e8;
                border-radius: 12px;
                padding: 15px;
            }
        """)
        
        # Connect sliders to live prediction
        self.sliders['density'].valueChanged.connect(self.predict)
        self.sliders['roughness'].valueChanged.connect(self.predict)
        self.sliders['hardness'].valueChanged.connect(self.predict)
        
        # Initial prediction
        self.predict()
        
    def predict(self):
        if not self.model_loaded:
            return
            
        try:
            density = self.sliders['density'].get_value()
            roughness = self.sliders['roughness'].get_value()
            hardness = self.sliders['hardness'].get_value()
            
            target = np.array([[density, roughness, hardness]])
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
                status = "OPTIMAL - Process stable"
                color = "#27ae60"
            elif 20 <= energy < 50:
                status = "LOW - Risk of lack of fusion"
                color = "#f39c12"
            elif 100 < energy <= 200:
                status = "HIGH - Risk of keyholing"
                color = "#f39c12"
            else:
                status = "CRITICAL - Process unstable"
                color = "#e74c3c"
            
            self.energy_status.setText(status)
            self.energy_status.setStyleSheet(f"color: {color}; padding: 15px; font-weight: bold;")
            
        except Exception as e:
            print(f"Prediction error: {e}")
            
    def reset(self):
        self.sliders['density'].set_value(95.0)
        self.sliders['roughness'].set_value(5.0)
        self.sliders['hardness'].set_value(100.0)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ModernProcessPredictor()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()