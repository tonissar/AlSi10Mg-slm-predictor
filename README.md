# AlSi10Mg SLM Process Parameter Predictor

A reverse-trained ANN GUI tool for predicting Selective Laser Melting process parameters for targeted mechanical properties (Relative density, Surface roughness and hardness) of AlSi10Mg alloy.

## Requirements

- **Python**: Version 3.8 or higher
- **Libraries**: PyTorch, NumPy, PyQt6
- **VS Code**

### Step 1: Install Python
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check **"Add Python to PATH"**

### Step 2: Install VS Code
Download from code.visualstudio.com
Install the Python extension (by Microsoft) from the Extensions marketplace

### Step 3: Install Required python Libraries
Open terminal in VS Code and run:
pip install torch numpy PyQt6

### Step 4: Run the console application
Option 1: VS Code
Open the folder in VS Code
Open Param_predictor.py
Click the ▶ Run button (top right) or press F5

Option 2: Terminal
python Param_predictor.py

#### Important: Keep all the files in the same folder location as Param_predictor.py

### Methodology
This tool uses a reverse-trained Artificial Neural Network that learns the inverse mapping from target mechanical properties (density, roughness, hardness) to optimal SLM process parameters. The 3-layer MLP is trained on data, with predictions validated against laser energy density thresholds to ensure process stability.