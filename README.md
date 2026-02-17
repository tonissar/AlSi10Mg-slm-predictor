# AlSi10Mg SLM Process Parameter Predictor

A reverse-trained ANN GUI tool for predicting Selective Laser Melting process parameters for targeted mechanical properties (Relative density, Surface roughness and hardness) of AlSi10Mg alloy.

## Requirements
  Download all files required to run the console from the folder /GUI_files
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
Open terminal in VS Code and run command:
### pip install torch numpy PyQt6

### Step 4: Run the console application Param_predictor.py
### Option 1: VS Code
Open the folder of downloaded GUI console files in VS Code:
Open python file Param_predictor.py.
Click the ▶ Run button (top right) or press F5

### Option 2: in the Terminal run the below command
python Param_predictor.py

#### Important: Keep all the files in the same folder location as Param_predictor.py

#### How to Use
Adjust Target Properties using the sliders (left panel):
Relative Density: 95-100%
Surface Roughness: 5-30 μm
Hardness: 100-150 HV
View Predictions (right panel):Laser Power, Scan Speed, Hatch Distance, Layer Thickness
laser energy density indicates the process stability.

### Methodology
This tool uses a reverse-trained Artificial Neural Network that learns the inverse mapping from target mechanical properties (density, roughness, hardness) to optimal SLM process parameters. The 3-layer MLP is trained on data, with predictions validated against laser energy density thresholds to ensure process stability.