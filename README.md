# 🚗 Smart Parking Dashboard

Interactive Streamlit application predicting parking occupancy and analyzing real-time parking zone patterns.

---

## 📋 Requirements

- **Python**: `>= 3.10, <= 3.14` (Developed on Python `3.14.5`)
- Dependencies listed in `requirements.txt`.

---

## 🚀 Quick Start (For Local Execution)

### Step 1: Install Python (If not already installed)

Download and install Python 3.10+ from [python.org/downloads](https://www.python.org/downloads/).  
_(On Windows, ensure you check **"Add Python to PATH"** during installation)._

### Step 2: Set Up Virtual Environment & Dependencies

Open your Terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) in this project folder:

```bash
# 1. Create a local virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install required libraries
pip install -r requirements.txt
```

### Step 3: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📦 What's Inside This Package

- `app.py`: Main Streamlit dashboard application.
- `best_parking_model_pipeline_hacked.joblib`: Pre-trained Random Forest AI model pipeline.
- `smart_parking_usage_occupancy_analytics.csv`: Historical occupancy dataset.
- `requirements.txt`: Exact dependency specifications.
