# Data Cleaning Dashboard

A simple and powerful web application for cleaning, analyzing, and visualizing your data.

## ✨ Features

- **Upload CSV/Excel files** - Simple drag and drop interface
- **Fix missing values** - Using mean, median, mode or custom values
- **Remove duplicates** - Clean your data with one click
- **Detect outliers** - Identify and handle unusual values
- **Create visualizations** - Charts, graphs, and plots
- **One-click setup** - Easy installation for everyone

## 🚀 Quick Start

### For Windows Users:
1. Download all files
2. **Double-click** `setup.bat`
3. **Double-click** `run.bat`
4. The app will open in your browser

### For Mac/Linux Users:
```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Start the application
streamlit run app.py
```

## 📦 What's Included

- `app.py` - Main application
- `requirements.txt` - Python dependencies
- `setup.bat` - Windows installer
- `setup.sh` - Mac/Linux installer
- `run.bat` - Windows launcher

## 🔧 Manual Installation

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate.bat

# Activate environment (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 How to Use

1. **Upload your data** using the file uploader
2. **Explore your data** with the overview dashboard
3. **Clean your data** using the various options
4. **Visualize results** with interactive charts
5. **Download cleaned data** as CSV


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/your-username/data-cleaning-dashboard.git

cd data-cleaning-dashboard
