@echo off
echo Setting up Data Cleaning Dashboard...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo Setup complete! Run 'streamlit run app.py' to start the application.