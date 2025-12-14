### Create a virtual environment named venv
python -m venv venv

### Activate the environment to run your files
#### On Windows
venv\Scripts\activate

#### On Linux 
Source venv/bin/activate

### Install required packages and libraries using pip
pip install numpy
pip install matplotlib
pip install seaborn

### Make sure to create a .gitignore before pushing it to any repo
    # Python virtual environment
    venv/
    __pycache__/
    *.pyc

    # Optional: common Python stuff
    .env
    *.pyo