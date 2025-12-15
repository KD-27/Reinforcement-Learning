
## Setting Up the Project

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

---

### 2. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

---

### 3. Install Required Packages

```bash
pip install numpy matplotlib seaborn
```

> These packages are required to run the environment, agent, and visualizations.

---

### 4. Git Ignore Recommendations

Before pushing to a repository, make sure to create a `.gitignore` file:

```
# Python virtual environment
venv/
__pycache__/
*.pyc

# Optional: common Python files
.env
*.pyo
```

> This prevents committing unnecessary files like virtual environments or compiled Python files.

---

### 4. Git Tag Recommendations
```
# Make sure you're on the right branch
git checkout main  # or master, depending on your repo

# Create a tag to mark the backup point
git tag -a backup-v1 -m "Backup before major update"

# Push the tag to remote
git push origin backup-v1
```