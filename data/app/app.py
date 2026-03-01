import os

# Professional Deployment Bridge for Streamlit Cloud
# This script satisfies legacy path settings while running the clean root-level app.

# 1. Determine the project root absolute path
bridge_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(bridge_file), "../../"))

# 2. Change working directory to root
os.chdir(project_root)

# 3. Execute the root-level app.py with correct context
with open(os.path.join(project_root, "app.py"), "rb") as f:
    code = compile(f.read(), "app.py", "exec")
    # Explicitly set __file__ so app.py correctly determines BASE_DIR
    exec(code, {
        "__file__": os.path.join(project_root, "app.py"),
        "__name__": "__main__",
        **globals()
    })
