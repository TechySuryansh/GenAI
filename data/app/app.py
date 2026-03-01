import os
import exec_file # Not a standard lib, using alternative

# This is a deployment bridge to satisfy Streamlit Cloud's configuration
# while maintaining a professional root-level project structure.

bridge_path = os.path.abspath(__file__)
root_dir = os.path.abspath(os.path.join(os.path.dirname(bridge_path), "../../"))
os.chdir(root_dir)

# Execute the main app.py
with open("app.py", "rb") as f:
    code = compile(f.read(), "app.py", "exec")
    exec(code, globals())
