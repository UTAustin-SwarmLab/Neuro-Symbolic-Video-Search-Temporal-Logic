# Python-Project-Template

This is a template repository. Please initialize your python project using this template.

1. Make sure you have a right python version installed locally

2. `your_project_name` is your project package name including src.

3. Development
   ```
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip build
   python -m pip install --editable ."[dev, test]"
   ```
   
5. If you want to build a project
   ```
   python -m build
   ```

