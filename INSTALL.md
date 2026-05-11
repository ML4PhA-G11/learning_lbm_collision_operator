# Installation instructions
For this project it is strongly recommended to use a venv in combination with uv.

## Step 1: Install uv
Mac/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Step 2: Create a venv and install dependencies
```bash
uv venv
uv sync
```

All done!

# Adding packages
To add a package, simply run:
```bash
uv add <package-name>
```
This will add the package to the pyproject.toml file and install it in the venv. 