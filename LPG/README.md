# LoadProfileGenerator Docker Pipeline

This project dockerizes the [LoadProfileGenerator (LPG)](https://www.loadprofilegenerator.de/) to generate synthetic household electricity datasets on a macOS host (running Linux container).

## Requirements

- Docker Desktop (macOS)
- `load` directory containing:
    - `LPG10.10.0_Linux/` (LPG binaries)
    - `pylpg-main/` (Python bindings source)

## Usage

The `run.sh` script builds the Docker image and runs the generation pipeline.

```bash
./run.sh [arguments]
```

### Common Arguments

- `--samples-per-type N`: Number of random seed samples per household type (default: 10)
- `--resolution-min N`: Time resolution in minutes (default: 15)
- `--device-level`: Enable export of individual device profiles (CSV)
- `--year YYYY`: Year to simulate

### Examples

**Standard run (10 samples, 15-min resolution):**
```bash
./run.sh --samples-per-type 10
```

**High resolution (1-min) with device breakdown:**
```bash
./run.sh --samples-per-type 1 --resolution-min 1 --device-level
```

## Output

Outputs are saved to `./output/` on the host.

- `metadata.csv`: Summary of all runs (filenames, seeds, resolution).
- `/CHRxx/`: Subdirectories for each household archetype.
    - `total_seedN.csv`: Total household electricity load.
    - `devices_seedN/`: (Optional) Individual device load profiles.

## Household Archetypes

- **CHR01**: Couple, both at work.
- **CHR41**: Family with 3 children, both parents at work.
- **CHR45**: Family with 1 child, one parent at work, one at home.
- **CHR54**: Retired couple, no work.

## Moving to Windows

This project is cross-platform and runs natively on Windows.

### Steps to Run on Windows

1.  **Copy Files**: Copy this entire folder (`load`) to your Windows machine.
2.  **Install Requirements**:
    -   [Python 3.10+](https://www.python.org/downloads/windows/)
    -   [.NET 8.0 Runtime](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
3.  **Run**:
    Open PowerShell or Command Prompt in the folder:

    ```powershell
    pip install -r requirements.txt
    python generate_dataset.py
    ```

    *The script will automatically download the correct Windows binaries (`simengine2.exe`) on the first run.*

### Docker on Windows

Alternatively, you can run the Docker container using Docker Desktop (WSL2 backend):

```powershell
docker build -t lpggen .
docker run --rm -v ${PWD}/output:/work/output lpggen python3 generate_dataset.py
```
