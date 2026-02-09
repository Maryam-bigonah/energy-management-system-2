# Legacy Code Archive

This directory contains the original hard-coded device and household definitions from the initial implementation.

**These files are NO LONGER USED by the generator.**

The generator now uses YAML configuration files in `../config/`:
- `../config/devices.yaml` - All device definitions
- `../config/households.yaml` - All household profiles

## Files in this directory

- `devices.py` - Original hard-coded device database (replaced by devices.yaml)
- `household_types.py` - Original hard-coded household types (replaced by households.yaml)

## Why keep these?

- **Reference**: Useful for comparing old vs new definitions
- **Documentation**: Shows the original implementation approach
- **Recovery**: In case specific details need to be cross-referenced

## Do not import these files

The current generator (`../generator.py`) does NOT import these files and should not.

If you need to modify device or household definitions, edit the YAML files in `../config/` instead.
