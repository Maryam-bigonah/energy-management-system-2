import argparse
import os
import sys
import pandas as pd
import glob
from pathlib import Path
from datetime import timedelta

# Import pylpg packages
# We assume pylpg is installed in the environment
from pylpg import lpg_execution, lpgdata, lpgpythonbindings

def main():
    parser = argparse.ArgumentParser(description="Generate household electricity datasets using LoadProfileGenerator")
    parser.add_argument("--samples-per-type", type=int, default=10, help="Number of samples (random seeds) per household type")
    parser.add_argument("--resolution-min", type=int, default=15, help="Time resolution in minutes (default: 15)")
    parser.add_argument("--year", type=int, default=2023, help="Year to simulate")
    parser.add_argument("--outdir", type=str, default="/work/output", help="Output directory")
    parser.add_argument("--device-level", action="store_true", help="Enable device-level output")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # Define household types mapping
    # Mappings based on user request:
    # CHR01 working couple -> Households.CHR01_Couple_both_at_Work
    # CHR41 working couple + 3 children -> Households.CHR41_Family_with_3_children_both_at_work
    # CHR45 one working, one home + 1 child -> Households.CHR45_Family_with_1_child_1_at_work_1_at_home
    # CHR54 retired couple -> Households.CHR54_Retired_Couple_no_work
    
    households = {
        "CHR01": lpgdata.Households.CHR01_Couple_both_at_Work,
        "CHR41": lpgdata.Households.CHR41_Family_with_3_children_both_at_work,
        "CHR45": lpgdata.Households.CHR45_Family_with_1_child_1_at_work_1_at_home,
        "CHR54": lpgdata.Households.CHR54_Retired_Couple_no_work
    }
    
    # We use a standard house type that doesn't introduce massive heating loads unless requested,
    # but the user asked for "total household electricity". 
    # HT20 is "Single Family House (no heating/cooling)" which is good for appliance load focus.
    house_type = lpgdata.HouseTypes.HT20_Single_Family_House_no_heating_cooling
    
    # Metadata list
    metadata = []
    
    # Determine where pylpg is installed to find the results directory
    # pylpg.lpg_execution is where the C1 directory will be created relative to.
    pylpg_dir = Path(lpg_execution.__file__).parent.absolute()
    results_dir = pylpg_dir / "C1" / "results" / "Results"
    
    # Calculate resolution string for LPG
    res_timedelta = timedelta(minutes=args.resolution_min)
    total_seconds = int(res_timedelta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    resolution_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    print(f"Starting generation with resolution {resolution_str} ({args.resolution_min} min)")
    
    for hh_name, hh_ref in households.items():
        print(f"\nProcessing Household Type: {hh_name} ({hh_ref.Name})")
        
        type_outdir = os.path.join(args.outdir, hh_name)
        os.makedirs(type_outdir, exist_ok=True)
        
        for i in range(args.samples_per_type):
            seed = i + 1  # 1-based seed
            run_id = f"{hh_name}_{seed}"
            print(f"  Running seed {seed}/{args.samples_per_type}...")
            
            calc_options = []
            if args.device_level:
                # Enable JSON device profiles
                # We use both options to be safe on what LPG generates
                calc_options.append(lpgpythonbindings.CalcOption.JsonDeviceProfilesIndividualHouseholds)
                calc_options.append(lpgpythonbindings.CalcOption.DeviceProfilesIndividualHouseholds)
            
            try:
                # Run simulation
                # execute_lpg_single_household uses C1 directory by default.
                data = lpg_execution.execute_lpg_single_household(
                    year=args.year,
                    householdref=hh_ref,
                    housetype=house_type,
                    resolution=resolution_str,
                    random_seed=seed,
                    calc_options=calc_options if calc_options else None
                )
                
                # Identify total load column
                elec_cols = [c for c in data.columns if "Electricity" in c and "Car" not in c and "Heating" not in c]
                if not elec_cols:
                    # Generic fallback
                    elec_cols = [c for c in data.columns if "Electricity" in c]
                
                # Create dataset with metadata
                dataset = data.copy()
                dataset['household_type'] = hh_name
                dataset['seed'] = seed
                
                dataset.reset_index(inplace=True)
                dataset.rename(columns={'index': 'timestamp'}, inplace=True)
                
                # Save total load CSV
                filename_total = f"total_seed{seed}.csv"
                filepath_total = os.path.join(type_outdir, filename_total)
                dataset.to_csv(filepath_total, index=False)
                
                file_devices = "N/A"
                
                # Handle device profiles if requested
                if args.device_level:
                    # Look for device profile JSONs in results_dir
                    # Pattern: DeviceProfiles.*.json or similar
                    device_files = list(results_dir.glob("DeviceProfiles.*.json"))
                    if device_files:
                        # We can either merge them or just move them to output
                        # Moving them is safer to avoid huge memory usage in DataFrame
                        device_out_dir = os.path.join(type_outdir, f"devices_seed{seed}")
                        os.makedirs(device_out_dir, exist_ok=True)
                        
                        count = 0
                        for df in device_files:
                            # Parse to CSV or keep as JSON? 
                            # User asked for "device-level electricity profiles". CSV is preferred.
                            # We can try to parse using lpg_execution.parse_json_profile
                            try:
                                profile = lpg_execution.LPGExecutor.parse_json_profile(str(df))
                                if profile and profile.Values:
                                    # Create Series
                                    dev_name = profile.LoadTypeName 
                                    if hasattr(profile, 'Header') and profile.Header:
                                         dev_name = f"{profile.Header.DeviceName}_{profile.LoadTypeName}"
                                    
                                    # Sanitize filename
                                    safe_name = "".join(x for x in dev_name if x.isalnum() or x in "_- ")
                                    dev_csv_path = os.path.join(device_out_dir, f"{safe_name}.csv")
                                    
                                    # Make DataFrame
                                    dev_df = pd.DataFrame({
                                        'timestamp': dataset['timestamp'],
                                        'power': profile.Values
                                    })
                                    dev_df.to_csv(dev_csv_path, index=False)
                                    count += 1
                            except Exception as e:
                                print(f"    Failed to parse device file {df.name}: {e}")
                        
                        file_devices = f"{count} files in {os.path.basename(device_out_dir)}"
                    
                metadata.append({
                    "run_id": run_id,
                    "household_type": hh_name,
                    "seed": seed,
                    "resolution_min": args.resolution_min,
                    "start": dataset['timestamp'].min(),
                    "end": dataset['timestamp'].max(),
                    "file_total": os.path.join(hh_name, filename_total),
                    "file_devices": file_devices
                })
                
            except Exception as e:
                print(f"    Error running {run_id}: {e}")
                # traceback.print_exc() # reducing clutter


    # Write metadata
    if metadata:
        meta_df = pd.DataFrame(metadata)
        meta_df.to_csv(os.path.join(args.outdir, "metadata.csv"), index=False)
        print(f"\nGeneration complete. Metadata saved to {os.path.join(args.outdir, 'metadata.csv')}")
    else:
        print("\nNo data generated.")

if __name__ == "__main__":
    main()
