"""
fix_city_names.py - Fix directory and file names with trailing characters
"""
from pathlib import Path


# Mapping of incorrect to correct city names
CORRECTIONS = {
    'New Yorkd': 'New York',
    'Baltimorem': 'Baltimore',
    'Louisvillel': 'Louisville',
    'Washingtonk': 'Washington',
    'Indianapolisg': 'Indianapolis',
    'Virginia Beachm': 'Virginia Beach',
    'Philadelphiae': 'Philadelphia',
    'Nashvillej': 'Nashville',
    'Denveri': 'Denver',
    'Jacksonvillef': 'Jacksonville',
    'San Franciscoh': 'San Francisco'
}


def fix_city_directories(data_dir='./data', dry_run=True):
    """
    Fix directory names and update all image filenames inside them.
    
    Args:
        data_dir: Root data directory
        dry_run: If True, only print what would be changed without making changes
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    print(f"{'='*70}")
    print(f"Fixing City Names in: {data_path.absolute()}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (making changes)'}")
    print(f"{'='*70}\n")
    
    changes_made = 0
    
    # Get all subdirectories
    for old_dir in data_path.iterdir():
        if not old_dir.is_dir():
            continue
        
        old_name = old_dir.name
        
        # Check if this directory needs to be renamed
        if old_name not in CORRECTIONS:
            continue
        
        new_name = CORRECTIONS[old_name]
        new_dir = data_path / new_name
        
        print(f"Found: '{old_name}' -> '{new_name}'")
        
        # Check if target already exists
        if new_dir.exists() and not dry_run:
            print(f"  ⚠️  WARNING: Target directory '{new_name}' already exists!")
            print(f"     Skipping to avoid overwriting.")
            continue
        
        # Get all image files in this directory
        image_files = list(old_dir.glob('*.jpg')) + list(old_dir.glob('*.png'))
        print(f"  Found {len(image_files)} images to update")
        
        if not dry_run:
            # First, rename all files inside the directory
            for old_file in image_files:
                filename = old_file.name
                
                # Update filename to use new city name
                # Pattern: "CityName_ (lat, lon).jpg"
                if filename.startswith(old_name):
                    new_filename = filename.replace(old_name, new_name, 1)
                    new_file = old_file.parent / new_filename
                    
                    old_file.rename(new_file)
                    changes_made += 1
            
            # Then rename the directory itself
            old_dir.rename(new_dir)
            print(f"  ✓ Renamed directory and {len(image_files)} files")
        else:
            # Dry run - show what would be changed
            example_files = image_files[:3]  # Show first 3 examples
            for old_file in example_files:
                filename = old_file.name
                if filename.startswith(old_name):
                    new_filename = filename.replace(old_name, new_name, 1)
                    print(f"    {filename}")
                    print(f"    -> {new_filename}")
            
            if len(image_files) > 3:
                print(f"    ... and {len(image_files) - 3} more files")
        
        print()
        changes_made += 1
    
    print(f"{'='*70}")
    if dry_run:
        print(f"DRY RUN COMPLETE")
        print(f"Would fix {changes_made} directories")
        print(f"Run with dry_run=False to make actual changes")
    else:
        print(f"CHANGES COMPLETE")
        print(f"Fixed {changes_made} directories")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix city directory names with trailing characters'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Actually make changes (default is dry run)'
    )
    
    args = parser.parse_args()
    
    # Run with dry_run=True by default for safety
    fix_city_directories(
        data_dir=args.data_dir,
        dry_run=not args.live
    )