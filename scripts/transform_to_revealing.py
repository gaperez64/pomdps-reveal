#!/usr/bin/env python3
"""
Transform all POMDP files in the examples directory to be strongly revealing.
Only transforms files that are not already strongly revealing.
"""

import os
import sys
import signal
import shutil
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from pomdpy.parsers import pomdp as pomdp_parser
from pomdpy.revealing import is_strongly_revealing, make_strongly_revealing


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


def count_lines(filepath):
    """Count the number of lines in a file"""
    with open(filepath) as f:
        return sum(1 for _ in f)


def transform_pomdp_file(input_path, output_path):
    """
    Transform a single POMDP file to be strongly revealing.
    Returns True if successful, False if skipped/failed.
    """
    try:
        # Check file size first
        line_count = count_lines(input_path)
        if line_count > 1500:
            print(f"⊘ Skipped (too large: {line_count} lines)")
            return False
        
        print("parsing...", end=" ", flush=True)
        with open(input_path) as f:
            content = f.read()
        
        pomdp = pomdp_parser.parse(content)
        
        # Check if already strongly revealing
        print("checking...", end=" ", flush=True)
        if is_strongly_revealing(pomdp):
            print("already revealing, skipping")
            return False
        
        print("transforming...", end=" ", flush=True)
        revealing_pomdp = make_strongly_revealing(pomdp)
        
        print("writing...", end=" ", flush=True)
        revealing_pomdp.to_pomdp_file(output_path)

        # Also copy/rename matching TLSF file if present
        tlsf_in = Path(input_path).with_suffix(".tlsf")
        tlsf_out = Path(output_path).with_suffix(".tlsf")
        if tlsf_in.exists():
            shutil.copy2(tlsf_in, tlsf_out)
        else:
            print("(no TLSF found)", end=" ")
        
        print("✓")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Process all POMDP files in the examples directory"""
    examples_dir = Path("examples")
    
    # Process ltl and parity folders separately
    ltl_dir = examples_dir / "ltl"
    parity_dir = examples_dir / "parity"
    ltl_revealing_dir = examples_dir / "ltl-revealing"
    parity_revealing_dir = examples_dir / "parity-revealing"
    
    # Create output directories
    ltl_revealing_dir.mkdir(exist_ok=True)
    parity_revealing_dir.mkdir(exist_ok=True)
    
    # Find all .pomdp files in ltl and parity folders
    pomdp_files = []
    if ltl_dir.exists():
        for filepath in ltl_dir.glob("**/*.pomdp"):
            if filepath.is_file():
                pomdp_files.append((filepath, ltl_revealing_dir))
    
    if parity_dir.exists():
        for filepath in parity_dir.glob("**/*.pomdp"):
            if filepath.is_file():
                pomdp_files.append((filepath, parity_revealing_dir))
    
    pomdp_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(pomdp_files)} POMDP files to process")
    print(f"  LTL: {len([x for x in pomdp_files if 'ltl' in str(x[0])])}")
    print(f"  Parity: {len([x for x in pomdp_files if 'parity' in str(x[0])])}")
    print("=" * 60)
    
    success_count = 0
    skipped_count = 0
    already_revealing_count = 0
    timeout_count = 0
    fail_count = 0
    
    for filepath, output_base_dir in pomdp_files:
        # Determine relative path to preserve subdirectory structure
        if 'ltl' in str(filepath):
            relative = filepath.relative_to(examples_dir / "ltl")
        else:
            relative = filepath.relative_to(examples_dir / "parity")
        
        output_path = output_base_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Display relative path for context
        display_name = str(relative) if 'ltl' in str(filepath) or 'parity' in str(filepath) else filepath.name
        print(f"Processing {display_name}... ", end="", flush=True)
        
        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        try:
            # Check file size first
            line_count = count_lines(filepath)
            if line_count > 3000:
                signal.alarm(0)
                print(f"⊘ Skipped (too large: {line_count} lines)")
                skipped_count += 1
                continue
            
            print("parsing...", end=" ", flush=True)
            with open(filepath) as f:
                content = f.read()
            
            pomdp = pomdp_parser.parse(content)
            
            # Check if already strongly revealing
            print("checking...", end=" ", flush=True)
            if is_strongly_revealing(pomdp):
                signal.alarm(0)
                print("already revealing, copying... ", end="", flush=True)
                # Copy the file with revealing_ prefix
                shutil.copy2(filepath, output_path)
                # Also copy matching TLSF file (if present)
                tlsf_in = filepath.with_suffix(".tlsf")
                tlsf_out = output_path.with_suffix(".tlsf")
                if tlsf_in.exists():
                    shutil.copy2(tlsf_in, tlsf_out)
                else:
                    print("(no TLSF found) ", end="", flush=True)
                print("✓")
                already_revealing_count += 1
                continue
            
            print("transforming...", end=" ", flush=True)
            revealing_pomdp = make_strongly_revealing(pomdp)
            
            print("writing...", end=" ", flush=True)
            revealing_pomdp.to_pomdp_file(output_path)
            # Also copy matching TLSF file (if present)
            tlsf_in = filepath.with_suffix(".tlsf")
            tlsf_out = output_path.with_suffix(".tlsf")
            if tlsf_in.exists():
                shutil.copy2(tlsf_in, tlsf_out)
            else:
                print("(no TLSF found)", end=" ", flush=True)
            
            signal.alarm(0)
            print("✓")
            success_count += 1
        
        except TimeoutError:
            signal.alarm(0)
            print("⊘ Timeout (>30s)")
            timeout_count += 1
            
        except Exception as e:
            signal.alarm(0)
            print(f"✗ Error: {e}")
            fail_count += 1
    
    print("=" * 60)
    print("Summary:")
    print(f"  Successfully transformed: {success_count}")
    print(f"  Already strongly revealing: {already_revealing_count}")
    print(f"  Skipped (too large): {skipped_count}")
    print(f"  Timeout: {timeout_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(pomdp_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
