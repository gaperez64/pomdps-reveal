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
    
    # Find all .pomdp files, excluding revealing_*.pomdp files
    pomdp_files = []
    for root, dirs, files in os.walk(examples_dir):
        for filename in files:
            if filename.endswith(".pomdp") and not filename.startswith("revealing_"):
                pomdp_files.append(Path(root) / filename)
    
    pomdp_files.sort()
    
    print(f"Found {len(pomdp_files)} POMDP files to process")
    print("=" * 60)
    
    success_count = 0
    skipped_count = 0
    already_revealing_count = 0
    timeout_count = 0
    fail_count = 0
    
    for filepath in pomdp_files:
        # Create output filename: revealing_<original_name>
        output_path = filepath.parent / f"revealing_{filepath.name}"
        
        print(f"Processing {filepath.name}... ", end="", flush=True)
        
        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            # Check file size first
            line_count = count_lines(filepath)
            if line_count > 1500:
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
