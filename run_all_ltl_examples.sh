#!/bin/bash
# Run belief_support_algorithm on all LTL examples with TLSF files

OUTPUT_DIR="results_ltl_examples"
mkdir -p "$OUTPUT_DIR"

echo "Running belief_support_algorithm on all LTL examples..."
echo "Results will be saved to $OUTPUT_DIR/"
echo ""

# Counter for progress
total=$(find examples/ltl -name "*.tlsf" | wc -l | tr -d ' ')
current=0

# Iterate through all TLSF files
for tlsf_file in examples/ltl/*.tlsf; do
    current=$((current + 1))
    
    # Get the corresponding POMDP file (same name but .pomdp extension)
    base_name=$(basename "$tlsf_file" .tlsf)
    pomdp_file="examples/ltl/${base_name}.pomdp"
    
    # Check if POMDP file exists
    if [ ! -f "$pomdp_file" ]; then
        echo "[$current/$total] SKIP: No matching POMDP for $tlsf_file"
        continue
    fi
    
    # Output file for this example
    output_file="$OUTPUT_DIR/${base_name}.txt"
    
    echo "[$current/$total] Running: $base_name"
    
    # Run belief_support_algorithm with 30 second timeout
    # Use a wrapper script that implements timeout
    uv run python -c "
import subprocess
import sys

timeout = 30
try:
    result = subprocess.run(
        ['uv', 'run', 'python', 'belief_support_algorithm.py',
         '--file', '$pomdp_file',
         '--tlsf_file', '$tlsf_file'],
        timeout=timeout,
        capture_output=False
    )
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print(f'\nTIMEOUT: Computation exceeded {timeout} seconds', file=sys.stderr)
    sys.exit(124)  # Use same exit code as GNU timeout
" > "$output_file" 2>&1
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ] || [ $exit_code -eq 134 ]; then
        # Success (0) or spot memory warning (134)
        # Extract key results
        winning_states=$(grep "Almost-sure winning POMDP states:" "$output_file" | sed 's/.*: //')
        belief_states=$(grep "from .* belief-support states" "$output_file" | sed 's/.*from //' | sed 's/ belief-support.*//')
        echo "  [OK] $winning_states ($belief_states BS states)"
    elif [ $exit_code -eq 124 ]; then
        # timeout command returns 124 when timeout is reached
        echo "  [TIMEOUT] Exceeded 30 seconds"
        echo "TIMEOUT: Computation exceeded 30 seconds" >> "$output_file"
    elif [ $exit_code -eq 2 ]; then
        echo "  [TIMEOUT] Exceeded 30 seconds"
    else
        echo "  [FAILED] Exit code: $exit_code"
    fi
done

echo ""
echo "Complete! Results saved to $OUTPUT_DIR/"
echo ""
echo "Summary:"
success_count=$(grep -l 'COMPUTATION COMPLETE' $OUTPUT_DIR/*.txt 2>/dev/null | wc -l | tr -d ' ')
timeout_count=$(grep -l 'TIMEOUT:' $OUTPUT_DIR/*.txt 2>/dev/null | wc -l | tr -d ' ')
failed_count=$(($total - $success_count - $timeout_count))
echo "  Success:  $success_count/$total"
echo "  Timeout:  $timeout_count/$total"
echo "  Failed:   $failed_count/$total"
