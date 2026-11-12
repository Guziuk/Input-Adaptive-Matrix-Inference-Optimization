#!/bin/bash
# run_cutoff_tests.sh - Convenience script for running cutoff tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if executable exists
if [ ! -f "./test_cutoff_binary_search" ]; then
    echo -e "${RED}Error: test_cutoff_binary_search not found${NC}"
    echo "Please build it first:"
    echo "  cd build && cmake .. && make test_cutoff_binary_search"
    exit 1
fi

# Function to run a test with nice formatting
run_test() {
    local name=$1
    shift
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Running: $name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ./test_cutoff_binary_search "$@"
}

# Parse command line
COMMAND=${1:-help}

case $COMMAND in
    "quick")
        echo -e "${YELLOW}Quick Test - Default settings${NC}"
        run_test "Quick Test" --layer 0 --weight q_proj --effort 0.5
        ;;
    
    "effort-sweep")
        LAYER=${2:-0}
        WEIGHT=${3:-q_proj}
        
        echo -e "${YELLOW}Effort Sweep - Layer $LAYER, $WEIGHT${NC}"
        
        for effort in 0.1 0.3 0.5 0.7 0.9; do
            run_test "Effort $effort" --layer $LAYER --weight $WEIGHT --effort $effort
        done
        ;;
    
    "compare-weights")
        LAYER=${2:-0}
        EFFORT=${3:-0.5}
        
        echo -e "${YELLOW}Comparing All Weights - Layer $LAYER, Effort $EFFORT${NC}"
        
        for weight in q_proj k_proj v_proj o_proj gate_proj up_proj down_proj; do
            run_test "$weight" --layer $LAYER --weight $weight --effort $EFFORT
        done
        ;;
    
    "input-types")
        LAYER=${2:-0}
        WEIGHT=${3:-q_proj}
        EFFORT=${4:-0.5}
        
        echo -e "${YELLOW}Testing Input Types - Layer $LAYER, $WEIGHT, Effort $EFFORT${NC}"
        
        for input_type in random gaussian zeros ones; do
            run_test "Input: $input_type" --layer $LAYER --weight $WEIGHT \
                --effort $EFFORT --input $input_type
        done
        ;;
    
    "layers")
        WEIGHT=${2:-q_proj}
        EFFORT=${3:-0.7}
        
        echo -e "${YELLOW}Testing Multiple Layers - $WEIGHT, Effort $EFFORT${NC}"
        
        for layer in 0 5 10 15 20 25 30; do
            run_test "Layer $layer" --layer $layer --weight $WEIGHT --effort $EFFORT
        done
        ;;
    
    "export")
        LAYER=${2:-0}
        WEIGHT=${3:-q_proj}
        EFFORT=${4:-0.5}
        
        echo -e "${YELLOW}Exporting Data - Layer $LAYER, $WEIGHT, Effort $EFFORT${NC}"
        
        run_test "Export Test" --layer $LAYER --weight $WEIGHT \
            --effort $EFFORT --save-csv
        
        CSV_FILE="cutoff_test_layer${LAYER}_${WEIGHT}.csv"
        
        if [ -f "$CSV_FILE" ]; then
            echo -e "\n${GREEN}✓ CSV exported: $CSV_FILE${NC}"
            
            # Check if Python script exists
            if [ -f "../plot_cutoff.py" ]; then
                echo -e "${BLUE}Generating plots...${NC}"
                python3 ../plot_cutoff.py "$CSV_FILE" --save
                echo -e "${GREEN}✓ Plots generated${NC}"
            else
                echo -e "${YELLOW}Python script not found. Run manually:${NC}"
                echo "  python3 plot_cutoff.py $CSV_FILE --save"
            fi
        else
            echo -e "${RED}Error: CSV file not created${NC}"
        fi
        ;;
    
    "full-analysis")
        LAYER=${2:-0}
        WEIGHT=${3:-q_proj}
        
        echo -e "${YELLOW}Full Analysis - Layer $LAYER, $WEIGHT${NC}"
        echo -e "${YELLOW}This will test multiple efforts and generate visualizations${NC}"
        
        mkdir -p analysis_results
        
        for effort in 0.1 0.3 0.5 0.7 0.9; do
            echo -e "\n${BLUE}Testing effort=$effort...${NC}"
            ./test_cutoff_binary_search --layer $LAYER --weight $WEIGHT \
                --effort $effort --save-csv > "analysis_results/effort_${effort}.txt"
            
            CSV_FILE="cutoff_test_layer${LAYER}_${WEIGHT}.csv"
            if [ -f "$CSV_FILE" ]; then
                mv "$CSV_FILE" "analysis_results/cutoff_effort_${effort}.csv"
            fi
        done
        
        echo -e "\n${GREEN}✓ Analysis complete! Results in analysis_results/${NC}"
        echo -e "${BLUE}Generating plots...${NC}"
        
        if [ -f "../plot_cutoff.py" ]; then
            for csv in analysis_results/cutoff_effort_*.csv; do
                python3 ../plot_cutoff.py "$csv" --save --bins 128
            done
            echo -e "${GREEN}✓ All plots generated${NC}"
        fi
        
        echo -e "\n${YELLOW}Summary:${NC}"
        echo "  Text outputs:  analysis_results/effort_*.txt"
        echo "  CSV data:      analysis_results/cutoff_effort_*.csv"
        echo "  Plots:         analysis_results/cutoff_effort_*.png"
        ;;
    
    "benchmark")
        LAYER=${2:-0}
        WEIGHT=${3:-q_proj}
        EFFORT=${4:-0.5}
        ITERATIONS=${5:-10}
        
        echo -e "${YELLOW}Benchmarking - Layer $LAYER, $WEIGHT, Effort $EFFORT${NC}"
        echo -e "${YELLOW}Running $ITERATIONS iterations...${NC}"
        
        TOTAL_TIME=0
        
        for i in $(seq 1 $ITERATIONS); do
            echo -ne "\r${BLUE}Iteration $i/$ITERATIONS...${NC}"
            
            OUTPUT=$(./test_cutoff_binary_search --layer $LAYER --weight $WEIGHT \
                --effort $EFFORT 2>&1)
            
            # Extract execution time (this is a simple grep, adjust as needed)
            TIME=$(echo "$OUTPUT" | grep "Execution Time:" | awk '{print $3}')
            
            if [ -n "$TIME" ]; then
                TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
            fi
        done
        
        echo -e "\r${GREEN}✓ Benchmark complete!${NC}                    "
        
        AVG_TIME=$(echo "scale=3; $TOTAL_TIME / $ITERATIONS" | bc)
        
        echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}Benchmark Results${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo "  Iterations:   $ITERATIONS"
        echo "  Total time:   ${TOTAL_TIME} ms"
        echo "  Average time: ${AVG_TIME} ms"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        ;;
    
    "help"|*)
        echo -e "${BLUE}Cutoff Binary Search Test Runner${NC}"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo -e "${GREEN}Commands:${NC}"
        echo ""
        echo "  quick"
        echo "    Run a quick test with default settings"
        echo "    Example: $0 quick"
        echo ""
        echo "  effort-sweep [layer] [weight]"
        echo "    Test multiple effort levels (0.1, 0.3, 0.5, 0.7, 0.9)"
        echo "    Example: $0 effort-sweep 0 q_proj"
        echo ""
        echo "  compare-weights [layer] [effort]"
        echo "    Compare all weight types (q, k, v, o, gate, up, down)"
        echo "    Example: $0 compare-weights 0 0.5"
        echo ""
        echo "  input-types [layer] [weight] [effort]"
        echo "    Test different input types (random, gaussian, zeros, ones)"
        echo "    Example: $0 input-types 0 q_proj 0.5"
        echo ""
        echo "  layers [weight] [effort]"
        echo "    Test multiple layers (0, 5, 10, 15, 20, 25, 30)"
        echo "    Example: $0 layers q_proj 0.7"
        echo ""
        echo "  export [layer] [weight] [effort]"
        echo "    Export data to CSV and generate plots"
        echo "    Example: $0 export 0 q_proj 0.5"
        echo ""
        echo "  full-analysis [layer] [weight]"
        echo "    Complete analysis with multiple efforts and visualizations"
        echo "    Example: $0 full-analysis 0 q_proj"
        echo ""
        echo "  benchmark [layer] [weight] [effort] [iterations]"
        echo "    Benchmark kernel execution time"
        echo "    Example: $0 benchmark 0 q_proj 0.5 10"
        echo ""
        echo "  help"
        echo "    Show this help message"
        echo ""
        echo -e "${YELLOW}Weight types:${NC} q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"
        echo -e "${YELLOW}Effort range:${NC} 0.0 (no pruning) to 1.0 (maximum pruning)"
        echo ""
        ;;
esac