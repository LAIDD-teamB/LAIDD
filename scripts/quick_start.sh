#!/bin/bash
# LAIDD Pipeline Quick Start Script
# This script provides easy commands to run the LAIDD pipeline

set -e  # Exit on error

echo "=========================================="
echo "LAIDD Pipeline Quick Start"
echo "=========================================="
echo ""

# Function to print colored messages
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.7 or higher."
    exit 1
fi

print_info "Python version: $(python3 --version)"

# Show menu
echo ""
echo "Please select an option:"
echo "  1. Create default configuration file"
echo "  2. Run full pipeline (pretraining + finetuning)"
echo "  3. Run pretraining only"
echo "  4. Run finetuning only (requires pretrained model)"
echo "  5. Check data files"
echo "  6. View pipeline results"
echo "  7. Exit"
echo ""

read -p "Enter your choice [1-7]: " choice

case $choice in
    1)
        print_info "Creating default configuration file..."
        python3 run_all_pipeline.py --create-config
        print_success "Configuration file created: pipeline_config.json"
        print_info "Please edit this file to customize settings before running the pipeline."
        ;;

    2)
        print_info "Running full pipeline (pretraining + finetuning)..."
        print_warning "This may take several hours depending on your data size and hardware."
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            python3 run_all_pipeline.py --config pipeline_config.json
            print_success "Pipeline completed! Check results in pipeline_results/"
        else
            print_info "Cancelled."
        fi
        ;;

    3)
        print_info "Running pretraining only..."
        python3 run_all_pipeline.py --skip-finetuning
        print_success "Pretraining completed!"
        ;;

    4)
        print_info "Running finetuning only..."
        print_warning "Make sure you have a pretrained model specified in pipeline_config.json"
        python3 run_all_pipeline.py --skip-pretraining
        print_success "Finetuning completed!"
        ;;

    5)
        print_info "Checking data files..."
        echo ""

        if [ -f "data/pretraining_data.tsv" ]; then
            lines=$(wc -l < data/pretraining_data.tsv)
            print_success "Pretraining data found: data/pretraining_data.tsv ($lines lines)"
        else
            print_error "Pretraining data NOT found: data/pretraining_data.tsv"
        fi

        if [ -f "data/finetuning_data.tsv" ]; then
            lines=$(wc -l < data/finetuning_data.tsv)
            print_success "Finetuning data found: data/finetuning_data.tsv ($lines lines)"
        else
            print_error "Finetuning data NOT found: data/finetuning_data.tsv"
        fi

        echo ""
        print_info "Please ensure both data files are present before running the pipeline."
        ;;

    6)
        print_info "Viewing pipeline results..."
        echo ""

        if [ -f "pipeline_results/pipeline_results.json" ]; then
            print_success "Results found!"
            echo ""
            cat pipeline_results/pipeline_results.json | python3 -m json.tool
        else
            print_warning "No results found. Run the pipeline first."
        fi
        ;;

    7)
        print_info "Exiting..."
        exit 0
        ;;

    *)
        print_error "Invalid choice. Please select 1-7."
        exit 1
        ;;
esac

echo ""
print_info "Done!"
