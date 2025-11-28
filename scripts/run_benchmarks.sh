#!/bin/bash
#
# RuVector Comprehensive Benchmark Suite Runner
# Proof ID: ed2551
# Version: 0.1.16
#
# Usage: ./scripts/run_benchmarks.sh [options]
#
# Options:
#   --quick       Run quick benchmarks (smaller datasets)
#   --full        Run full benchmark suite (may take hours)
#   --ablation    Run only GNN ablation study
#   --beir        Run only BEIR benchmarks
#   --ann         Run only ANN-benchmarks export
#   --help        Show this help message

set -e

PROOF_ID="ed2551"
VERSION="0.1.16"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
OUTPUT_BASE="bench_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
QUICK_MODE=false
FULL_MODE=false
RUN_ABLATION=true
RUN_BEIR=true
RUN_ANN=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        --ablation)
            RUN_ABLATION=true
            RUN_BEIR=false
            RUN_ANN=false
            shift
            ;;
        --beir)
            RUN_ABLATION=false
            RUN_BEIR=true
            RUN_ANN=false
            shift
            ;;
        --ann)
            RUN_ABLATION=false
            RUN_BEIR=false
            RUN_ANN=true
            shift
            ;;
        --help)
            echo "RuVector Benchmark Suite Runner"
            echo ""
            echo "Usage: ./scripts/run_benchmarks.sh [options]"
            echo ""
            echo "Options:"
            echo "  --quick       Run quick benchmarks (smaller datasets)"
            echo "  --full        Run full benchmark suite (may take hours)"
            echo "  --ablation    Run only GNN ablation study"
            echo "  --beir        Run only BEIR benchmarks"
            echo "  --ann         Run only ANN-benchmarks export"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          RuVector Comprehensive Benchmark Suite             ║"
echo "║                    Proof ID: ${PROOF_ID}                         ║"
echo "║                    Version: ${VERSION}                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Timestamp: ${TIMESTAMP}${NC}"
echo ""

# Set parameters based on mode
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}Running in QUICK mode (smaller datasets)${NC}"
    NUM_VECTORS=10000
    NUM_QUERIES=100
    RUNS=1
    EF_SEARCH_VALUES="50,100"
    M_VALUES="16,32"
    EF_CONSTRUCTION_VALUES="100"
elif [ "$FULL_MODE" = true ]; then
    echo -e "${YELLOW}Running in FULL mode (large datasets, may take hours)${NC}"
    NUM_VECTORS=1000000
    NUM_QUERIES=10000
    RUNS=5
    EF_SEARCH_VALUES="10,20,40,80,120,200,400,800"
    M_VALUES="8,16,32,64"
    EF_CONSTRUCTION_VALUES="100,200,400"
else
    echo -e "${YELLOW}Running in DEFAULT mode${NC}"
    NUM_VECTORS=100000
    NUM_QUERIES=1000
    RUNS=3
    EF_SEARCH_VALUES="50,100,200,400"
    M_VALUES="16,32"
    EF_CONSTRUCTION_VALUES="100,200"
fi

echo ""

# Create output directories
mkdir -p "${OUTPUT_BASE}/ablation"
mkdir -p "${OUTPUT_BASE}/beir"
mkdir -p "${OUTPUT_BASE}/ann-benchmarks"

# Record environment info
echo -e "${BLUE}Recording environment information...${NC}"
ENV_FILE="${OUTPUT_BASE}/environment.json"
cat > "${ENV_FILE}" << EOF
{
  "proof_id": "${PROOF_ID}",
  "version": "${VERSION}",
  "timestamp": "${TIMESTAMP}",
  "rust_version": "$(rustc --version | cut -d' ' -f2)",
  "cargo_version": "$(cargo --version | cut -d' ' -f2)",
  "os": "$(uname -s) $(uname -r)",
  "arch": "$(uname -m)",
  "cpu_info": "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d':' -f2 | xargs || echo 'unknown')",
  "memory_gb": $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0"),
  "quick_mode": ${QUICK_MODE},
  "full_mode": ${FULL_MODE}
}
EOF
echo -e "${GREEN}✓ Environment recorded to ${ENV_FILE}${NC}"
echo ""

# Build benchmarks
echo -e "${BLUE}Building benchmark binaries...${NC}"
cargo build --release -p ruvector-bench 2>&1 | tail -5
echo -e "${GREEN}✓ Build complete${NC}"
echo ""

# Run GNN Ablation Study
if [ "$RUN_ABLATION" = true ]; then
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "                    GNN ABLATION STUDY"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    cargo run --release --bin gnn_ablation_benchmark -- \
        --num-vectors ${NUM_VECTORS} \
        --num-queries ${NUM_QUERIES} \
        --dimensions 128 \
        --k 10 \
        --runs ${RUNS} \
        --output "${OUTPUT_BASE}/ablation" \
        --distribution normal
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo -e "${GREEN}✓ GNN Ablation complete in ${DURATION}s${NC}"
    echo ""
fi

# Run BEIR Benchmarks
if [ "$RUN_BEIR" = true ]; then
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "                    BEIR BENCHMARKS"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    cargo run --release --bin beir_benchmark -- \
        --dataset synthetic \
        --num-docs ${NUM_VECTORS} \
        --num-queries ${NUM_QUERIES} \
        --dimensions 384 \
        --max-k 100 \
        --ef-search-values ${EF_SEARCH_VALUES} \
        --output "${OUTPUT_BASE}/beir"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo -e "${GREEN}✓ BEIR Benchmarks complete in ${DURATION}s${NC}"
    echo ""
fi

# Run ANN-Benchmarks Export
if [ "$RUN_ANN" = true ]; then
    echo -e "${BLUE}"
    echo "════════════════════════════════════════════════════════════════"
    echo "                 ANN-BENCHMARKS EXPORT"
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    # SIFT1M-like benchmark
    cargo run --release --bin ann_benchmarks_export -- \
        --dataset sift1m \
        --num-vectors ${NUM_VECTORS} \
        --num-queries ${NUM_QUERIES} \
        --k 10 \
        --m-values ${M_VALUES} \
        --ef-construction-values ${EF_CONSTRUCTION_VALUES} \
        --ef-search-values ${EF_SEARCH_VALUES} \
        --output "${OUTPUT_BASE}/ann-benchmarks"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo -e "${GREEN}✓ ANN-Benchmarks Export complete in ${DURATION}s${NC}"
    echo ""
fi

# Generate summary report
echo -e "${BLUE}Generating summary report...${NC}"
REPORT_FILE="${OUTPUT_BASE}/BENCHMARK_REPORT.md"

cat > "${REPORT_FILE}" << EOF
# RuVector Benchmark Report

**Proof ID:** ${PROOF_ID}
**Version:** ${VERSION}
**Generated:** ${TIMESTAMP}

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Vectors | ${NUM_VECTORS} |
| Queries | ${NUM_QUERIES} |
| Runs | ${RUNS} |
| Mode | $([ "$FULL_MODE" = true ] && echo "Full" || ([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Default")) |

## Results Summary

EOF

# Add ablation results if available
if [ -f "${OUTPUT_BASE}/ablation/ablation_summary.csv" ]; then
    echo "### GNN Ablation Study" >> "${REPORT_FILE}"
    echo "" >> "${REPORT_FILE}"
    echo "| Configuration | Mean QPS | Recall@10 | p99 (ms) | Improvement |" >> "${REPORT_FILE}"
    echo "|--------------|----------|-----------|----------|-------------|" >> "${REPORT_FILE}"
    tail -n +2 "${OUTPUT_BASE}/ablation/ablation_summary.csv" | while IFS=',' read -r name qps std_qps recall std_recall lat std_lat mem imp; do
        printf "| %s | %.0f | %.2f%% | %.2f | %+.1f%% |\n" "${name}" "${qps}" "$(echo "${recall} * 100" | bc -l 2>/dev/null || echo ${recall})" "${lat}" "${imp}" >> "${REPORT_FILE}"
    done
    echo "" >> "${REPORT_FILE}"
fi

# Add BEIR results if available
if [ -f "${OUTPUT_BASE}/beir/beir_synthetic_results.csv" ]; then
    echo "### BEIR Evaluation" >> "${REPORT_FILE}"
    echo "" >> "${REPORT_FILE}"
    echo "| ef_search | NDCG@10 | MAP@10 | Recall@10 | Recall@100 | QPS |" >> "${REPORT_FILE}"
    echo "|-----------|---------|--------|-----------|------------|-----|" >> "${REPORT_FILE}"
    tail -n +2 "${OUTPUT_BASE}/beir/beir_synthetic_results.csv" | while IFS=',' read -r dataset split ef ndcg map recall100 mrr qps p99; do
        printf "| %s | %.4f | %.4f | %.4f | %.4f | %.0f |\n" "${ef}" "${ndcg}" "${map}" "${recall100}" "${recall100}" "${qps}" >> "${REPORT_FILE}"
    done
    echo "" >> "${REPORT_FILE}"
fi

# Add ANN-benchmarks results if available
if [ -f "${OUTPUT_BASE}/ann-benchmarks/sift1m_results.csv" ]; then
    echo "### ANN-Benchmarks (SIFT1M-like)" >> "${REPORT_FILE}"
    echo "" >> "${REPORT_FILE}"
    echo "| Parameters | Recall | QPS | p99 (ms) | Memory (MB) |" >> "${REPORT_FILE}"
    echo "|------------|--------|-----|----------|-------------|" >> "${REPORT_FILE}"
    tail -n +2 "${OUTPUT_BASE}/ann-benchmarks/sift1m_results.csv" | head -10 | while IFS=',' read -r alg params recall qps lat p99 build mem; do
        printf "| %s | %.4f | %.0f | %.2f | %.1f |\n" "${params}" "${recall}" "${qps}" "${p99}" "${mem}" >> "${REPORT_FILE}"
    done
    echo "" >> "${REPORT_FILE}"
fi

cat >> "${REPORT_FILE}" << EOF

## File Locations

- Ablation Results: \`${OUTPUT_BASE}/ablation/\`
- BEIR Results: \`${OUTPUT_BASE}/beir/\`
- ANN-Benchmarks: \`${OUTPUT_BASE}/ann-benchmarks/\`
- Environment: \`${ENV_FILE}\`

## Reproducibility

To reproduce these results:

\`\`\`bash
git checkout ${PROOF_ID}
cargo build --release -p ruvector-bench
./scripts/run_benchmarks.sh $([ "$FULL_MODE" = true ] && echo "--full" || ([ "$QUICK_MODE" = true ] && echo "--quick" || echo ""))
\`\`\`

---
*Generated by RuVector Benchmark Suite | Proof ID: ${PROOF_ID}*
EOF

echo -e "${GREEN}✓ Report generated: ${REPORT_FILE}${NC}"
echo ""

# Print final summary
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  BENCHMARK SUITE COMPLETE                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Results saved to: ${OUTPUT_BASE}/${NC}"
echo ""
echo "Output files:"
ls -la "${OUTPUT_BASE}"/*.{json,md} 2>/dev/null || true
echo ""
echo -e "${YELLOW}Proof ID: ${PROOF_ID}${NC}"
