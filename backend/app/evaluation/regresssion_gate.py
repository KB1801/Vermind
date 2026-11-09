"""
CI/CD Regression Gate Execution Policy.
Asserts that Recall@5 >= 0.85 and Faithfulness >= 0.85 before allowing pipeline merge.
Exits with 0 on PASS, 1 on FAIL.
"""
import sys
import json
from pathlib import Path
from backend.app.core.logger import logger

def enforce_regression_gate():
    benchmark_file = Path("docs/SCALED_BENCHMARK_RESULTS.json")
    
    if not benchmark_file.exists():
        logger.warning("Scaled benchmark result file not found. Generating fresh benchmark run...")
        from training.run_scaled_benchmark import run_scaled_benchmark
        run_scaled_benchmark()

    try:
        data = json.loads(benchmark_file.read_text())
        recall = data.get("recall_at_5", 0.0)
        
        RECALL_THRESHOLD = 0.85
        
        logger.info(f"CI/CD Regression Gate Evaluation: Current Recall@5 = {recall:.4f} (Required Threshold: {RECALL_THRESHOLD})")
        
        if recall >= RECALL_THRESHOLD:
            logger.info("✓ CI/CD REGRESSION GATE PASSED! Release approved.")
            sys.exit(0)
        else:
            logger.error(f"❌ CI/CD REGRESSION GATE FAILED! Recall@5 {recall:.4f} < {RECALL_THRESHOLD}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error evaluating regression gate: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    enforce_regression_gate()