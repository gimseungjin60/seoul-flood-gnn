"""Week 1 파이프라인: 원본 CSV → data/processed/sewer_nodes_cleaned.csv + data_quality_report.txt"""
from src.data import clean_nodes

if __name__ == "__main__":
    clean_nodes.main()
