"""
DIP Exploratory Data Analysis (EDA) Module
Analyzes the AID Multi-Label Dataset:
- Class frequency distribution across 17 classes
- Label co-occurrence matrix
- Generates summary statistics and distribution bar chart
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_eda(csv_path: Path, output_graph_path: Path = None):
    """
    Run exploratory data analysis on multi-label annotations.
    """
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    img_col = df.columns[0]
    classes = [c.strip() for c in df.columns[1:]]

    print(f"Total Images: {len(df)}")
    print(f"Total Multi-Label Classes: {len(classes)}")

    class_counts = df[classes].sum().sort_values(ascending=False)
    print("\nClass Frequency Distribution:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:<15}: {count:>5} images ({count / len(df) * 100:.1f}%)")

    # Labels per image
    labels_per_img = df[classes].sum(axis=1)
    print(f"\nLabels per image — Mean: {labels_per_img.mean():.2f}, Min: {labels_per_img.min()}, Max: {labels_per_img.max()}")

    # Plot class distribution
    if output_graph_path:
        output_graph_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6), dpi=300)
        bars = plt.bar(class_counts.index, class_counts.values, color='#1f77b4', edgecolor='black', alpha=0.85)
        plt.xlabel('Aerial Class', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Annotations', fontsize=12, fontweight='bold')
        plt.title('AID Multi-Label Dataset: Class Frequency Distribution', fontsize=14, fontweight='bold', pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_graph_path)
        print(f"\nDistribution chart saved to: {output_graph_path}")


if __name__ == '__main__':
    csv_file = Path('multilabel .csv')
    if not csv_file.exists():
        csv_file = Path('Codes/data/multilabel.csv')
    run_eda(csv_file, Path('Results/graphs/class_distribution.png'))
