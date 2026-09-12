# Results Directory

This directory stores all visual and tabular outputs generated during experiments.

## 📂 Subdirectories
- `graphs/`: Loss curves, metric comparisons, ROC curves, histograms, bar charts, etc.
- `images/`: Input test images, intermediate feature maps, edge detection maps, filtered/enhanced images, segmentation masks, and visual comparisons.
- `tables/`: CSV, Excel, or markdown tables summarizing quantitative benchmarks (e.g., PSNR, SSIM, MSE, runtime).

## 💡 Best Practices
- Save high-resolution figures (`.png` or `.pdf` for vector graphics).
- Use clear filenames indicating method and metric (e.g., `canny_vs_sobel_comparison.png`, `psnr_benchmark_table.csv`).
- Keep raw bulky dataset files out of git; track only key sample outputs and evaluation artifacts.
