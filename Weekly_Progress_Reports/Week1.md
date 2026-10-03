-# Week 1 Progress Report

## 1. What the Project Needs

Develop an explainable classical computer-vision-based CBIR system for multi-label aerial images that retrieves visually and semantically similar images based on handcrafted features.

### Input
- AID Multi-label aerial image dataset — https://github.com/Hua-YS/AID-Multilabel-Dataset
- Query image with multi-label annotations
- Images with variations in scale, orientation, illumination, object combinations, and scene complexity
- Handcrafted features: color, texture, shape, and spatial features

### Expected Output
- Top-K ranked similar images
- Similarity/relevance score for different thresholds
- Multi-label semantic similarity
- Visual explanation of matching regions/features
- Identification of important visual characteristics
- Detection of uncertain or unreliable retrievals
- Performance evaluation with suitable metrics

## 2. Dataset

Dataset Name: AID Multi-Label Dataset

Dataset Link: https://github.com/Hua-YS/AID-Multilabel-Dataset

Dataset Details:
- Total images: 3000
- Training images: 2400
- Test images: 600
- Number of classes/labels: 17
- Image size: 600 × 600 pixels
- Label format: To be confirmed from the downloaded dataset structure
- Note: Dataset is already provided (no separate dataset collection required)

## 3. Planned Pipeline

Feature extraction (color / texture / shape) → similarity → ranking → explainability

## 4. Task Split

| Team Member | Responsibility |
|---|---|
| Dev Patel | Data handling and processing |
| Dhruvi Faldu | Feature extraction - color, texture and shape |
| Aastha Gandhi | Similarity calculation and ranking |
| Aashi Bhimjiyani | Explainability, evaluation and documentation |

This task split is tentative and may be adjusted as the project progresses.
*(Note: Since the dataset is already provided, no data collection is required; work focuses on data loading, preprocessing, and pipeline development).*





## Week 1 Progress
### 1. Dataset and Image Verification

- Dataset used: AID Multi-label aerial image dataset
- Total images: 3,000
- Training images: 2,400
- Test images: 600
- Image size: 600 × 600
- Number of object labels: 17
- All 3,000 images were checked for image validity.
- No problematic or corrupt images were identified.
- Pillow was used for reliable RGB image loading during feature extraction.

### 2. Feature Extraction

- HSV color histogram was used as the color descriptor.
- LBP (Local Binary Pattern) was used as the texture descriptor.
- HOG (Histogram of Oriented Gradients) was used as the shape/edge descriptor.
- Feature extraction was performed for both training and test images.
- Extracted features were cached to avoid repeated computation.

### 3. CBIR Retrieval Methods

Three individual feature-based retrieval methods were implemented:

- HSV color histogram retrieval
- LBP texture retrieval
- HOG shape/edge retrieval

Feature fusion was also tested by combining:
- HSV + LBP
- HSV + LBP + HOG

Similarity scores were used to rank the training images for each test query.

### 4. Evaluation

- Retrieval was evaluated using the 600 test images as queries against the 2,400 training images.
- Class-based and multi-label retrieval metrics were used.
- Precision@5 and Precision@10 were calculated.
- Mean Average Precision (mAP) was also calculated.
- Jaccard similarity and nDCG were used for multi-label retrieval evaluation.
- Results were saved in CSV format for further analysis.

### 5. Week 1 Results

- HSV provided the strongest individual baseline among the tested feature types.
- LBP provided useful complementary texture information.
- HOG performed weaker as an individual descriptor but contributed useful information when combined with other features.
- Feature fusion generally improved retrieval performance compared with individual descriptors.
- Among the tested configurations, HSV + LBP + HOG with weights 0.50, 0.25, and 0.25 gave the strongest overall results.
- Performance varied across different scene classes.

### 6. Week 1 Conclusion

Week 1 established a classical-CV CBIR baseline for the AID Multi-label dataset. HSV color, LBP texture, and HOG shape/edge descriptors were implemented and evaluated individually and in combination. Feature caching was introduced to make repeated experiments more efficient. Evaluation was performed using the 600 test images as queries against the 2,400 training images. Overall, feature fusion provided improved retrieval performance among the tested configurations, with performance varying across different scene classes.
