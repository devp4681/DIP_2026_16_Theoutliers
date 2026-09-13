# Week 1 Progress Report

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

## 3. Planned Pipeline

Feature extraction (color / texture / shape) → similarity → ranking → explainability

