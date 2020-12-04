# ai_thyroid
An Ultrasound Computer-Assisted Diagnosis Model to Improve the Diagnostic Accuracy of the Suspicious Thyroid Nodules

## Clinical feature extraction
Module: ./ai_thyroid_clinical_feature 
- This module covers feature extraction of Aspect Ratio, Echogenicity and Composition
- To run example: feature_demo.ipynb

## Calcification segmentation
Module: ./ai_thyroid_calcification
- This module covers feature extraction of Calcification
- To train model: calcification_training.ipynb
- To run example: calcification_example.ipynb

## Malignancy assessment
Module: ./ai_thyroid_malignancy
- This module covers nodule malignancy classification
- To train and evaluate model: malignancy.ipynb

## Multi-task Net
Module: ./ai_thyroid_multi_task
- Experimental Multi-task segmentation and classification module, for simplifying workflow
- This module could perform Lesion segmentation, Calcification segmentation and Malignancy assessment
- No performance reduction in segmentation task
