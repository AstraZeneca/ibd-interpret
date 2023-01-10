### Immune cell detection and classification in H\&E slides

The model was trained on the publicly available annotated [CoNIC dataset](https://conic-challenge.grand-challenge.org/).
The dataset was part of a Grand Medical Challenges in July 2022.

#### How to run the trained model on your dataset

1. First, build a virtual environment and install pre-requisites specified in requirements.txt;
2. Prepare your H\&E dataset. You will need to split the slides into smaller 224x224 patches at 20x.
3. Run inference on your data. This is done in `inference.py` module;
4. Finally, you can visualise the predictions on cellular or on patch level in forms of heatmaps.
Everything is done in `visualisation.py` module.