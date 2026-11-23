🛰️ Satellite Imagery Classification: Agricultural vs Non-Agricultural Land

Binary classification of satellite image tiles into agricultural and non-agricultural land, comparing CNNs and CNN–Vision Transformer hybrids implemented in both Keras/TensorFlow and PyTorch.

📚 Course context: Final project of AI Capstone Project with Deep Learning (IBM, Coursera). The scenario, dataset, and notebook template were provided by the course; the implementation, training, and evaluation in this repository are my own work on top of that template.

🎯 Objective

In the course scenario, a fertilizer company wants to expand into new territory and needs to find agricultural land automatically instead of classifying imagery by hand. The task is to label each 64×64 tile as agri or non-agri.

📦 Dataset
6,000 RGB image tiles, perfectly balanced (3,000 per class), provided by the course
File names follow the Sentinel-2 Level-2A convention (acquired April 2025), so the imagery appears to derive from Copernicus Sentinel-2 data
Classes: class_0_non_agri and class_1_agri
80/20 train/validation split (4,800 training and 1,200 validation images)
Images resized to 64×64 and normalized
🧠 Project Structure

1. Data loading & augmentation

A custom Python generator for lazy, batch-by-batch image loading (no need to hold the dataset in memory)
A Keras tf.data pipeline (image_dataset_from_directory) with random flips and rotation, plus cache() and prefetch() for speed, benchmarked at 4.5 s for two passes over the training set
A PyTorch pipeline with ImageFolder and torchvision transforms (resize, flips, rotation, normalization)

2. CNN classifiers (Keras vs PyTorch)

Keras CNN: 4 × Conv2D (5×5 kernels, 32→256 filters) with MaxPooling and BatchNorm, global average pooling, then 4 Dense layers (64, 128, 256, 512) with BatchNorm and Dropout 0.4, and a sigmoid output. Trained with Adam (LR 1e-3), batch size 128, 3 epochs, using ImageDataGenerator augmentation (rotation, shifts, shear, zoom, horizontal flip).
PyTorch CNN: 6 × Conv2d (5×5 kernels, 32→1024 filters) with MaxPooling and BatchNorm, adaptive average pooling, Linear(2048) with BatchNorm and Dropout 0.4, and a 2-class output. Trained with Adam (LR 1e-3), cross-entropy loss, batch size 128, 3 epochs.

3. CNN–Vision Transformer hybrids

Keras hybrid: intermediate feature maps from the frozen, previously trained Keras CNN are flattened into tokens, given learned positional embeddings, and passed through 4 Transformer encoder blocks (8 attention heads, MLP dimension 2048), then global average pooling and a softmax output. Adam (LR 1e-4), batch size 4, 128 steps per epoch, 3 epochs.
PyTorch hybrid: CNN feature extractor initialized from the trained PyTorch CNN, followed by a 1×1-convolution patch embedding (1024→768), a class token, learned positional embeddings, and 3 Transformer blocks (6 heads, dropout 0.1), trained end to end with Adam (LR 1e-3), batch size 32, 5 epochs, keeping the checkpoint with the lowest validation loss.

4. Evaluation

Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and training/validation curves for all four models
📊 Results
Model	Validation accuracy (held-out 20%)	Accuracy on all 6,000 images*
Keras CNN	74.8%	70.3%
PyTorch CNN	99.7%	99.8%
Keras CNN–ViT hybrid	98.7%	99.0%
PyTorch CNN–ViT hybrid	99.1%	99.4%

Full-dataset metrics for the agri class:

Model	Precision	Recall	F1-score	ROC-AUC
Keras CNN	0.840	0.502	0.629	0.820
PyTorch CNN	0.998	0.998	0.998	1.000
Keras CNN–ViT hybrid	0.999	0.981	0.990	1.000
PyTorch CNN–ViT hybrid	0.991	0.996	0.994	1.000

* The final model comparison runs on the full dataset, which includes the training images, so these figures are optimistic. The validation column is the more reliable estimate.

Observations

The PyTorch CNN and both hybrids reach about 99% validation accuracy. The Keras CNN lagged and was unstable across epochs (validation accuracy 52% → 48% → 75%) and missed about half of the agricultural tiles (recall 0.50).
The Keras hybrid, built on top of that weaker CNN, reached 98.7% validation accuracy, suggesting that the Transformer blocks add useful global context on top of the CNN features.
All models were trained for only 3–5 epochs on a CPU.
⚠️ Known Limitations
No held-out test set: the final comparison reuses the full dataset (see the note above).
Possible spatial leakage: the tiles come from only three Sentinel-2 scenes and were split randomly, so neighboring tiles may appear in both the training and validation sets, which can inflate scores.
PyTorch augmentation: the training and validation subsets share one underlying dataset object, so the validation transform overrides the augmentation transform and the PyTorch models were effectively trained without augmentation.
Keras validation: the validation generator reuses the augmenting ImageDataGenerator, so validation images are also augmented.
🛠️ Tech Stack

Python · TensorFlow / Keras · PyTorch / torchvision · scikit-learn · NumPy · Matplotlib · Pillow · tqdm · Jupyter Notebook / Google Colab

🚀 How to Run

The dataset (images_dataSAT.tar) is provided by the course environment and is not included in this repository. Place the archive next to the notebook; the first notebook cell extracts it with tar. After downloading or cloning this repository, open a terminal in its folder and run:

bash
pip install tensorflow torch torchvision scikit-learn matplotlib pillow tqdm
# place images_dataSAT.tar in this folder, then:
jupyter notebook Satellite_Imagery_Classification_via_Deep_Learning1.ipynb

Training on a CPU is slow (a few minutes per epoch). A GPU runtime such as Google Colab is recommended.

📁 Repository Structure
├── Satellite_Imagery_Classification_via_Deep_Learning1.ipynb   # Main notebook
├── satellite_imagery_classification_via_deep_learning1.py      # Script version
└── README.md
🔮 Future Improvements
Create a true held-out test set and split by geographic scene to avoid spatial leakage
Fix the PyTorch data pipeline so the training and validation subsets use separate transforms
Train longer, tune hyperparameters systematically, and add confusion-matrix plots to the report
Try pre-trained backbones (ResNet, EfficientNet) and pre-trained Vision Transformers
🙏 Acknowledgments

Project structure, scenario, and dataset provided by the AI Capstone Project with Deep Learning course (IBM, Coursera). The imagery appears to derive from Copernicus Sentinel-2 data; please check the applicable attribution terms before reusing it.

👩‍💻 Author

Ghaydaa-Aldirawi
