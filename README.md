# Live MNIST Fashion Classifier
<div align="justify">

## Project Overview
This project leverages the **Fashion-MNIST dataset**, which consists of **70,000 grayscale images** (28×28 pixels) of Zalando's fashion articles categorized into **10 classes**. The application allows users to **draw fashion items**, with the model providing real-time predictions **every second**.



## Features
- **Real-Time Predictions**: Draw a fashion item and receive instant predictions every second.
- **Pretrained Model**: Uses a **Convolutional Neural Network (CNN)** trained on the Fashion-MNIST dataset.
- **Interactive Interface**: GUI built with **PyGame**, featuring a drawing canvas and a greyscale slider.  

## Tech Stack
- **PyTorch**: For model training, inference, and deep learning tasks.
- **PyGame**: Handles the GUI for drawing and displaying real-time predictions.

## Requirements
- **Python 3.6 or higher** - Ensure you have the correct version installed before proceeding.


## Installation Instructions
**Clone this repository**:
### 
    git clone https://github.com/branebb/live-MNIST-fashion-classifier.git

**Navigate to the project directory**:
###
    cd live-MNIST-fashion-classifier

**Create a virtual environment**:
###
    python -m venv <venv-name>

**Activate the virtual environment (Windows)**:
###
    <venv-name>\Scripts\activate

**Upgrade pip (Recommended)**:
###
    python.exe -m pip install --upgrade pip

**[Optional] Install CUDA for GPU Acceleration**:
    - If you want to run the application on your **GPU with CUDA support**, you need to install the appropriate version of PyTorch with CUDA.
    - Note: The installation is **~2.5GB**. If you prefer a **faster setup**, you can **skip this step** and run the application on your CPU instead.
    - **For CUDA installation instructions, visit the [PyTorch website](https://pytorch.org/get-started/locally/)**.

**Install the required dependencies**:
###
    pip install -r requirements.txt


**Run the main part of the project**:
###
    python main.py

## Usage Instructions
Once installed and running, a window will open, allowing you to draw clothing items. The model will predict the category in real-time every second.

### Example Screenshots

Below are some example screenshots of the application in action:
#### Drawing and Prediction:
![ss1](https://github.com/user-attachments/assets/28b71ebc-67ec-4206-8313-2e618704f351)

![ss2](https://github.com/user-attachments/assets/b568138a-e8b9-4d05-87d4-c5492d56e6e3)


### How It Works:
1. **Draw an item**: Use the canvas area to draw a clothing item.
2. **Prediction updates every second**: The model analyzes the drawing and displays the predicted category.

</div>
