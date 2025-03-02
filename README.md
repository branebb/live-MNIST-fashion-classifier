# Live MNIST Fashion Classifier
<div align="justify">

## Project Overview
This project leverages the **Fashion-MNIST dataset**, consisting of 70,000 grayscale images of Zalando's fashion articles, each 28x28 pixels, categorized into 10 classes. The application allows users to draw fashion items, with the model providing real-time predictions **every second**.

## Features
- **Real-Time Predictions**: Draw a fashion item and receive instant predictions every second.
- **Pretrained Model**: Utilizes a Convolutional Neural Network (CNN) trained on the Fashion-MNIST dataset.
- **Interactive Interface**: Graphical user interface (GUI) built with PyGame, featuring a drawing canvas for input and slider for changing greyscale values.  

## Tech Stack
- **PyTorch**: All model-related tasks, including training and inference.
- **PyGame**: Handling the GUI for drawing and displaying predictions in real time.

## Requirements
This project requires **Python 3.6** or higher. Ensure you have the correct Python version installed before proceeding with the installation.

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

**Before installing anything, upgrade ```pip``` to the latest version**:
###
    python.exe -m pip install --upgrade pip

**[Optional] Install CUDA for GPU Acceleration**:

If you want to run the application on your **GPU with CUDA support**, you need to install the appropriate version of PyTorch with CUDA. However, be aware that the installation is **around 2.5GB**. If you prefer a **faster setup**, **you can skip** this step and run the application on your CPU instead. For installation instructions, visit the [PyTorch website](https://pytorch.org/get-started/locally/).

**Install the required dependencies**:
###
    pip install -r requirements.txt


**Run the main part of the project**:
###
    python main.py

## Usage Instructions
Once you've completed the installation and are running the application, you should see a window that allows you to draw clothing items. The model will predict the category of the clothing in real-time as you draw.

### Example Screenshots

Below are some example screenshots of the application in action:
#### Drawing and Prediction:
![ss1](https://github.com/user-attachments/assets/28b71ebc-67ec-4206-8313-2e618704f351)

![ss2](https://github.com/user-attachments/assets/b568138a-e8b9-4d05-87d4-c5492d56e6e3)


### How It Works:
- **Draw an item**: Use the canvas area to draw the clothing item.
- **Prediction**: Every 1 second, the model will predict the clothing category and display the result.

This visual feedback will help ensure you understand how the application is meant to function.

</div>
