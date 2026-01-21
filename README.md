# Live Fashion-MNIST Classifier

<div align="justify">

This project implements a **real-time Fashion-MNIST classifier**, allowing users to draw clothing items while a trained model predicts their category every second. The application combines **PyTorch** for deep learning with **PyGame** for an interactive graphical interface. It serves as a hands-on demonstration of neural network inference and real-time prediction in Python.

## Project Overview

The application leverages the **Fashion-MNIST dataset**, which contains **70,000 grayscale images** (28×28 pixels) of fashion items categorized into **10 classes**. Users can draw items on a canvas, and the CNN model provides predictions **every second**.

---

## Components Overview

### Model
- **CNN (Convolutional Neural Network)** trained on Fashion-MNIST.
- Performs real-time inference on user drawings.

### Dataset
- **Fashion-MNIST**: 70,000 labeled grayscale images of clothing items.

### GUI
- **PyGame** interface with a drawing canvas.
- Greyscale slider for adjusting drawing intensity.
- Displays predicted class in real-time.

### Real-Time Loop
- Captures user input continuously.
- Preprocesses drawings for the CNN.
- Updates predictions every second.

---

## Features

- **Real-Time Predictions**: Draw a fashion item and receive predictions every second.
- **Interactive Interface**: GUI built with **PyGame**, featuring a drawing canvas and greyscale adjustment.
- **Pretrained Model**: Uses a CNN trained on Fashion-MNIST for immediate inference.

---

## Tech Stack

- **Python 3.6+**
- **PyTorch** – deep learning model training and inference
- **PyGame** – interactive GUI for drawing and predictions

---

## Requirements

- **Python 3.6 or higher**
- **PyTorch** (CPU or GPU version)
- **PyGame**

> Optional for GPU acceleration:
> - Install the CUDA-compatible PyTorch version.
> - See [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Installation Instructions

1. **Clone the repository**
```bash
git clone https://github.com/branebb/live-MNIST-fashion-classifier.git
```

2. **Navigate to the project folder**
```bash
cd live-MNIST-fashion-classifier
```

3. **Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

4. **Upgrade pip**
```bash
python -m pip install --upgrade pip
```

5. **Install dependencies**
```bash
pip install -r requirements.txt
```

6. **Run the application**
```bash
python main.py
```

---

## Usage Instructions

1. **Draw an item**: Use the canvas area to sketch a clothing item.  
2. **Prediction updates**: The CNN analyzes your drawing and updates the predicted category every second.  
3. **Adjust intensity**: Use the greyscale slider to change drawing intensity.

### Example Screenshots

#### Drawing and Prediction:
![ss1](https://github.com/user-attachments/assets/28b71ebc-67ec-4206-8313-2e618704f351)
![ss2](https://github.com/user-attachments/assets/b568138a-e8b9-4d05-87d4-c5492d56e6e3)



