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






