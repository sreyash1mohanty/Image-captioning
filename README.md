# Image Captioning using Deep Learning

## 📌 Overview
This project implements an **Image Captioning Model** using **Keras** and **Transfer Learning**. The model generates captions for images by combining **Convolutional Neural Networks (CNNs)** for feature extraction and **Recurrent Neural Networks (RNNs)** with **LSTM** for sequence generation.The model is trained on Flickr8k Dataset.

## 🚀 Key Features
- Uses **ResNet50** (pretrained) for image feature extraction (2048 features per image).
- Applies **Global Average Pooling** to modify the ResNet50 output layer.
- Utilizes **GloVe 6B50d embeddings** for word representation.
- Implements an **LSTM-based decoder** for sequential caption generation.
- Uses **Custom Data Generator** to efficiently preprocess captions and images.
- Trained on a dataset with a vocabulary size of **1848 words**.
- Implements **Dropout Regularization** to prevent overfitting.


## 🏗️ Model Architecture
The model consists of two main parts:

### 1️⃣ Feature Extractor (CNN - ResNet50)
- **Input:** Image
- **Output:** 2048-dimensional feature vector
- **Modifications:** Replaced final layer with Global Average Pooling

### 2️⃣ Caption Generator (RNN - LSTM)
- **Input:** Tokenized captions
- **Embedding Layer:** Uses pre-trained **GloVe 6B50d** word embeddings
- **LSTM Layer:** Generates sequential words based on input captions and image features
- **Fully Connected Layers:** Dense layers for final word prediction

### 🔹 Neural Network Layers
```python
# Image Feature generated from ResNet50 are passed here
input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256, activation='relu')(inp_img1)

# Caption Processing
input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)

# Decoder
decoder1 = add([inp_img2, inp_cap3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Model
model = Model(inputs=[input_img_features, input_captions], outputs=outputs)
```

## 📊 Data Preprocessing and Trasnfer learning
### 🔹 Captions
- Cleaned captions by removing punctuation and special characters.
- Tokenized captions and built a vocabulary of **1848 unique words**.
- Applied **GloVe 6B50d word embeddings** to map words into vector space.

### 🔹 Images
- Resized all images to the required input size for **ResNet50**.
- Extracted **2048-dimensional feature vectors** and using **Resnet50 base **.
- Stored preprocessed image features for efficient training.

## 🏋️ Training
### 🔹 Loss Function
- The model is trained using **Categorical Cross-Entropy Loss**.

### 🔹 Optimizer
- Used **Adam Optimizer** with a learning rate of `0.001`.

### 🔹 Batch Processing
- Used a **Custom Data Generator** to efficiently process large datasets in batches.



## ⚡ Future Improvements
- Train on **larger datasets** to improve generalization.


### 🔹 Required Libraries
- `TensorFlow / Keras`
- `NLTK`
- `NumPy`
- `Pandas`
- `Matplotlib`

