# Military-Aircraft-Conditional-GAN-model-Combination1
【 Conditional GAN (cGAN) Image Generation Framework-Combination-1 】

This repository provides a **complete PyTorch implementation of a Conditional Generative Adversarial Network (cGAN)** for class-conditional image generation.

The framework is designed for **research, academic experiments, and data augmentation scenarios**, enabling the generation of realistic images conditioned on class labels.

This implementation was trained and validated using a **primarily developed and validated using the Military Aircraft Detection Dataset from Kaggle** and is suitable for tasks such as **dataset balancing, visual data augmentation, and generative modeling research**.

---

## 🚀 Features

* Conditional GAN (cGAN) architecture
* Label-conditioned image generation
* Configurable generator and discriminator
* Stable training with Adam optimizer
* Automatic loss tracking and visualization
* Periodic model checkpoint saving
* Generated image sampling during training

---

## 📁 Project Structure

```bash
.
├── Conditional_GAN_model_epoch350_lr0.00011.py   # Main training script
├── data/                                        # Training dataset
│   ├── class_0/
│   ├── class_1/
│   └── class_2/
├── outputs/                                     # Generated images
│   ├── samples_epoch_50/
│   ├── samples_epoch_100/
│   └── ...
└── checkpoints/                                 # Saved model weights
    ├── generator_epoch_100.pth
    ├── discriminator_epoch_100.pth
    └── ...
```

---

## 🧠 Model Architecture

### Generator

* Input:

  * Random noise vector (latent vector `z`)
  * Class label (embedded and concatenated)
* Output:

  * Synthetic image conditioned on the given label

### Discriminator

* Input:

  * Image (real or generated)
  * Corresponding class label
* Output:

  * Probability indicating whether the image is real or fake

Both generator and discriminator incorporate **label embeddings** to ensure class-conditional learning.

---

## 🔢 Training Configuration

| Parameter        | Value                      |
| ---------------- | -------------------------- |
| Epochs           | 350                        |
| Learning Rate    | 0.00011                    |
| Optimizer        | Adam                       |
| Loss Function    | Binary Cross Entropy (BCE) |
| Latent Dimension | Configurable in code       |

---

## 🖼️ Dataset Format
This project is **primarily designed and evaluated using the following Kaggle dataset**:

**Military Aircraft Detection Dataset (Kaggle)**  
<img width="2203" height="371" alt="image" src="https://github.com/user-attachments/assets/6361547c-3d02-4f6f-a2f1-88c90a11e3cf" />

🔗 https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data


The dataset should be organized in a **class-based directory structure**:

```bash
data/
├── class_0/
│   ├── img001.jpg
│   ├── img002.jpg
├── class_1/
│   ├── img001.jpg
└── class_2/
```

Each subdirectory represents one class label used for conditional generation.

---

## 🔄 Training Process

1. Sample random noise vectors and corresponding class labels
2. Generator produces fake images conditioned on labels
3. Discriminator evaluates real and fake images with labels
4. Update discriminator and generator alternately
5. Save generated image samples periodically
6. Save model checkpoints at predefined epochs

---

## 📈 Outputs & Visualization

During training, the following outputs are automatically generated:

* Generated image grids per epoch
* Generator and discriminator loss curves
* Periodic visual inspection samples

Example output structure:

```bash
outputs/samples_epoch_200/
├── class_0_sample.png
├── class_1_sample.png
└── class_2_sample.png
```

---

## 💾 Model Checkpoints

Saved at regular intervals:

```bash
checkpoints/
├── generator_epoch_50.pth
├── discriminator_epoch_50.pth
├── ...
```

These checkpoints can be used for:

* Resuming training
* Image generation only (inference)
* Data augmentation pipelines


---

## 📦 Requirements

```txt
torch
torchvision
numpy
matplotlib
tqdm
```

Recommended:

* Python ≥ 3.8
* CUDA-enabled GPU

---

## ⚠️ Notes

* GAN training can be unstable; monitoring generated samples is recommended
* Hyperparameters such as learning rate and latent dimension significantly affect performance
* Label imbalance in the dataset may affect generation quality

---

## 📜 License

This project is intended for **research and academic use only**.

Please evaluate generated data carefully before using it in downstream tasks.

---

## ✉️ Experimental Results
<img width="1465" height="589" alt="image" src="https://github.com/user-attachments/assets/843d5599-588a-449b-9a52-ef1dfe35d065" />

<img width="1706" height="99" alt="image" src="https://github.com/user-attachments/assets/3269545e-c97b-4645-a744-8124d3e0b204" />

⭐ If you find this project useful, consider starring the repository on GitHub!
