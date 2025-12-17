#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#import os
import sys
import glob
import argparse
import time
import json
import shutil  

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from tqdm import tqdm  
import matplotlib.pyplot as plt

##############################################################
# Dataset
##############################################################
class ConditionalDataset(Dataset):
    """
    自定義資料集類別，用於條件生成器的訓練。
    """
    def __init__(self, root, manufacturers, transform=None, mode='train'):
        self.root = root
        self.manufacturers = manufacturers
        self.transform = transform
        self.mode = mode
        self.image_paths = []
        self.labels = []

        # 遍歷每個製造商，收集影像路徑和對應標籤
        for label, cls in enumerate(manufacturers):
            cls_dir = os.path.join(root, mode, cls)
            if not os.path.exists(cls_dir):
                print(f"警告: {cls_dir} 不存在，跳過此類別。")
                continue
            # 匹配多種影像格式
            imgs = glob.glob(os.path.join(cls_dir, '*.jpg')) + \
                   glob.glob(os.path.join(cls_dir, '*.jpeg')) + \
                   glob.glob(os.path.join(cls_dir, '*.png')) + \
                   glob.glob(os.path.join(cls_dir, '*.bmp')) + \
                   glob.glob(os.path.join(cls_dir, '*.gif'))

            for img_path in imgs:
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 若影像讀取失敗，回傳隨機黑色影像
            image = Image.new('RGB', (64,64), 'black')

        if self.transform:
            image = self.transform(image)

        # 確保影像尺寸為 (3,64,64)
        if image.size(1) != 64 or image.size(2) != 64:
            image = transforms.functional.resize(image, (64,64))

        return image, label

##############################################################
# Model
##############################################################
class Generator(nn.Module):
    """
    生成器模型，用於根據噪聲和標籤生成影像。
    """
    def __init__(self, noise_dim, label_dim, img_channels=3, feature_g=64):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)

        # 輸入維度：noise_dim + label_dim
        input_dim = noise_dim + label_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, feature_g * 8 * 4 * 4),
            nn.BatchNorm1d(feature_g * 8 * 4 * 4),
            nn.ReLU(True),

            nn.Unflatten(1, (feature_g * 8, 4, 4)),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # 將標籤轉為嵌入向量
        label_emb = self.label_embedding(labels)
        # 將噪聲與標籤嵌入向量串接
        gen_input = torch.cat((noise, label_emb), dim=1)  # [batch, noise_dim + label_dim]
        img = self.model(gen_input)
        return img

class Discriminator(nn.Module):
    """
    判別器模型，用於區分真實影像與生成影像。
    """
    def __init__(self, img_channels=3, label_dim=30, feature_d=64):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(label_dim, label_dim)

        # 將 label 轉為一熱編碼並擴展到與影像相同的空間尺寸
        self.img_channels = img_channels
        self.label_dim = label_dim

        # 使用 DCGAN-like 結構
        self.model = nn.Sequential(
            nn.Conv2d(img_channels + label_dim, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 將標籤轉為嵌入向量
        label_emb = self.label_embedding(labels)
        # 將嵌入向量擴展到與影像相同的空間尺寸
        label_emb = label_emb.unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.repeat(1, 1, img.size(2), img.size(3))  
        # 將影像與標籤嵌入向量串接
        d_in = torch.cat((img, label_emb), 1)  
        validity = self.model(d_in)
        return validity.view(-1,1)

##############################################################
# Evaluation Metrics
##############################################################
def calculate_metrics(real_images, fake_images):
    """
    計算 MSE、PSNR 和 SSIM 評估指標。
    """
    # 計算 MSE
    mse = np.mean((real_images - fake_images) ** 2)
    
    # 計算 PSNR
    psnr_val = np.mean([psnr(real, fake, data_range=1.0) for real, fake in zip(real_images, fake_images)])
    
    # 計算 SSIM
    ssim_val = np.mean([ssim(real.transpose(1, 2, 0), fake.transpose(1, 2, 0), data_range=1.0, channel_axis=2) 
                        for real, fake in zip(real_images, fake_images)])
    
    return mse, psnr_val, ssim_val

##############################################################
# Training function
##############################################################
def train_cgan(root, manufacturers, epochs=350, batch_size=64, lr=0.00011, noise_dim=100, sample_interval=100, device='cpu', output_dir='output_new_1000_epoch350_lr000011'):
    """
    訓練 Conditional GAN 的主函數。
    """
    # 創建影像保存目錄
    os.makedirs(os.path.join(output_dir, 'images_new_1000'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'best'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'final_models'), exist_ok=True)

    # 定義影像轉換
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # 強制縮放至64x64
        transforms.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
        transforms.RandomRotation(degrees=15),  # 隨機旋轉±15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 亮度、對比、飽和度
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # 將像素值從 [0,1] 轉換到 [-1,1]
    ])


    # 創建資料集與資料加載器
    dataset = ConditionalDataset(root, manufacturers, transform=transform, mode='train')
    if len(dataset) == 0:
        print("Dataset is empty. 請檢查資料路徑和類別資料夾是否正確。")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 定義生成器和判別器
    label_dim = len(manufacturers)
    generator = Generator(noise_dim, label_dim).to(device)
    discriminator = Discriminator(img_channels=3, label_dim=label_dim).to(device)

    # 初始化模型權重
    def weights_init(m):
        classname = m.__class__.__name__
        if 'Conv' in classname or 'Linear' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # 定義損失函數和優化器
    criterion = nn.BCELoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5,0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5,0.999))

    # 固定噪聲和標籤，用於生成一致性的樣本
    fixed_noise = torch.randn(label_dim, noise_dim, device=device)
    fixed_labels = torch.arange(label_dim, device=device)

    # 初始化記錄損失和評估指標的列表
    G_losses = []
    D_losses = []
    metrics = []

    # 初始化最佳 FID
    best_fid = float('inf')

    # 訓練開始時間
    start_time = time.time()

    # 開始訓練
    for epoch in range(epochs):
        epoch_start_time = time.time()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")

        for i, (imgs, labels) in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            batch_size_current = imgs.size(0)

            # 定義真實和偽造的目標
            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            # --------------------------------------------------------------------
            # 訓練生成器

            optimizer_G.zero_grad()

            # 生成偽造影像
            z = torch.randn(batch_size_current, noise_dim, device=device)
            gen_imgs = generator(z, labels)

            # 判別生成的影像
            validity = discriminator(gen_imgs, labels)
            g_loss = criterion(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # --------------------------------------------------------------------
            # 訓練判別器

            optimizer_D.zero_grad()

            # 判別真實影像
            real_pred = discriminator(imgs, labels)
            d_real_loss = criterion(real_pred, valid)

            # 判別偽造影像
            fake_pred = discriminator(gen_imgs.detach(), labels)
            d_fake_loss = criterion(fake_pred, fake)

            # 判別器總損失
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 記錄損失
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            # 更新進度條描述
            progress_bar.set_postfix({"D loss": d_loss.item(), "G loss": g_loss.item()})

            # 生成影像示例
            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                with torch.no_grad():
                    test_imgs = generator(fixed_noise, fixed_labels)
                test_imgs = (test_imgs + 1)/2  # 將範圍從 [-1,1] 轉換到 [0,1]
                # 保存至 images_new 資料夾，每隔100步保存一次
                save_image(test_imgs, os.path.join(output_dir, 'images_new_1000', f"{batches_done}.png"), nrow=label_dim, normalize=False)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{epochs}] completed in {epoch_duration:.2f} seconds.")

        # --------------------------------------------------------------------
        # 評估指標計算

        with torch.no_grad():
            generated_images = []
            for class_id, class_name in enumerate(manufacturers):
                noise = torch.randn(10, noise_dim, device=device)
                labels_sample = torch.full((10,), class_id, dtype=torch.long, device=device)
                gen_imgs = generator(noise, labels_sample).cpu().numpy()
                # 將生成的影像範圍從 [-1,1] 轉換到 [0,1]
                gen_imgs = (gen_imgs + 1) / 2
                generated_images.extend(gen_imgs)

            # 確保生成的影像數量為100
            generated_images = generated_images[:100]

            # 取每個類別10張真實影像進行評估
            real_sample = []
            for class_id, class_name in enumerate(manufacturers):
                # 獲取該類別的所有索引
                class_indices = [i for i, label in enumerate(dataset.labels) if label == class_id]
                if len(class_indices) < 10:
                    print(f"警告: 類別 {class_name} 的影像數量少於10張。")
                    sampled_indices = class_indices  # 使用所有可用影像
                else:
                    sampled_indices = np.random.choice(class_indices, 10, replace=False)
                real_sample.extend([dataset[i][0].numpy() for i in sampled_indices])

            # 確保真實影像數量與生成影像數量一致
            real_sample = real_sample[:len(generated_images)]
            fake_sample = generated_images[:len(real_sample)]  # 確保不超出範圍

            # 檢查影像數量
            if len(real_sample) == 0 or len(fake_sample) == 0:
                print("警告: 沒有足夠的影像來計算評估指標。")
                continue

            real_sample = np.array(real_sample)
            fake_sample = np.array(fake_sample)

            # 計算 MSE、PSNR、SSIM
            mse, psnr_val, ssim_val = calculate_metrics(real_sample, fake_sample)

            # 計算 FID
            # 將生成的影像保存到臨時目錄，直接保存所有影像，不分類別
            temp_fake_dir = os.path.join(output_dir, f'epoch_{epoch+1}', 'temp_fake')
            os.makedirs(temp_fake_dir, exist_ok=True)
            for idx, img in enumerate(fake_sample):
                img_tensor = torch.from_numpy(img)
                save_image(img_tensor, os.path.join(temp_fake_dir, f"fake_{idx+1}.png"))

            # 將真實影像保存到另一個臨時目錄，直接保存所有影像，不分類別
            temp_real_dir = os.path.join(output_dir, f'epoch_{epoch+1}', 'temp_real')
            os.makedirs(temp_real_dir, exist_ok=True)
            for idx, img in enumerate(real_sample):
                img_tensor = torch.from_numpy(img)
                save_image(img_tensor, os.path.join(temp_real_dir, f"real_{idx+1}.png"))

            # 計算 FID
            fid_val = fid_score.calculate_fid_given_paths([temp_real_dir, temp_fake_dir], batch_size=50, device=device, dims=2048)

            # 不刪除暫存目錄，保留供後續使用

            # 記錄評估指標
            metrics.append({
                'epoch': epoch + 1,
                'MSE': float(mse),
                'PSNR': float(psnr_val),
                'SSIM': float(ssim_val),
                'FID': float(fid_val)
            })

            print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | MSE: {mse:.4f} | PSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f} | FID: {fid_val:.4f}")

            # 檢查是否為最佳 FID，若是則保存最佳模型
            if fid_val < best_fid:
                best_fid = fid_val
                torch.save(generator.state_dict(), os.path.join(output_dir, 'best', 'best_G.pth'))
                torch.save(discriminator.state_dict(), os.path.join(output_dir, 'best', 'best_D.pth'))
                # 保存最佳 metrics
                with open(os.path.join(output_dir, 'best', 'best_metrics.json'), 'w') as f:
                    json.dump(metrics[-1], f, indent=4)
                print(f"最佳模型更新：Best FID = {best_fid:.4f}")

        # 訓練結束後保存最終模型
        torch.save(generator.state_dict(), os.path.join(output_dir, 'final_models', 'final_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, 'final_models', 'final_discriminator.pth'))
        print(f"最終模型已保存至 {os.path.join(output_dir, 'final_models')} 資料夾中。")

        # 保存所有評估指標到 JSON 檔案
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"評估指標已保存到 {metrics_file}")


        if metrics:
            epochs_list = [m['epoch'] for m in metrics]
            mse_list = [m['MSE'] for m in metrics]
            psnr_list = [m['PSNR'] for m in metrics]
            ssim_list = [m['SSIM'] for m in metrics]
            fid_list = [m['FID'] for m in metrics]

            plt.figure(figsize=(12,8))
            plt.subplot(2,2,1)
            plt.plot(epochs_list, mse_list, label='MSE', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Mean Squared Error (MSE)')
            plt.legend()

            plt.subplot(2,2,2)
            plt.plot(epochs_list, psnr_list, label='PSNR', color='orange', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.title('Peak Signal-to-Noise Ratio (PSNR)')
            plt.legend()

            plt.subplot(2,2,3)
            plt.plot(epochs_list, ssim_list, label='SSIM', color='green', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')
            plt.title('Structural Similarity Index (SSIM)')
            plt.legend()

            plt.subplot(2,2,4)
            plt.plot(epochs_list, fid_list, label='FID', color='red', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('FID')
            plt.title('Fréchet Inception Distance (FID)')
            plt.legend()

            plt.tight_layout()
            loss_curve_path = os.path.join(output_dir, 'metrics_curve.png')
            plt.savefig(loss_curve_path)
            plt.close()
            print(f"評估指標曲線已保存到 {loss_curve_path}")
        else:
            print("沒有評估指標可供繪製。")

##############################################################
# Main
##############################################################
def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description="Conditional GAN Training Script")
    parser.add_argument('--dataroot', type=str, required=True, help='根目錄的資料集路徑')
    parser.add_argument('--manufacturer_file', type=str, required=True, help='manufacturers.txt 文件的路徑')
    parser.add_argument('--output_dir', type=str, default='./output_new_3000', help='輸出目錄路徑')
    parser.add_argument('--epochs', type=int, default=100, help='訓練的總 epoch 數')
    parser.add_argument('--batch_size', type=int, default=64, help='訓練的批次大小')
    parser.add_argument('--lr', type=float, default=0.0002, help='初始學習率')
    parser.add_argument('--noise_dim', type=int, default=100, help='噪聲向量的維度')
    parser.add_argument('--sample_interval', type=int, default=100, help='生成影像的間隔步數')
    parser.add_argument('--use_cuda', action='store_true', help='是否使用 CUDA 加速')
    args = parser.parse_args()

    # 設定裝置
    device = torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    # 讀取 manufacturers.txt
    if not os.path.exists(args.manufacturer_file):
        print(f"錯誤: {args.manufacturer_file} 不存在。")
        sys.exit(1)
    
    with open(args.manufacturer_file, 'r') as f:
        manufacturers = [line.strip() for line in f.readlines()]
    
    if len(manufacturers) == 0:
        print("錯誤: manufacturers.txt 中沒有任何類別。")
        sys.exit(1)
    
    # 檢查訓練資料夾是否存在對應的類別資料夾
    missing = []
    for m in manufacturers:
        p = os.path.join(args.dataroot, 'train', m)
        if not os.path.isdir(p):
            missing.append(m)
    if missing:
        print(f"錯誤: 缺少以下類別的資料夾: {missing}")
        sys.exit(1)

    # 呼叫訓練函數
    train_cgan(
        root=args.dataroot,
        manufacturers=manufacturers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_dim=args.noise_dim,
        sample_interval=args.sample_interval,
        device=device,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
