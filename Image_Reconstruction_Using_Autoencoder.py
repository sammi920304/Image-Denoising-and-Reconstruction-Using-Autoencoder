import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from torchsummary import summary
# 自訂 Dataset：用於無標籤圖片（適合 Autoencoder）
class UnlabeledImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
# 圖片轉換流程
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 統一尺寸
    transforms.ToTensor(),          # [0, 255] → [0.0, 1.0]
])

# 圖片資料夾路徑
img_dir = ""
# 測試集資料夾路徑
test_img_dir = ""


print(os.listdir(img_dir))
# 建立 Dataset 與 DataLoader
train_dataset = UnlabeledImageDataset(img_dir=img_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# 建立測試集 Dataset 與 DataLoader
test_dataset = UnlabeledImageDataset(img_dir=test_img_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=288):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # [B, 32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, bottleneck_dim, 2, stride=2, padding=0), # [B, bottleneck_dim, 16, 16]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim, 64, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# 選擇設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
summary(model, input_size=(3, 128, 128))

print("CUDA ", torch.cuda.is_available())
print("目前使用的裝置：", device)
print("目前 GPU 名稱：", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "無 GPU")

#設定損失函數與優化器
criterion_mse = nn.MSELoss()
#criterion_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
#定義訓練與驗證流程
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_psnr = 0
    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr(output, batch).item()
    return total_loss / len(dataloader), total_psnr / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)

            total_loss += loss.item()
            total_psnr += psnr(output, batch).item()
    return total_loss / len(dataloader), total_psnr / len(dataloader)

# 訓練主程式（紀錄 loss 與 PSNR）
num_epochs = 50
train_losses, val_losses = [], []
train_psnrs, val_psnrs = [], []

val_dataloader = test_dataloader

for epoch in range(num_epochs):
    train_loss, train_psnr_val = train(model, train_dataloader, optimizer, criterion_mse, device)
    val_loss, val_psnr_val = validate(model, val_dataloader, criterion_mse, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_psnrs.append(train_psnr_val)
    val_psnrs.append(val_psnr_val)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, PSNR: {train_psnr_val:.2f}, Test Loss: {val_loss:.4f}, PSNR: {val_psnr_val:.2f}")
'''''
# 繪製 PSNR 曲線
plt.figure(figsize=(10, 5))
plt.plot(train_psnrs, label=f"Train PSNR (Last: {train_psnrs[-1]:.2f} dB)")
plt.plot(val_psnrs, label=f"Test PSNR (Last: {val_psnrs[-1]:.2f} dB)")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Curve")
plt.legend()
plt.grid(True)
plt.show()
'''''
# 繪製圖表（損失 + PSNR）
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

plt.plot(range(1, num_epochs+1), train_losses, label=f"Train (Last: {train_losses[-1]:.4f})")
plt.plot(range(1, num_epochs+1), val_losses, label=f"Test (Last: {val_losses[-1]:.4f})")

plt.title("Loss Curve")
# 設定 x 軸和 y 軸刻度間隔
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(0, 51, 10))
plt.yticks(np.arange(0, 0.02, 0.005))
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_psnrs, label=f"Train (Last: {train_psnrs[-1]:.2f})")
plt.plot(range(1, num_epochs+1), val_psnrs, label=f"Test (Last: {val_psnrs[-1]:.2f})")
plt.title("PSNR Curve")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
# 設定 x 軸和 y 軸刻度間隔
plt.xticks(np.arange(0, 51, 10))
plt.yticks(np.arange(0, 41, 10))
plt.legend()
plt.show()

# 顯示測試集中的兩張圖片與其重建結果
model.eval()
sample_indices = [0, 1]  # 指定測試集中要觀察的圖片 index
images = [test_dataset[i] for i in sample_indices]  # 取得兩張圖片
images_tensor = torch.stack(images).to(device)      # [2, 3, 128, 128]

with torch.no_grad():
    reconstructions = model(images_tensor)          # 重建圖片

# 將 tensor 轉為 NumPy，調整為 [H, W, C]
def tensor_to_numpy(img_tensor):
    return img_tensor.cpu().permute(1, 2, 0).numpy()

# 顯示原始圖與重建圖
plt.figure(figsize=(8, 4))
for i in range(2):
    # 原圖
    plt.subplot(2, 2, i * 2 + 1)
    plt.imshow(tensor_to_numpy(images_tensor[i]))
    plt.title(f"Original #{sample_indices[i]}")
    plt.axis("off")

    # 重建圖
    plt.subplot(2, 2, i * 2 + 2)
    plt.imshow(tensor_to_numpy(reconstructions[i]))
    plt.title(f"Reconstructed #{sample_indices[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

