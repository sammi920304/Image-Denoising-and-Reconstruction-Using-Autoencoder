import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
from torchsummary import summary

class DenoisingImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_filenames = sorted(os.listdir(noisy_dir))
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, idx):
        noisy_name = self.noisy_filenames[idx]
        clean_name = noisy_name.replace("noisy", "clear")

        noisy_path = os.path.join(self.noisy_dir, noisy_name)
        clean_path = os.path.join(self.clean_dir, clean_name)

        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        noisy = self.transform(noisy_img)
        clean = self.transform(clean_img)

        return noisy, clean  # noisy 當輸入，clean 當 ground truth

# 自訂 Dataset：用於無標籤圖片（適合 Autoencoder）

# 設定資料夾路徑
train_dataset = DenoisingImageDataset(
    noisy_dir=r"",
    clean_dir=r"",
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = DenoisingImageDataset(
    noisy_dir=r"",
    clean_dir=r"",
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

activation = nn.ELU()

class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super(ConvAutoencoder, self).__init__()
        self.act = activation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # [B, 32, 64, 64]
            self.act,
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B, 64, 32, 32]
            self.act,
            nn.Conv2d(64, bottleneck_dim, 2, stride=2, padding=0), # [B, bottleneck_dim, 16, 16]
            self.act
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_dim, 64, 2, stride=2, padding=0),
            self.act,
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            self.act,
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# 選擇設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
summary(model, input_size=(3, 128, 128))
#設定損失函數與優化器
criterion_mse = nn.MSELoss()

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
    for noisy, clean in tqdm(dataloader, desc="Training", leave=False):
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = criterion(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr(output, clean).item()
    return total_loss / len(dataloader), total_psnr / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Validating", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)

            total_loss += loss.item()
            total_psnr += psnr(output, clean).item()
    return total_loss / len(dataloader), total_psnr / len(dataloader)

# 訓練主程式（紀錄 loss 與 PSNR）
num_epochs = 80
train_losses, val_losses = [], []
train_psnrs, val_psnrs = [], []

val_dataloader = test_loader  


for epoch in range(num_epochs):
    train_loss, train_psnr_val = train(model, train_dataloader, optimizer, criterion_mse, device)
    val_loss, val_psnr_val = validate(model, val_dataloader, criterion_mse, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_psnrs.append(train_psnr_val)
    val_psnrs.append(val_psnr_val)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, PSNR: {train_psnr_val:.2f}, Test Loss: {val_loss:.4f}, PSNR: {val_psnr_val:.2f}")

# 繪製圖表（損失 + PSNR）
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

plt.plot(range(1, num_epochs+1), train_losses, label=f"Train (Last: {train_losses[-1]:.4f})")
plt.plot(range(1, num_epochs+1), val_losses, label=f"Test (Last: {val_losses[-1]:.4f})")

plt.title("Loss Curve")
# 設定 x 軸和 y 軸刻度間隔
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 0.02, 0.005))
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_psnrs, label=f"Train (Last: {train_psnrs[-1]:.2f})")
plt.plot(range(1, num_epochs+1), val_psnrs, label=f"Test (Last: {val_psnrs[-1]:.2f})")
plt.title("PSNR Curve")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
# 設定 x 軸和 y 軸刻度間隔
plt.xticks(np.arange(0, 81, 10))
plt.yticks(np.arange(0, 41, 10))
plt.legend()
plt.show()

# 顯示測試集中的圖片與其重建結果
model.eval()
sample_indices = [0,1]  
noisy_imgs = []
clean_imgs = []
for i in sample_indices:
    noisy, clean = test_dataset[i]
    noisy_imgs.append(noisy)
    clean_imgs.append(clean)
noisy_tensor = torch.stack(noisy_imgs).to(device)
clean_tensor = torch.stack(clean_imgs).to(device)


with torch.no_grad():
    reconstructions = model(noisy_tensor)
       # 重建圖片

# 將 tensor 轉為 NumPy，調整為 [H, W, C]
def tensor_to_numpy(img_tensor):
    return img_tensor.cpu().permute(1, 2, 0).numpy()

# 顯示原始圖與重建圖
plt.figure(figsize=(10, 6))
for i in range(2):
    # Noisy image
    plt.subplot(2, 3, i * 3 + 1)
    plt.imshow(tensor_to_numpy(noisy_tensor[i]))
    plt.title(f"Noisy #{sample_indices[i]}")
    plt.axis("off")

    # Reconstructed image
    plt.subplot(2, 3, i * 3 + 2)
    plt.imshow(tensor_to_numpy(reconstructions[i]))
    plt.title(f"Reconstructed #{sample_indices[i]}")
    plt.axis("off")

    # Clean image
    plt.subplot(2, 3, i * 3 + 3)
    plt.imshow(tensor_to_numpy(clean_tensor[i]))
    plt.title(f"Clean #{sample_indices[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()