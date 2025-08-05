import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import datetime
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import matplotlib.pyplot as plt


class DarkFieldDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir, image_size=(256, 256)):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.image_size = image_size
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
        assert len(self.clean_files) == len(self.noisy_files), "Mismatch in dataset size"
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert("L")
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert("L")
        return self.transform(noisy_img), self.transform(clean_img)


def show_sample_batch(loader):
    noisy, clean = next(iter(loader))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(noisy[i, 0], cmap='gray')
        axes[1, i].imshow(clean[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Noisy")
    axes[1, 0].set_ylabel("Clean")
    plt.tight_layout()
    plt.show()


# ==== MODELS ====
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()

        for feature in features:
            self.encoder.append(UNetBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(UNetBlock(feature * 2, feature))

        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(len(self.decoder)):
            x = self.upconv[idx](x)
            x = torch.cat((x, skips[idx]), dim=1)
            x = self.decoder[idx](x)

        x = self.final(x)
        # Add sigmoid activation to ensure output is in [0, 1] range
        x = torch.sigmoid(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)  # Fixed: multiply by in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_ch = in_chans

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.head(x)
        # Fixed: proper reshaping for reconstruction
        patches_per_side = self.img_size // self.patch_size
        x = x.view(B, patches_per_side, patches_per_side, self.patch_size, self.patch_size, self.out_ch)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_ch, self.img_size, self.img_size)
        # Add sigmoid activation to ensure output is in [0, 1] range
        x = torch.sigmoid(x)
        return x


# ==== LOSSES ====
def get_loss_fn(name="l1"):
    if name == "l1":
        return nn.L1Loss()
    elif name == "l2":
        return nn.MSELoss()
    elif name == "ssim":
        return lambda x, y: 1 - ssim(x, y, data_range=1.0, size_average=True)
    elif name == "combined":
        l1 = nn.L1Loss()
        return lambda x, y: l1(x, y) + (1 - ssim(x, y, data_range=1.0, size_average=True))
    else:
        raise NotImplementedError(f"Loss '{name}' is not supported.")


# ==== DATASET ====
class PairedImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=256):
        self.noisy_files = sorted(glob(os.path.join(noisy_dir, "*.png")))
        self.clean_files = sorted(glob(os.path.join(clean_dir, "*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_files[idx]).convert("L")
        clean = Image.open(self.clean_files[idx]).convert("L")
        return self.transform(noisy), self.transform(clean)


# ==== TRAINING ====
def train_model(model, dataloader, loss_fn, optimizer, device, epochs=10, save_path="model.pth", resume=False):
    model.to(device)
    if resume and os.path.exists(f"{save_path}/trained_model.pth"):
        print(f"Loading existing model from {save_path}")
        model.load_state_dict(torch.load(f"{save_path}/trained_model.pth", map_location=device))

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = loss_fn(output, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/trained_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)

    model_name = "darkfield"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_path}/{model_name}_epoch{epochs}_loss_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------------
# Run Inference on Test Set
# -------------------------------
def run_inference(model, test_loader, save_dir, device):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device)
            output = model(noisy)
            for i in range(noisy.shape[0]):
                in_img = noisy[i, 0].cpu().numpy()
                gt_img = clean[i, 0].cpu().numpy()
                out_img = output[i, 0].cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(in_img, cmap='gray')
                ax[1].imshow(gt_img, cmap='gray')
                ax[2].imshow(out_img, cmap='gray')
                for a in ax: 
                    a.axis('off')
                ax[0].set_title("Noisy Input")
                ax[1].set_title("Ground Truth")
                ax[2].set_title("Model Output")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"test_result_{idx}_{i}.png"))
                plt.close()


# ==== RUN PIPELINE ====
def run_training_pipeline(model_type="unet", loss_type="l1", data_path="darkfield_lung_sim",
                          img_size=256, batch_size=4, lr=1e-4, epochs=10,
                          resume=False, save_model_path="trained_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_type == "unet":
        model = UNet(in_channels=1, out_channels=1)
    elif model_type == "transformer":
        model = SimpleViT(img_size=img_size, patch_size=16, in_chans=1)
    else:
        raise ValueError("Invalid model type.")

    model.to(device)  # Move model to device

    # Fixed path construction
    combined_clean = f"{data_path}/clean"
    combined_noisy = f"{data_path}/noisy"

    dataset = DarkFieldDataset(combined_clean, combined_noisy, image_size=(img_size, img_size))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Use provided batch_size

    # Visualize batch
    show_sample_batch(train_loader)

    loss_fn = get_loss_fn(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, loss_fn, optimizer, device, epochs, save_path=save_model_path, resume=resume)


# Try to import piq for advanced SSIM
try:
    import piq
    has_ssim = True
except ImportError:
    print("⚠️ SSIM loss requires `piq`. Run `pip install piq` if needed.")
    has_ssim = False


# -------------------------------
# Configuration
# -------------------------------
config = {
    "model_type": "transformer",  # 'unet' or 'transformer'
    "image_size": (256, 256),
    "batch_size": 8,
    "epochs": 5,
    "learning_rate": 1e-4,
    "loss_type": "combined",  # 'l1', 'l2', 'ssim', 'combined'
    "ssim_weight": 0.84,  # used for combined loss
    "save_path": "e:/NingWang/All/ML-Sim/model/darkfield/best_model.pth",
    "data_path": "e:/NingWang/All/ML-Sim/darkfield_dataset"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load datasets
# -------------------------------
parent_path = f"{config['data_path']}/combined"
print(parent_path)

# -------------------------------
# Data Split + Test Inference
# -------------------------------
all_data = PairedImageDataset(f"{parent_path}/noisy", f"{parent_path}/clean", 256)    
train_dataset, test_dataset = train_test_split(all_data, test_size=0.2, random_state=42)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# -------------------------------
# Model Setup
# -------------------------------
if config["model_type"] == "unet":
    model = UNet().to(device)
elif config["model_type"] == "transformer":
    model = SimpleViT(img_size=256, patch_size=16, in_chans=1).to(device)  # Fixed: move to device
else:
    raise ValueError("Invalid model type")

# Only load existing model if it matches the current model type
if os.path.exists(config["save_path"]):
    try:
        print(f"📦 Loading existing model from {config['save_path']}...")
        state_dict = torch.load(config["save_path"], map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully!")
    except RuntimeError as e:
        print(f"⚠️ Cannot load saved model - architecture mismatch: {str(e)}")
        print("🔄 Starting training from scratch with new architecture...")
        # Optionally, you can rename/backup the old model file
        backup_path = config["save_path"].replace(".pth", "_backup_unet.pth")
        os.rename(config["save_path"], backup_path)
        print(f"📁 Old model backed up to: {backup_path}")


# -------------------------------
# Loss Function Setup
# -------------------------------
def get_loss_function(name):
    if name == "l1":
        return nn.L1Loss()
    elif name in ["l2", "mse"]:
        return nn.MSELoss()
    elif name == "ssim":
        if not has_ssim:
            raise ImportError("SSIM loss requires `piq`. Install with `pip install piq`.")
        return lambda pred, target: 1.0 - piq.ssim(pred, target, data_range=1.0)
    elif name == "combined":
        if not has_ssim:
            print("⚠️ SSIM not available, falling back to L1 loss")
            return nn.L1Loss()
        alpha = config.get("ssim_weight", 0.84)
        mse = nn.MSELoss()
        return lambda pred, target: (1 - alpha) * mse(pred, target) + alpha * (1.0 - piq.ssim(pred, target, data_range=1.0))
    else:
        raise ValueError("Unsupported loss type")

# Add input validation function
def clamp_tensor(tensor, min_val=0.0, max_val=1.0):
    """Clamp tensor values to ensure they're in valid range for SSIM"""
    return torch.clamp(tensor, min_val, max_val)

criterion = get_loss_function(config["loss_type"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# -------------------------------
# Training Loop
# -------------------------------
best_val_loss = float("inf")
for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        
        # Ensure output is in valid range for SSIM if using combined loss
        if config["loss_type"] in ["ssim", "combined"]:
            output = clamp_tensor(output, 0.0, 1.0)
            clean = clamp_tensor(clean, 0.0, 1.0)
            
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            
            # Ensure output is in valid range for SSIM if using combined loss
            if config["loss_type"] in ["ssim", "combined"]:
                output = clamp_tensor(output, 0.0, 1.0)
                clean = clamp_tensor(clean, 0.0, 1.0)
                
            val_loss += criterion(output, clean).item()

    avg_train = running_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        print("✅ Saving best model...")
        best_val_loss = avg_val
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(config["save_path"])
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), config["save_path"])


# -------------------------------
# Test Inference (Fixed version)
# -------------------------------
def run_inference_fixed(model, test_loader, save_dir="test_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device)
            output = model(noisy)
            for i in range(noisy.size(0)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(noisy[i, 0].cpu(), cmap='gray')
                axs[1].imshow(clean[i, 0].cpu(), cmap='gray')
                axs[2].imshow(output[i, 0].cpu(), cmap='gray')
                axs[0].set_title("Noisy Input")
                axs[1].set_title("Ground Truth")
                axs[2].set_title("Model Output")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"test_result_{idx}_{i}.png"))
                plt.close()


# -------------------------------
# Run Testing
# -------------------------------
if os.path.exists(config["save_path"]):
    try:
        model.load_state_dict(torch.load(config["save_path"], map_location=device, weights_only=True))
        print("✅ Model loaded for testing!")
    except RuntimeError as e:
        print(f"⚠️ Cannot load saved model for testing - using current model state")
        
run_inference_fixed(model, test_loader)


# Example usage (uncomment to execute)
"""
run_training_pipeline(
    model_type="transformer",                   
    loss_type="combined",               
    data_path="e:/NingWang/All/ML-Sim/darkfield_dataset/combined",
    epochs=5,
    save_model_path="e:/NingWang/All/ML-Sim/model/darkfield_lung_sim_transformer",
    resume=False
)
"""





exit(0)














import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import datetime
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
#from dataset_loader import DarkFieldDataset
#from model_unet import UNet
#from model_transformer import TransformerModel
from torchvision.utils import save_image
import matplotlib.pyplot as plt



class DarkFieldDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noisy_dir, image_size=(256, 256)):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.image_size = image_size
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.png')])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
        assert len(self.clean_files) == len(self.noisy_files), "Mismatch in dataset size"
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.clean_files)
    def __getitem__(self, idx):
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_files[idx])).convert("L")
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_files[idx])).convert("L")
        return self.transform(noisy_img), self.transform(clean_img)

def show_sample_batch(loader):
    noisy, clean = next(iter(loader))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i in range(8):
        axes[0, i].imshow(noisy[i, 0], cmap='gray')
        axes[1, i].imshow(clean[i, 0], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Noisy")
    axes[1, 0].set_ylabel("Clean")
    plt.tight_layout()
    plt.show()


# ==== MODELS ====
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()

        for feature in features:
            self.encoder.append(UNetBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.upconv.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(UNetBlock(feature * 2, feature))

        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(len(self.decoder)):
            x = self.upconv[idx](x)
            x = torch.cat((x, skips[idx]), dim=1)
            x = self.decoder[idx](x)

        return self.final(x)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, patch_size * patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_ch = in_chans

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(B, self.img_size // self.patch_size, self.img_size // self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, self.img_size, self.img_size)
        return x

# ==== LOSSES ====
def get_loss_fn(name="l1"):
    if name == "l1":
        return nn.L1Loss()
    elif name == "l2":
        return nn.MSELoss()
    elif name == "ssim":
        return lambda x, y: 1 - ssim(x, y, data_range=1.0, size_average=True)
    elif name == "combined":
        l1 = nn.L1Loss()
        return lambda x, y: l1(x, y) + (1 - ssim(x, y, data_range=1.0, size_average=True))
    else:
        raise NotImplementedError(f"Loss '{name}' is not supported.")

# ==== DATASET ====
class PairedImageDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=256):
        self.noisy_files = sorted(glob(os.path.join(noisy_dir, "*.png")))
        self.clean_files = sorted(glob(os.path.join(clean_dir, "*.png")))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = Image.open(self.noisy_files[idx]).convert("L")
        clean = Image.open(self.clean_files[idx]).convert("L")
        return self.transform(noisy), self.transform(clean)

# ==== TRAINING ====
def train_model(model, dataloader, loss_fn, optimizer, device, epochs=10, save_path="model.pth", resume=False):
    model.to(device)
    if resume and os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        model.load_state_dict(torch.load(f"{save_path}/trained_model.pth", map_location=device))

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = loss_fn(output, clean)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"{save_path}/trained_model.pth")

    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)



    model_name = "darkfield"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_path}/{model_name}_epoch{epochs}_loss_{timestamp}.png"
    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # High-res and neat margins


    plt.show()



# -------------------------------
# Run Inference on Test Set
# -------------------------------
def run_inference(model, test_loader, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device)
            output = model(noisy)
            for i in range(noisy.shape[0]):
                in_img = noisy[i, 0].cpu().numpy()
                gt_img = clean[i, 0].cpu().numpy()
                out_img = output[i, 0].cpu().numpy()
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(in_img, cmap='gray')
                ax[1].imshow(gt_img, cmap='gray')
                ax[2].imshow(out_img, cmap='gray')
                for a in ax: a.axis('off')
                ax[0].set_title("Noisy Input")
                ax[1].set_title("Ground Truth")
                ax[2].set_title("Model Output")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"test_result_{idx}_{i}.png"))
                plt.close()



# ==== RUN PIPELINE ====
def run_training_pipeline(model_type="unet", loss_type="l1", data_path="darkfield_lung_sim",
                          img_size=256, batch_size=4, lr=1e-4, epochs=10,
                          resume=False, save_model_path="trained_model.pth"):

    if model_type == "unet":
        model = UNet(in_channels=1, out_channels=1)
    elif model_type == "transformer":
        model = SimpleViT(img_size=img_size, patch_size=16, in_chans=1)
    else:
        raise ValueError("Invalid model type.")


    # 🔁 Replace this with your actual path
    #combined_clean = "e:/NingWang/All/ML-Sim/darkfield_dataset/combined/clean"
    #combined_noisy = "e:/NingWang/All/ML-Sim/darkfield_dataset/combined/noisy"
    combined_clean = f"{data_path}/clean"
     
    combined_noisy = f"{data_path}/noisy"

    dataset = DarkFieldDataset(combined_clean, combined_noisy, image_size=(img_size, img_size))
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Visualize batch
    show_sample_batch(train_loader)




    loss_fn = get_loss_fn(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, loss_fn, optimizer, device, epochs, save_path=save_model_path, resume=resume)




try:
    import piq
    has_ssim = True
except ImportError:
    print("⚠️ SSIM loss requires `piq`. Run `pip install piq` if needed.")
    has_ssim = False

# -------------------------------
# Configuration
# -------------------------------
config = {
    "model_type": "transformer",  # 'unet' or 'transformer'
    "image_size": (256, 256),
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 1e-4,
    "loss_type": "combined",  # 'l1', 'l2', 'ssim', 'combined'
    "ssim_weight": 0.84,  # used for combined loss
    "save_path": "e:NingWang/All/ML-Sim/model/darkfield/best_model.pth",
    "data_path": "e:NingWang/All/ML-Sim/darkfield_dataset"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load datasets
# -------------------------------
parent_path=f"{config['data_path']}/combined"
print(parent_path)
# -------------------------------
# Data Split + Test Inference
# -------------------------------

# Load and split filenames
#all_files_clean = sorted(os.listdir(f"{config.data_path}/clean"))
#all_files_noise = sorted(os.listdir(f"{config.data_path}/noisy"))
                         

all_data=PairedImageDataset(f"{parent_path}/noisy",f"{parent_path}/clean",256)    
train_dataset,test_dataset = train_test_split(all_data, test_size=0.2, random_state=42)
train_dataset, val_dataset= train_test_split(train_dataset, test_size=0.1, random_state=42)
    
#train_files_noise, train_files_clean,test_files_noise,test_files_clean = train_test_split(all_files_noise, all_files_clean test_size=0.2, random_state=42)
#train_files_noise,train_files_clean, val_files_noise, val_files_clean= train_test_split(train_files_noise,train_files_clean, test_size=0.1, random_state=42)


class NoisyCleanImageDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths, transform=None):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths
        self.transform = transform

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_image = Image.open(self.noisy_paths[idx]).convert('RGB')
        clean_image = Image.open(self.clean_paths[idx]).convert('RGB')

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


#train_dataset = NoisyCleanImageDataset(train_files_noise, train_files_clean, transform=transform)
#val_dataset = NoisyCleanImageDataset(val_files_noise, val_files_clean, transform=transform)
#test_dataset = NoisyCleanImageDataset(test_files_noise, test_files_clean, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# -------------------------------
# Model Setup
# -------------------------------
if config["model_type"] == "unet":
    model = UNet().to(device)
elif config["model_type"] == "transformer":
    model = SimpleViT(img_size=256, patch_size=16, in_chans=1)
    #model = TransformerModel().to(device)
else:
    raise ValueError("Invalid model type")



if os.path.exists(config["save_path"]):
    print(f"📦 Loading existing model from {config['save_path']}...")
    model.load_state_dict(torch.load(config["save_path"]))


# -------------------------------
# Loss Function Setup
# -------------------------------
def get_loss_function(name):
    if name == "l1":
        return nn.L1Loss()
    elif name in ["l2", "mse"]:
        return nn.MSELoss()
    elif name == "ssim":
        if not has_ssim:
            raise ImportError("SSIM loss requires `piq`. Install with `pip install piq`.")
        return lambda pred, target: 1.0 - piq.ssim(pred, target, data_range=1.0)
    elif name == "combined":
        if not has_ssim:
            raise ImportError("Combined loss requires SSIM. Install with `pip install piq`.")
        alpha = config.get("ssim_weight", 0.84)
        mse = nn.MSELoss()
        return lambda pred, target: (1 - alpha) * mse(pred, target) + alpha * (1.0 - piq.ssim(pred, target, data_range=1.0))
    else:
        raise ValueError("Unsupported loss type")

criterion = get_loss_function(config["loss_type"])
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# -------------------------------
# Training Loop
# -------------------------------
best_val_loss = float("inf")
for epoch in range(config["epochs"]):
    model.train()
    running_loss = 0.0
    for noisy, clean in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            val_loss += criterion(output, clean).item()

    avg_train = running_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    if avg_val < best_val_loss:
        print("✅ Saving best model...")
        best_val_loss = avg_val
        torch.save(model.state_dict(), config["save_path"])

# -------------------------------
# Test Inference
# -------------------------------
def run_inference(model, test_loader, save_dir="test_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(test_loader):
            noisy = noisy.to(device)
            output = model(noisy)
            for i in range(noisy.size(0)):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(noisy[i, 0].cpu(), cmap='gray')
                axs[1].imshow(clean[i, 0].cpu(), cmap='gray')
                axs[2].imshow(output[i, 0].cpu(), cmap='gray')
                axs[0].set_title("Noisy Input")
                axs[1].set_title("Ground Truth")
                axs[2].set_title("Model Output")
                for ax in axs:
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"test_result_{idx}_{i}.png"))
                plt.close()

# -------------------------------
# Run Testing
# -------------------------------
model.load_state_dict(torch.load(config["save_path"]))
run_inference(model, test_loader)












# Example run (uncomment to execute)
# run_training_pipeline(model_type="unet", loss_type="combined", data_path="/mnt/data/darkfield_lung_sim", epochs=5)
'''
run_training_pipeline(
    model_type="transformer",                   # or "transformer"
    loss_type="combined",               # "l1", "l2", "ssim", or "combined"
    data_path="e:NingWang/All/ML-Sim/data/darkfield_lung_sim",
    epochs=5,
    save_model_path="e:NingWang/All/ML-Sim/model/darkfield_lung_sim_transformer",
    resume=True
)

run_training_pipeline(
    model_type="unet",                   # or "transformer"
    loss_type="combined",               # "l1", "l2", "ssim", or "combined"
    #data_path="e:NingWang/All/ML-Sim/data/darkfield_lung_sim_hybrid",e:/NingWang/All/ML-Sim/darkfield_dataset/combined
    data_path="e:/NingWang/All/ML-Sim/darkfield_dataset/combined",
    epochs=5,
    save_model_path="e:NingWang/All/ML-Sim/model/darkfield_lung_sim_unet",
    resume=False
)
'''