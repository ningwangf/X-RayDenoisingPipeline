# Combine all the simulation logic into one self-contained Python script

import numpy as np
import os
import imageio
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# General configuration
img_size = (512, 512)
img_format = "png"
num_images = 20
dose_range = (0.3, 0.9)
visibility_range = (0.4, 0.95)
base_dir = "e:/NingWang/All/ML-Sim/darkfield_dataset"
organs = ["lung", "breast", "spine", "hand", "sinus"]
combined_clean_dir = os.path.join(base_dir, "combined", "clean")
combined_noisy_dir = os.path.join(base_dir, "combined", "noisy")
os.makedirs(combined_clean_dir, exist_ok=True)
os.makedirs(combined_noisy_dir, exist_ok=True)

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def compute_scatter_map(phantom):
    local_mean = gaussian_filter(phantom, sigma=3)
    local_sq_mean = gaussian_filter(phantom**2, sigma=3)
    local_var = local_sq_mean - local_mean**2
    visibility = np.exp(-local_var * 10)
    return normalize(1.0 - visibility)

def add_noise(image, dose=0.5, visibility=0.5):
    signal = image * dose * visibility
    noisy = np.random.poisson(signal * 255) / 255.0
    return np.clip(noisy, 0, 1)

def simulate_lung_phantom(shape):
    base = np.zeros(shape)
    cv2.ellipse(base, (shape[1]//2, shape[0]//2), (shape[1]//3, shape[0]//2), 0, 0, 360, 1, -1)
    base = gaussian_filter(base, sigma=8)
    for _ in range(15):
        x, y = np.random.randint(80, shape[1]-80), np.random.randint(80, shape[0]-80)
        angle = np.random.rand() * 360
        length = np.random.randint(30, 60)
        thickness = np.random.randint(1, 3)
        cv2.ellipse(base, (x, y), (length, 2), angle, 0, 360, 0.5 + 0.5*np.random.rand(), -1)
    for _ in range(5):
        x, y = np.random.randint(100, shape[1]-100), np.random.randint(100, shape[0]-100)
        cv2.circle(base, (x, y), np.random.randint(4, 10), 1, -1)
    for i in range(30, shape[0]-30, 32):
        cv2.line(base, (0, i), (shape[1], i), 0.3, thickness=2)
    micro = gaussian_filter(np.random.rand(*shape), sigma=1.5)
    return normalize(base * micro)


def simulate_breast_phantom(shape=(512, 512)):
    img = np.zeros(shape)
    for _ in range(120):
        x, y = np.random.randint(50, shape[1]-50), np.random.randint(50, shape[0]-50)
        radius = np.random.randint(10, 40)
        intensity = np.random.uniform(0.4, 0.8)
        cv2.circle(img, (x, y), radius, intensity, -1)
    center_x = shape[1] // 2
    for i in range(10):
        y = shape[0] - i * 40
        length = np.random.randint(30, 70)
        angle = np.random.randint(-30, 30)
        cv2.ellipse(img, (center_x, y), (length, 5), angle, 0, 360, 0.9, -1)
    for _ in range(10):
        cx, cy = np.random.randint(100, 400), np.random.randint(100, 400)
        for _ in range(np.random.randint(5, 15)):
            dx, dy = np.random.randint(-10, 10), np.random.randint(-10, 10)
            img[cy+dy, cx+dx] = 1.0
    texture = gaussian_filter(np.random.rand(*shape), sigma=6)
    img += texture * 0.2
    return normalize(img)

def simulate_spine_phantom(shape=(512, 512)):
    img = np.zeros(shape)
    for y in range(100, shape[0]-100, 80):
        width = np.random.randint(100, 120)
        height = np.random.randint(30, 40)
        x_center = shape[1] // 2 + np.random.randint(-10, 10)
        cv2.rectangle(img, (x_center - width//2, y), (x_center + width//2, y + height), 0.8, -1)
        for _ in range(100):
            tx = np.random.randint(x_center - width//2 + 5, x_center + width//2 - 5)
            ty = np.random.randint(y + 5, y + height - 5)
            img[ty, tx] = 1.0
    for y in range(120, shape[0]-100, 90):
        cv2.ellipse(img, (shape[1]//2, y), (150, 10), 0, 0, 360, 0.3, 1)
    img += gaussian_filter(np.random.rand(*shape), sigma=4) * 0.2
    return normalize(img)



# ------------------------
# Hybrid Procedural: HAND
# ------------------------
def simulate_procedural_hand(shape=(512, 512)):
    img = np.zeros(shape)

    # Finger bones
    for i in range(5):
        x_base = 80 + i * 80
        for j in range(3):  # phalanges
            y1 = 100 + j * 80
            y2 = y1 + 40
            width = np.random.randint(12, 20)
            cv2.rectangle(img, (x_base - width // 2, y1), (x_base + width // 2, y2), 0.8, -1)
            # Add trabecular pattern
            for _ in range(30):
                tx = np.random.randint(x_base - width // 2 + 2, x_base + width // 2 - 2)
                ty = np.random.randint(y1 + 2, y2 - 2)
                img[ty, tx] = 1.0

    # Palm bones
    for i in range(5):
        x_base = 80 + i * 80
        y1, y2 = 340, 420
        width = np.random.randint(20, 30)
        cv2.rectangle(img, (x_base - width // 2, y1), (x_base + width // 2, y2), 0.7, -1)
        for _ in range(50):
            tx = np.random.randint(x_base - width // 2 + 2, x_base + width // 2 - 2)
            ty = np.random.randint(y1 + 2, y2 - 2)
            img[ty, tx] = 1.0

    # Soft tissue mask
    tissue_mask = gaussian_filter(np.random.rand(*shape), sigma=8)
    img += tissue_mask * 0.2
    return normalize(img)

# ------------------------
# Hybrid Procedural: SINUS
# ------------------------
def simulate_procedural_sinus(shape=(512, 512)):
    img = np.zeros(shape)

    # Skull base
    cv2.ellipse(img, (shape[1]//2, shape[0]//2), (180, 80), 0, 0, 360, 0.5, -1)

    # Nasal cavity and sinuses
    nasal_x = shape[1] // 2
    for offset in [-40, 40]:  # left and right sinuses
        cv2.circle(img, (nasal_x + offset, shape[0]//2), 25, 0.1, -1)
        cv2.circle(img, (nasal_x + offset, shape[0]//2 + 40), 15, 0.15, -1)

    # Septum and turbinates
    cv2.line(img, (nasal_x, shape[0]//2 - 40), (nasal_x, shape[0]//2 + 60), 0.3, 2)
    for y in range(shape[0]//2, shape[0]//2 + 60, 15):
        cv2.ellipse(img, (nasal_x, y), (10, 3), 0, 0, 360, 0.25, -1)

    # Add cranial noise and asymmetry
    texture = gaussian_filter(np.random.rand(*shape), sigma=6)
    img += texture * 0.25
    return normalize(img)


def simulate_hand_phantom(shape):
    phantom = np.zeros(shape)
    for i in range(5):
        base_x = 80 + i * 80
        for j in range(3):
            y1 = 100 + j * 80
            y2 = y1 + 40
            cv2.rectangle(phantom, (base_x - 10, y1), (base_x + 10, y2), 0.8, -1)
            for _ in range(20):
                x = np.random.randint(base_x - 8, base_x + 8)
                y = np.random.randint(y1 + 4, y2 - 4)
                phantom[y, x] = 1.0
    return normalize(gaussian_filter(phantom + np.random.normal(0, 0.01, shape), sigma=1.5))

def simulate_sinus_phantom(shape):
    phantom = np.zeros(shape)
    cv2.ellipse(phantom, (shape[1]//2, shape[0]//2), (120, 60), 0, 0, 360, 1, -1)
    cv2.circle(phantom, (shape[1]//2 - 40, shape[0]//2), 20, 0, -1)
    cv2.circle(phantom, (shape[1]//2 + 40, shape[0]//2), 20, 0, -1)
    phantom += gaussian_filter(np.random.rand(*shape) * 0.3, sigma=1)
    return normalize(phantom)

# Main generation loop
image_index = 0
for organ in organs:
    organ_dir = os.path.join(base_dir, organ)
    clean_dir = os.path.join(organ_dir, "clean")
    noisy_dir = os.path.join(organ_dir, "noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Generating {organ} dataset"):
        if organ == "lung":
            phantom = simulate_lung_phantom(img_size)
        elif organ == "breast":
            phantom = simulate_breast_phantom(img_size)
        elif organ == "spine":
            phantom = simulate_spine_phantom(img_size)
        elif organ == "hand":
            phantom = simulate_procedural_hand(img_size)
        elif organ == "sinus":
            phantom = simulate_procedural_sinus(img_size)
        else:
            continue

        clean = compute_scatter_map(phantom)
        dose = np.random.uniform(*dose_range)
        visibility = np.random.uniform(*visibility_range)
        noisy = add_noise(clean, dose, visibility)

        clean_path = os.path.join(clean_dir, f"{organ}_{i:04d}_clean.{img_format}")
        noisy_path = os.path.join(noisy_dir, f"{organ}_{i:04d}_noisy.{img_format}")
        imageio.imwrite(clean_path, (clean * 255).astype(np.uint8))
        imageio.imwrite(noisy_path, (noisy * 255).astype(np.uint8))

        combined_clean = os.path.join(combined_clean_dir, f"{organ}_{image_index:05d}_clean.{img_format}")
        combined_noisy = os.path.join(combined_noisy_dir, f"{organ}_{image_index:05d}_noisy.{img_format}")
        imageio.imwrite(combined_clean, (clean * 255).astype(np.uint8))
        imageio.imwrite(combined_noisy, (noisy * 255).astype(np.uint8))
        image_index += 1

# ----------------- Visualization ------------------
fig, axes = plt.subplots(len(organs), 3, figsize=(12, 3 * len(organs)))
for idx, organ in enumerate(organs):
    if organ == "lung":
        phantom = simulate_lung_phantom(img_size)
    elif organ == "breast":
        phantom = simulate_breast_phantom(img_size)
    elif organ == "spine":
        phantom = simulate_spine_phantom(img_size)
    elif organ == "hand":
        phantom = simulate_procedural_hand(img_size)
    elif organ == "sinus":
        phantom = simulate_procedural_sinus(img_size)
    clean = compute_scatter_map(phantom)
    noisy = add_noise(clean, dose=0.5, visibility=0.5)

    axes[idx, 0].imshow(phantom, cmap='gray')
    axes[idx, 0].set_title(f'{organ} phantom')
    axes[idx, 1].imshow(clean, cmap='gray')
    axes[idx, 1].set_title(f'{organ} clean')
    axes[idx, 2].imshow(noisy, cmap='gray')
    axes[idx, 2].set_title(f'{organ} noisy')
    for ax in axes[idx]:
        ax.axis('off')
plt.tight_layout()
plt.show()


exit(0)





import numpy as np
import os
import imageio
from scipy.ndimage import gaussian_filter
from skimage import exposure, util
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Paths
OUTPUT_DIR = "e:NingWang/All/ML-Sim/data/darkfield_lung_sim_hybrid"
GT_DIR = os.path.join(OUTPUT_DIR, "clean")
NOISY_DIR = os.path.join(OUTPUT_DIR, "noisy")
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Configuration
IMG_SIZE = (512, 512)
NUM_IMAGES = 50
IMG_FORMAT = "png"
DOSE_RANGE = (0.3, 0.9)
VISIBILITY_RANGE = (0.4, 0.95)
NOISE_TYPE = "poisson"

# Utility functions
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def generate_microstructure(shape):
    """Generate fine-scale alveolar-like structure using Perlin-style noise"""
    base = np.random.rand(*shape)
    micro = gaussian_filter(base, sigma=1.5)
    return normalize(micro)




# Add simulation support for hand and sinus images
def simulate_hand_phantom(shape):
    """Simulate long bones and joints typical of a hand."""
    phantom = np.zeros(shape)
    # Simulate 5 fingers with phalanges and joints
    for i in range(5):
        base_x = 80 + i * 80
        for j in range(3):
            y1 = 100 + j * 80
            y2 = y1 + 40
            cv2.rectangle(phantom, (base_x - 10, y1), (base_x + 10, y2), 0.8, -1)
            # Add trabecular texture
            for _ in range(20):
                x = np.random.randint(base_x - 8, base_x + 8)
                y = np.random.randint(y1 + 4, y2 - 4)
                phantom[y, x] = 1.0
    return normalize(gaussian_filter(phantom + np.random.normal(0, 0.01, shape), sigma=1.5))

def simulate_sinus_phantom(shape):
    """Simulate nasal cavity, air pockets, and bone structures."""
    phantom = np.zeros(shape)
    cv2.ellipse(phantom, (shape[1]//2, shape[0]//2), (120, 60), 0, 0, 360, 1, -1)
    cv2.circle(phantom, (shape[1]//2 - 40, shape[0]//2), 20, 0, -1)
    cv2.circle(phantom, (shape[1]//2 + 40, shape[0]//2), 20, 0, -1)
    phantom += gaussian_filter(np.random.rand(*shape) * 0.3, sigma=1)
    return normalize(phantom)

# Update organ list
organs = ["lung", "breast", "spine", "hand", "sinus"]
combined_clean_dir = "e:/NingWang/All/ML-Sim/darkfield_dataset/combined/clean"
combined_noisy_dir = "e:/NingWang/All/ML-Sim/darkfield_dataset/combined/noisy"
os.makedirs(combined_clean_dir, exist_ok=True)
os.makedirs(combined_noisy_dir, exist_ok=True)

# Generate datasets for all organs and merge into combined set
image_index = 0
for organ in organs:
    organ_dir = os.path.join(base_dir, organ)
    clean_dir = os.path.join(organ_dir, "clean")
    noisy_dir = os.path.join(organ_dir, "noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Generating {organ} dataset"):
        if organ == "lung":
            phantom = simulate_lung_phantom(img_size)
        elif organ == "breast":
            phantom = simulate_breast_phantom(img_size)
        elif organ == "spine":
            phantom = simulate_spine_phantom(img_size)
        elif organ == "hand":
            phantom = simulate_hand_phantom(img_size)
        elif organ == "sinus":
            phantom = simulate_sinus_phantom(img_size)
        else:
            continue

        clean = compute_scatter_map(phantom)
        dose = np.random.uniform(*dose_range)
        visibility = np.random.uniform(*visibility_range)
        noisy = add_noise(clean, dose, visibility)

        clean_path = os.path.join(clean_dir, f"{organ}_{i:04d}_clean.{img_format}")
        noisy_path = os.path.join(noisy_dir, f"{organ}_{i:04d}_noisy.{img_format}")
        imageio.imwrite(clean_path, (clean * 255).astype(np.uint8))
        imageio.imwrite(noisy_path, (noisy * 255).astype(np.uint8))

        # Also save to combined folder
        imageio.imwrite(os.path.join(combined_clean_dir, f"{organ}_{image_index:05d}_clean.{img_format}"), (clean * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(combined_noisy_dir, f"{organ}_{image_index:05d}_noisy.{img_format}"), (noisy * 255).astype(np.uint8))
        image_index += 1

def generate_lung_phantom(shape):
    """Add anatomical lung shape with ribs, nodules, and microstructure."""
    base = np.zeros(shape)
    cv2.ellipse(base, (shape[1]//2, shape[0]//2), (shape[1]//3, shape[0]//2), 0, 0, 360, 1, -1)
    base = gaussian_filter(base, sigma=8)

    # Vessels/bronchi
    for _ in range(15):
        x, y = np.random.randint(80, shape[1]-80), np.random.randint(80, shape[0]-80)
        angle = np.random.rand() * 360
        length = np.random.randint(30, 60)
        thickness = np.random.randint(1, 3)
        cv2.ellipse(base, (x, y), (length, 2), angle, 0, 360, 0.5 + 0.5*np.random.rand(), -1)

    # Nodules
    for _ in range(5):
        x, y = np.random.randint(100, shape[1]-100), np.random.randint(100, shape[0]-100)
        cv2.circle(base, (x, y), np.random.randint(4, 10), 1, -1)

    # Ribs
    for i in range(30, shape[0]-30, 32):
        cv2.line(base, (0, i), (shape[1], i), 0.3, thickness=2)

    base = gaussian_filter(base + np.random.normal(0, 0.02, shape), sigma=2)
    structure = generate_microstructure(shape)
    phantom = base * structure
    return normalize(phantom)

def compute_scat_map(phantom):
    """Estimate small-angle scattering map using local variance."""
    local_mean = gaussian_filter(phantom, sigma=3)
    local_sq_mean = gaussian_filter(phantom**2, sigma=3)
    local_variance = local_sq_mean - local_mean**2
    return normalize(local_variance)

def simulate_darkfield_visibility_loss(phantom):
    """Hybrid model: V = V0 * exp(-scatter), where scatter ∝ local variance."""
    scatter_map = compute_scat_map(phantom)
    V0 = 1.0  # base visibility
    darkfield = V0 * np.exp(-scatter_map * 10)  # exaggerate scatter impact
    return normalize(1.0 - darkfield)  # inverted: higher scatter → brighter

def add_noise(image, dose=0.5, visibility=0.5, noise_type='poisson'):
    signal = image * dose * visibility
    if noise_type == 'poisson':
        noisy = np.random.poisson(signal * 255) / 255.0
    elif noise_type == 'gaussian':
        noisy = signal + np.random.normal(0, 0.01, signal.shape)
    return np.clip(noisy, 0, 1)

# Generate data
for i in tqdm(range(NUM_IMAGES), desc="Generating hybrid dark-field dataset"):
    phantom = generate_lung_phantom(IMG_SIZE)
    clean = simulate_darkfield_visibility_loss(phantom)

    dose = np.random.uniform(*DOSE_RANGE)
    visibility = np.random.uniform(*VISIBILITY_RANGE)
    noisy = add_noise(clean, dose=dose, visibility=visibility, noise_type=NOISE_TYPE)

    clean_path = os.path.join(GT_DIR, f"lung_{i:04d}_clean.{IMG_FORMAT}")
    noisy_path = os.path.join(NOISY_DIR, f"lung_{i:04d}_noisy.{IMG_FORMAT}")

    imageio.imwrite(clean_path, (clean * 255).astype(np.uint8))
    imageio.imwrite(noisy_path, (noisy * 255).astype(np.uint8))

# Visual example
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray')
plt.title("Lung + Microstructure")
plt.subplot(1, 3, 2)
plt.imshow(clean, cmap='gray')
plt.title("Clean Dark-field (Hybrid Model)")
plt.subplot(1, 3, 3)
plt.imshow(noisy, cmap='gray')
plt.title("Noisy Dark-field")
plt.tight_layout()
plt.show()


exit(0)


import numpy as np
import os
import imageio
from scipy.ndimage import gaussian_filter
from skimage import exposure
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel

# Paths
OUTPUT_DIR = "e:NingWang/All/ML-Sim/data/darkfield_lung_sim_v2"
GT_DIR = os.path.join(OUTPUT_DIR, "clean")
NOISY_DIR = os.path.join(OUTPUT_DIR, "noisy")
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Configuration
IMG_SIZE = (512, 512)         # Higher resolution
NUM_IMAGES = 50               # More variety
IMG_FORMAT = "png"
DOSE_RANGE = (0.3, 0.9)
VISIBILITY_RANGE = (0.4, 0.95)
NOISE_TYPE = "poisson"

# Utility functions
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def generate_realistic_lung_phantom(shape=(512, 512)):
    """Simulate lung-like regions with embedded structures like nodules and ribs."""
    base = np.zeros(shape)
    cv2.ellipse(base, (shape[1]//2, shape[0]//2), (shape[1]//3, shape[0]//2), 0, 0, 360, 1, -1)
    base = gaussian_filter(base, sigma=8)

    # Simulate bronchi and vessels
    for _ in range(20):
        x, y = np.random.randint(100, shape[1]-100), np.random.randint(100, shape[0]-100)
        angle = np.random.rand() * 360
        length = np.random.randint(30, 80)
        thickness = np.random.randint(1, 4)
        cv2.ellipse(base, (x, y), (length, 2), angle, 0, 360, 0.5 + 0.5*np.random.rand(), -1)

    # Add nodules
    for _ in range(5):
        x, y = np.random.randint(120, shape[1]-120), np.random.randint(120, shape[0]-120)
        cv2.circle(base, (x, y), np.random.randint(5, 15), 1, -1)

    # Simulate ribs with horizontal bands
    for i in range(30, shape[0]-30, 30):
        cv2.line(base, (0, i), (shape[1], i), 0.3, thickness=2)

    noisy_structure = base + np.random.normal(0, 0.05, shape)
    noisy_structure = gaussian_filter(noisy_structure, sigma=2)

    return normalize(noisy_structure)



def simulate_darkfield_local(phantom):
    edges = sobel(phantom)
    darkfield = gaussian_filter(edges**2, sigma=2)
    return normalize(darkfield)

def simulate_darkfield(phantom):
    """Simulate dark-field effect via scattering model (Fourier power spectrum)."""
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(phantom)))**2
    darkfield = gaussian_filter(spectrum, sigma=3)
    darkfield = normalize(darkfield)
    return exposure.equalize_adapthist(darkfield, clip_limit=0.03)

def add_noise(image, dose=0.5, visibility=0.5, noise_type='poisson'):
    """Add dose-dependent and visibility-related noise."""
    signal = image * dose * visibility
    if noise_type == 'poisson':
        noisy = np.random.poisson(signal * 255) / 255.0
    elif noise_type == 'gaussian':
        noisy = signal + np.random.normal(0, 0.01, signal.shape)
    return np.clip(noisy, 0, 1)

# Generate data
for i in tqdm(range(NUM_IMAGES), desc="Generating realistic dark-field data"):
    phantom = generate_realistic_lung_phantom(IMG_SIZE)
    #clean = simulate_darkfield(phantom)
    clean = simulate_darkfield_local(phantom)

    dose = np.random.uniform(*DOSE_RANGE)
    visibility = np.random.uniform(*VISIBILITY_RANGE)
    noisy = add_noise(clean, dose=dose, visibility=visibility, noise_type=NOISE_TYPE)

    clean_path = os.path.join(GT_DIR, f"lung_{i:04d}_clean.{IMG_FORMAT}")
    noisy_path = os.path.join(NOISY_DIR, f"lung_{i:04d}_noisy.{IMG_FORMAT}")

    imageio.imwrite(clean_path, (clean * 255).astype(np.uint8))
    imageio.imwrite(noisy_path, (noisy * 255).astype(np.uint8))

# Visual example
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray')
plt.title("Simulated Lung Structure")
plt.subplot(1, 3, 2)
plt.imshow(clean, cmap='gray')
plt.title("Simulated Clean Dark-field")
plt.subplot(1, 3, 3)
plt.imshow(noisy, cmap='gray')
plt.title("Simulated Noisy Dark-field")
plt.tight_layout()
plt.show()


exit(0)





import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

# Output directories
OUTPUT_DIR = "e:NingWang/All/ML-Sim/data/darkfield_lung_sim"
GT_DIR = os.path.join(OUTPUT_DIR, "clean")
NOISY_DIR = os.path.join(OUTPUT_DIR, "noisy")
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Image parameters
IMG_SIZE = (256, 256)
IMG_FORMAT = "png"
NUM_IMAGES = 20

# Noise parameters
DOSE_RANGE = (0.3, 1.0)
VISIBILITY_RANGE = (0.4, 1.0)
NOISE_TYPE = 'poisson'

# Load a sample lung mask or synthetic lung texture (simulate one here)
def generate_lung_structure(shape=(256, 256)):
    img = np.zeros(shape)
    cv2.ellipse(img, (shape[1]//2, shape[0]//2), (shape[1]//3, shape[0]//2), 0, 0, 360, 1, -1)
    noise = np.random.normal(0, 0.2, shape)
    lung_texture = gaussian_filter(img + noise, sigma=3)
    return np.clip(lung_texture, 0, 1)

def simulate_darkfield(structure):
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft2(structure)))**2
    darkfield = gaussian_filter(power_spectrum, sigma=2)
    return (darkfield - np.min(darkfield)) / (np.max(darkfield) - np.min(darkfield))

def add_noise(image, dose=0.5, visibility=0.5, noise_type='poisson'):
    signal = image * dose * visibility
    if noise_type == 'poisson':
        noisy = np.random.poisson(signal * 255) / 255.0
    elif noise_type == 'gaussian':
        noisy = signal + np.random.normal(0, 0.02, signal.shape)
    else:
        raise ValueError("Unsupported noise type.")
    return np.clip(noisy, 0, 1)

# Generate realistic anatomical simulation
for i in tqdm(range(NUM_IMAGES), desc="Generating lung-based dark-field dataset"):
    lung_structure = generate_lung_structure(IMG_SIZE)
    clean_img = simulate_darkfield(lung_structure)

    dose = np.random.uniform(*DOSE_RANGE)
    visibility = np.random.uniform(*VISIBILITY_RANGE)
    noisy_img = add_noise(clean_img, dose=dose, visibility=visibility, noise_type=NOISE_TYPE)

    clean_path = os.path.join(GT_DIR, f"lung_{i:04d}_clean.{IMG_FORMAT}")
    noisy_path = os.path.join(NOISY_DIR, f"lung_{i:04d}_noisy.{IMG_FORMAT}")

    imageio.imwrite(clean_path, (clean_img * 255).astype(np.uint8))
    imageio.imwrite(noisy_path, (noisy_img * 255).astype(np.uint8))

# Visual example
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(lung_structure, cmap='gray')
plt.title("Synthetic Lung Structure")
plt.subplot(1, 3, 2)
plt.imshow(clean_img, cmap='gray')
plt.title("Simulated Clean Dark-field")
plt.subplot(1, 3, 3)
plt.imshow(noisy_img, cmap='gray')
plt.title("Simulated Noisy Image")
plt.tight_layout()
plt.show()


exit(0)

import os
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "/NingWang/All/ML-Sim/data/darkfield_sim"
GT_DIR = os.path.join(OUTPUT_DIR, "clean")
NOISY_DIR = os.path.join(OUTPUT_DIR, "noisy")
IMG_FORMAT = "png"  # Could be "png", "jpg", or "tiff"
IMG_SIZE = (256, 256)  # Customizable image size
NUM_IMAGES = 20  # Number of image pairs to generate

# Noise parameters
DOSE_RANGE = (0.2, 1.0)        # Simulates exposure (lower = more noisy)
VISIBILITY_RANGE = (0.3, 1.0)  # Simulates grating contrast
NOISE_TYPE = 'poisson'         # Options: 'poisson', 'gaussian'

# Create directories
os.makedirs(GT_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Utility functions
def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def generate_phantom(shape):
    phantom = np.random.rand(*shape)
    phantom = gaussian_filter(phantom, sigma=4)
    return normalize(phantom)

def simulate_darkfield(phantom):
    scattering_power = np.abs(np.fft.fftshift(np.fft.fft2(phantom)))**2
    darkfield = gaussian_filter(scattering_power, sigma=3)
    return normalize(darkfield)

def add_noise(image, dose=0.5, visibility=0.5, noise_type='poisson'):
    signal = image * dose * visibility
    if noise_type == 'poisson':
        noisy = np.random.poisson(signal * 255) / 255.0
    elif noise_type == 'gaussian':
        noisy = signal + np.random.normal(0, 0.02, signal.shape)
    else:
        raise ValueError("Unsupported noise type.")
    return np.clip(noisy, 0, 1)

# Generate dataset
for i in tqdm(range(NUM_IMAGES), desc="Generating dataset"):
    phantom = generate_phantom(IMG_SIZE)
    clean_img = simulate_darkfield(phantom)

    dose = np.random.uniform(*DOSE_RANGE)
    visibility = np.random.uniform(*VISIBILITY_RANGE)
    noisy_img = add_noise(clean_img, dose=dose, visibility=visibility, noise_type=NOISE_TYPE)

    # Save images
    clean_path = os.path.join(GT_DIR, f"img_{i:04d}_clean.{IMG_FORMAT}")
    noisy_path = os.path.join(NOISY_DIR, f"img_{i:04d}_noisy.{IMG_FORMAT}")

    imageio.imwrite(clean_path, (clean_img * 255).astype(np.uint8))
    imageio.imwrite(noisy_path, (noisy_img * 255).astype(np.uint8))

# Visual check for last image
plt.subplot(1, 2, 1)
plt.imshow(clean_img, cmap='gray')
plt.title("Clean Image")
plt.subplot(1, 2, 2)
plt.imshow(noisy_img, cmap='gray')
plt.title("Noisy Image")
plt.show()
