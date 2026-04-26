import numpy as np
import cv2
import os
from tqdm import tqdm as tq

def lensless_speckle(image, speckle_size=5, strength=0.8):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape

    # Random phase
    phase = np.random.uniform(0, 2*np.pi, (h, w))

    # Complex field
    field = np.exp(1j * phase)

    # Fourier transform
    field_ft = np.fft.fft2(field)

    # Frequency grid
    ky = np.fft.fftfreq(h)
    kx = np.fft.fftfreq(w)
    KX, KY = np.meshgrid(kx, ky)

    # Low-pass filter
    filter_mask = np.exp(-(KX**2 + KY**2) * (speckle_size**2))
    field_ft_filtered = field_ft * filter_mask

    # Back to spatial domain
    field_spatial = np.fft.ifft2(field_ft_filtered)

    # Intensity
    speckle = np.abs(field_spatial) ** 2
    speckle = speckle / np.mean(speckle)

    # Apply to image
    noisy = gray * (1 + strength * (speckle - 1))
    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)


# ===== INPUT / OUTPUT ROOT =====
input_root = ""
output_root = ""

# Create output root if not exists
os.makedirs(output_root, exist_ok=True)

# ===== Traverse folders =====
for subfolder in tq(os.listdir(input_root)):
    subfolder_path = os.path.join(input_root, subfolder)

    if not os.path.isdir(subfolder_path):
        continue

    # Create corresponding output subfolder
    output_subfolder = os.path.join(output_root, subfolder)
    os.makedirs(output_subfolder, exist_ok=True)

    for filename in tq(os.listdir(subfolder_path)):
        input_path = os.path.join(subfolder_path, filename)

        # Skip non-images
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue

        # Read image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Skipping unreadable file: {input_path}")
            continue

        # Apply speckle
        speckle_img = lensless_speckle(img, speckle_size=5, strength=0.8)

        # Save output
        output_path = os.path.join(output_subfolder, filename)
        cv2.imwrite(output_path, speckle_img)

        # print(f"Saved: {output_path}")

print("✅ Done processing all images.")