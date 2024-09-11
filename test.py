import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def generate_image(show_hair=True, mouth_open=0.26, mouth_smile=-0.01):
    img_size = (700, 600)
    y, x = np.indices(img_size)
    
    # Initialize the image with base color
    img = np.ones(img_size + (3,)) * (0.9, 0.85, 0.75)
    img[y > 300] = (0.5, 0.4, 0.3)
    img[(y > 200) & (y < 230)] = (0.5, 0.4, 0.3)
    img[(y > 530) & (y < 560)] = (0.3, 0.1, 0.0)
    
    # Draw various features
    img[(y < 200) & (x < 180) & (x > 160)] = (0.5, 0.4, 0.3)
    img[(y < 200) & (x < 480) & (x > 460)] = (0.5, 0.4, 0.3)
    img[edge(img, delta=0.1)] = 0.1

    # Apply random noise and blur
    img0 = img.copy()
    img[1:, 1:] *= np.random.uniform(0.8, 1.2, size=img[1:, 1:].shape)
    for _ in range(20):
        blur(img)
    
    # Eye details
    R = 40
    eye1 = (y - 380)**2 + 0.3 * (x - 190)**2 - (R + 3)**2
    eye2 = (y - 395)**2 + 0.05 * (x - 215)**2 - (R - 3)**2
    eye_all = (eye1 < 0) & (eye2 > 0)
    img[eye_all] = 0
    eye1 = (y - 380)**2 + 0.3 * (x - 190)**2 - R**2
    eye2 = (y - 395)**2 + 0.05 * (x - 215)**2 - R**2
    eye_all = (eye1 < 0) & (eye2 > 0)
    img[eye_all] = (1, 1, 1)
    eye_ball = (y - 350)**2 + 0.5 * (x - 195)**2 - (22)**2
    eye_ball = eye_all & (eye_ball < 0)
    img[eye_ball] = 0
    
    # Mouth details
    lip1 = 480 - 15 * mouth_open
    lip2 = 495 + 15 * mouth_open
    lip3 = 470 - 20 * mouth_smile
    k1 = (lip1 - lip3) / 100**2
    k2 = (lip2 - lip3) / 100**2
    mouth1 = (lip1 - k1 * (x - 300)**2 - y)
    mouth2 = (lip2 - k2 * (x - 300)**2 - y)
    img[(mouth1 <= 0) & (mouth2 >= 0)] = 1
    img[edge((mouth1 <= 0) & (mouth2 >= 0))] = 0
    img[(np.abs(mouth2) <= 1) & (mouth1 <= 5)] = 0
    
    # Eyebrows
    eyebrow1 = (x - 200)**2 - (y - 320) * 150
    eyebrow_all = (np.abs(eyebrow1) < 100) & (x > 200) & (x < 250)
    img[eyebrow_all] = 0
    eyebrow1 = 0.1 * (x - 200)**2 - (y - 235) * 150
    eyebrow_all = (np.abs(eyebrow1) < 140) & (x > 160) & (x < 250)
    img[eyebrow_all] = 0
    
    # Hair
    hair1 = 1.5 * np.abs(x - 300)**2.5 + np.abs(y - 430)**2.5 - 240**2.5
    hair2 = 1.5 * np.abs(x - 300)**2.5 + np.abs(y - 430)**2.5 - 240**2.5
    hair = (hair1 < 0) & (hair2 > 0)
    if show_hair:
        img[hair] = hair_color
        img[edge(hair)] = 0
        img[(hair1 < 0) & (hair1 > -20000 * (y - 400) / 300)] *= 0.8
    
    # Convert to PIL image and display
    return Image.fromarray(np.uint8((img * 255).clip(0, 255)))

# Example usage
image = generate_image()
plt.imshow(image)
plt.axis('off')
plt.show()
