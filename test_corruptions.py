import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from imagecorruptions import corrupt

def test_single_corruption():
    # Find the first image in the Train/0 folder
    class_0_dir = os.path.join('data', 'Train', '0')
    first_image_name = os.listdir(class_0_dir)[0]
    image_path = os.path.join(class_0_dir, first_image_name)

    # imagecorruptions requires a numpy array
    image = Image.open(image_path).resize((32, 32))
    image = np.array(image)

    # Apply fog at a medium severity (1 to 5)
    corrupted_image = corrupt(image, corruption_name='fog', severity=3)

    # Plot and save as PDF
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis('off')
    
    ax[1].imshow(corrupted_image)
    ax[1].set_title("Fog - Severity 3")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("corruption_test.pdf", format="pdf")
    print("Success! Saved 'corruption_test.pdf'.")

if __name__ == "__main__":
    test_single_corruption()