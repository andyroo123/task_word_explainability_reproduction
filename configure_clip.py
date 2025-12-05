import torch_directml
import torch
import clip
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

def plot_similarity(text_descriptions, similarity_score, images_for_display):
    count = len(text_descriptions)
    fig, ax = plt.subplots(figsize=(18, 15))
    im = ax.imshow(similarity_score, cmap=plt.cm.YlOrRd)
    plt.colorbar(im, ax=ax)

    # y-axis ticks: text descriptions
    ax.set_yticks(np.arange(count))
    ax.set_yticklabels(text_descriptions, fontsize=12)
    ax.set_xticklabels([])
    ax.xaxis.set_visible(False) 

    for i, image in enumerate(images_for_display):
        ax.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        

    for x in range(similarity_score.shape[1]):
        for y in range(similarity_score.shape[0]):
            ax.text(x, y, f"{similarity_score[y, x]:.2f}", ha="center", va="center", size=10)

    ax.spines[["left", "top", "right", "bottom"]].set_visible(False)

    # Setting limits for the x and y axes
    ax.set_xlim([-0.5, count - 0.5])
    ax.set_ylim([count + 0.5, -2])

    # Adding title to the plot
    ax.set_title("Text and Image Similarity Score calculated with CLIP", size=14)

    plt.show()

if __name__ == "__main__":
    device = torch_directml.device()
    # in this blog we will load ViT-L/14@336px clip model
    model, preprocess = clip.load("ViT-L/14@336px", device="cpu")
    #model.cuda().eval()
    model = model.to(device).eval()
    # check the model architecture
    print(model)
    # check the preprocessor
    print(preprocess)

    # use images from COCO dataset and their textual descriptions
    image_urls  = [
        "http://farm1.staticflickr.com/6/8378612_34ab6787ae_z.jpg",
        "http://farm9.staticflickr.com/8456/8033451486_aa38ee006c_z.jpg",
        "http://farm9.staticflickr.com/8344/8221561363_a6042ba9e0_z.jpg",
        "http://farm5.staticflickr.com/4147/5210232105_b22d909ab7_z.jpg",
        "http://farm4.staticflickr.com/3098/2852057907_29f1f35ff7_z.jpg",
        "http://farm4.staticflickr.com/3324/3289158186_155a301760_z.jpg",
        "http://farm4.staticflickr.com/3718/9148767840_a30c2c7dcb_z.jpg",
        "http://farm9.staticflickr.com/8030/7989105762_4ef9e7a03c_z.jpg"
    ]

    text_descriptions = [
        "a cat standing on a wooden floor",
        "an airplane on the runway",
        "a white truck parked next to trees",
        "an elephant standing in a zoo",
        "a laptop on a desk beside a window",
        "a giraffe standing in a dirt field",
        "a bus stopped at a bus stop",
        "two bunches of bananas in the market"
    ]

    images_for_display=[]
    images=[]

    # Create a new figure
    plt.figure(figsize=(12, 6))
    size = (400, 320)
    # Loop through each URL and plot the image in a subplot
    for i, url1 in enumerate(image_urls):
        # # Get image from URL
        response = requests.get(url1)
        image = Image.open(BytesIO(response.content))
        image = image.resize(size)

        # Add subplot (2 rows, 4 columns, index i+1)
        plt.subplot(2, 4, i + 1)

        # Plot image
        plt.imshow(image)
        plt.axis('off')  # Turn off axes labels

        # Add a title (optional)
        plt.title(f'{text_descriptions[i]}')

        images_for_display.append(image)
        images.append(preprocess(image))

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

    device = torch_directml.device()
    image_inputs = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(["It is " + text for text in text_descriptions]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity_score = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    plot_similarity(text_descriptions, similarity_score, images_for_display)