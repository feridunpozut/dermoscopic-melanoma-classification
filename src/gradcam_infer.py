'''
Grad-CAM (Gradient-weighted Class Activation Mapping)
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from model import LiteSkinLesionClassifier

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) class for visualizing
    the important regions in the input image that contribute to the model's decision.
    This implementation uses PyTorch hooks to capture gradients and activations.
    """
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM instance.
        Args:
            model (torch.nn.Module): The model for which Grad-CAM is to be computed.
            target_layer (torch.nn.Module): The layer in the model to which Grad-CAM will be applied.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """
        Saves the activations from the target layer.
        Args:
            module (torch.nn.Module): The target layer module.
            input (tuple): The input to the layer.
            output (torch.Tensor): The output from the layer.
        """
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Saves the gradients from the target layer.
        Args:
            module (torch.nn.Module): The target layer module.
            grad_input (tuple): The gradients with respect to the input.
            grad_output (tuple): The gradients with respect to the output.
        """
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generates the Grad-CAM heatmap for the given input tensor.
        Args:
            input_tensor (torch.Tensor): The input image tensor.
            class_idx (int, optional): The index of the class for which to generate the CAM.
        Returns:
            np.ndarray: The generated CAM heatmap.
        """
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients
        activations = self.activations
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[0, i, :, :] *= pooled_grads[i]

        cam = activations[0].sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

def visualize_gradcam_on_image_sl(img_tensor, cam, alpha=0.5):
    """
    Visualizes the Grad-CAM heatmap on the input image using Streamlit.
    Args:
        img_tensor (torch.Tensor): The input image tensor.
        cam (np.ndarray): The Grad-CAM heatmap.
        alpha (float): The transparency factor for the overlay.
    Returns:
        PIL.Image: The overlay image with Grad-CAM heatmap applied.
    """
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    overlay_pil = Image.fromarray(overlay)
    return overlay_pil

def visualize_gradcam_on_image(img_tensor, cam, alpha=0.5):
    """
    Visualizes the Grad-CAM heatmap on the input image using Matplotlib.
    Args:
        img_tensor (torch.Tensor): The input image tensor.
        cam (np.ndarray): The Grad-CAM heatmap.
        alpha (float): The transparency factor for the overlay.
    Returns:
        None: Displays the images using Matplotlib.
    """
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("../experiments/results/gradcam_overlay.png")
    
if __name__ == "__main__":
    image_path = "../data/processed/artefact_removal/ISIC_7128019.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteSkinLesionClassifier(model_name='efficientnet_b0', pretrained=False).to(device)
    model.load_state_dict(torch.load("../experiments/checkpoints/best_model_efficientnet_b0.pt", map_location=device))

    target_layer = model.backbone.blocks[-1]
    cam_generator = GradCAM(model, target_layer)

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam = cam_generator.generate_cam(input_tensor)
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()])
    visualize_gradcam_on_image(transform(image).unsqueeze(0).to(device), cam)
