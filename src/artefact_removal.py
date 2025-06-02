'''
Artefact Removal Module
This module provides functions to clean dermoscopic images by removing artefacts such as hair, correcting brightness, and addressing vignette effects.
'''

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class ArtefactRemoval():
    '''
    Class for removing artefacts from dermoscopic images.
    '''
    def __init__(self, image):
        self.image = image
        
    def remove_hair(self, image):
        '''
        Removes hair artefacts from the image using morphological operations and inpainting.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            numpy.ndarray: Image with hair artefacts removed.
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(median, threshold1=10, threshold2=70)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        closed_mask = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        inpainted = cv2.inpaint(image, closed_mask, 1, cv2.INPAINT_TELEA)

        return inpainted

    def normalize_brightness(self,image, target_mean=110):
        '''
        Normalizes the brightness of the image to a target mean value.
        Args:
            image (numpy.ndarray): Input image in RGB format.
            target_mean (int): Target mean brightness value.
        Returns:
            numpy.ndarray: Brightness-normalized image.
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        current_mean = np.mean(gray)

        if current_mean < target_mean:
            ratio = target_mean / (current_mean + 1e-5)
            image = np.clip(image * ratio, 0, 255).astype(np.uint8)
        return image

    def correct_vignette(self,image):
        '''
        Corrects vignette effects in the image using a Gaussian mask.
        Args:
            image (numpy.ndarray): Input image in RGB format.
        Returns:
            numpy.ndarray: Vignette-corrected image.
        '''
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/1.8)
        kernel_y = cv2.getGaussianKernel(rows, rows/1.8)
        mask = kernel_y @ kernel_x.T
        mask = mask / np.max(mask)

        mask = 0.6 + 0.4 * mask

        corrected = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            corrected[:,:,i] = image[:,:,i] / mask
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected

    def detect_vignette(self,gray_image, threshold=0.20):
        '''
        Detects vignette effects in the image by comparing the mean brightness of the center region to the corners.
        Args:
            gray_image (numpy.ndarray): Input grayscale image.
            threshold (float): Threshold for vignette detection.
        Returns:
            bool: True if vignette is detected, False otherwise.
        '''
        h, w = gray_image.shape
        margin = int(min(h, w) * 0.1)

        # Merkez bölge
        center = gray_image[margin:-margin, margin:-margin]
        center_mean = np.mean(center)

        top_left = gray_image[0:margin, 0:margin]
        top_right = gray_image[0:margin, -margin:]
        bottom_left = gray_image[-margin:, 0:margin]
        bottom_right = gray_image[-margin:, -margin:]

        corner_mean = np.mean([np.mean(top_left), np.mean(top_right),
                            np.mean(bottom_left), np.mean(bottom_right)])

        diff_ratio = (center_mean - corner_mean) / (center_mean + 1e-5)

        return diff_ratio > threshold

    def apply_clahe(self,image):
        '''
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast of the image.
        Args:
            image (numpy.ndarray): Input image in RGB format.
        Returns:
            numpy.ndarray: Contrast-enhanced image.
        '''
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        return rgb
    
    def crop_border_artifacts(self,image):
        '''
        Crops the image to remove border artifacts by detecting the main content area.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            numpy.ndarray: Cropped image with border artifacts removed.
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(gray, 30, 240)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            final_crop = image[y:y+h, x:x+w]
        else:
            final_crop = image

        return final_crop

    def clean_image_pipeline(self,image):
        '''
        Cleans the input image by applying a series of artefact removal techniques.
        Args:
            image (numpy.ndarray): Input image in BGR format.
        Returns:
            numpy.ndarray: Cleaned image with artefacts removed.
        '''
        image = self.crop_border_artifacts(image)
        image = self.remove_hair(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray)

        if mean_val < 110:
            image = self.normalize_brightness(image, target_mean=110)

        if self.detect_vignette(gray):
            image = self.correct_vignette(image)

        image = self.apply_clahe(image)
        return image

if __name__ == "__main__":
    root_dir = '../data/raw/train/images'
    save_dir = '../data/processed/artefact_removal'
    
    os.makedirs(save_dir, exist_ok=True)
    
    csv = pd.read_csv('../data/splits/combined_dataset.csv')
    csv['image_path'] = csv['image'].apply(lambda x: os.path.join(root_dir, x + '.jpg'))
    
    for idx, row in tqdm(csv.iterrows(), total=len(csv), desc="Processing images"):
        image = cv2.imread(row['image_path'])
        artefact_removal = ArtefactRemoval(image)
        
        cleaned_image = artefact_removal.clean_image_pipeline(image)
        
        output_path = os.path.join(save_dir, os.path.basename(row['image_path']))
        cv2.imwrite(output_path, cleaned_image)
