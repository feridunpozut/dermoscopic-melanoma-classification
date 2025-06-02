'''
ISIC API Download Script
'''
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import time
from combine_dataset import combine_isic_datasets

API_BASE_URL = "https://api.isic-archive.com/api/v2/images"
IMAGES_DIR = "../data/raw/images"

def create_images_directory():
    """
    Create the images directory if it does not exist.
    """
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        print(f"'{IMAGES_DIR}' dizini oluşturuldu.")

def download_image(image_id):
    """
    Download a single image from the ISIC API.
    Args:
        image_id (str): The ID of the image to download.
        Returns:
        bool: True if the image was downloaded successfully, False otherwise.
    """
    try:
        image_url = f"{API_BASE_URL}/{image_id}"
        
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        else:
            ext = '.jpg'
        
        filename = f"{image_id}{ext}"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        image_url = response.json()['files']['full']['url']
        byte_img = requests.get(image_url, timeout=30)
        with open(filepath, 'wb') as f:
            f.write(byte_img.content)
        
        print(f"✓ {filename} indirildi")
        return True
        
    except Exception as e:
        print(f"✗ {image_id} indirilemedi: {str(e)}")
        return False

def download_images_parallel(image_ids, max_workers=10):
    """
    Download images in parallel using multiple threads.
    Args:
        image_ids (list): List of image IDs to download.
        max_workers (int): Maximum number of threads to use for downloading.
    """
    create_images_directory()
    
    print(f"{len(image_ids)} görsel indiriliyor...")
    start_time = time.time()
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_image, image_ids))
        success_count = sum(results)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nİndirme tamamlandı!")
    print(f"Başarılı: {success_count}/{len(image_ids)}")
    print(f"Süre: {duration:.2f} saniye")

if __name__ == "__main__":
    image_ids, _, _ = combine_isic_datasets()
    image_ids = image_ids.tolist()
    if not image_ids:
        print("Lütfen image_ids listesine görsel ID'lerinizi ekleyin!")
    else:
        download_images_parallel(image_ids)
