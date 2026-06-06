import os
import urllib.request
import zipfile
import cv2
import pandas as pd
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

DATA_DIR = './data/isic2016'
IMAGES_ZIP_URL = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_Data.zip'
CSV_URL = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv'

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    zip_path = os.path.join(DATA_DIR, 'ISBI2016_ISIC_Part3B_Training_Data.zip')
    csv_path = os.path.join(DATA_DIR, 'ISBI2016_ISIC_Part3B_Training_GroundTruth.csv')
    extract_dir = os.path.join(DATA_DIR, 'ISBI2016_ISIC_Part3B_Training_Data')
    if not os.path.exists(csv_path):
        print(f"Downloading Ground Truth to {csv_path}...")
        urllib.request.urlretrieve(CSV_URL, csv_path)
    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print(f"Downloading Images to {zip_path}...")
            urllib.request.urlretrieve(IMAGES_ZIP_URL, zip_path)
        
        print(f"Extracting Images to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    return extract_dir, csv_path

def load_isic_data(img_size=(224, 224)):
    """
    Downloads/loads the ISIC 2016 Part 3B dataset.
    Returns:
        A dict {'image': ndarray, 'label': ndarray} containing 900 images.
    """
    extract_dir, csv_path = download_and_extract()
    
    # Read CSV
    # Usually the ISIC 2016 Part 3B CSV has two columns, no header:
    # ISIC_0000000, 0.0 (or similar)
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return None
    
    images = []
    labels = []
    
    print("Loading and resizing images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row.iloc[0])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = os.path.join(extract_dir, img_name)
        
        if not os.path.exists(img_path):
            # Sometimes inside a subfolder
            img_path = os.path.join(extract_dir, 'ISBI2016_ISIC_Part3B_Training_Data', img_name)
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, img_size)
        
        # Normalize with ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        images.append(img)
        
        # Determine label (0 for benign, 1 for malignant)
        val = row.iloc[1]
        if isinstance(val, str):
            label = 1 if 'malignant' in val.lower() else 0
        else:
            label = int(val)
        labels.append(label)
        
    return {
        'image': jnp.array(images),
        'label': jnp.array(labels)
    }

def get_isic_splits(dataset, seed=42):
    """
    Splits the dataset into a balanced test set (100 pos, 100 neg)
    and returns the remainder as the training pool.
    
    Returns:
        test_set: dict
        pool: dict
    """
    labels = np.array(dataset['label'])
    images = np.array(dataset['image'])
    
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    rng = np.random.default_rng(seed)
    
    test_pos_idx = rng.choice(pos_idx, size=100, replace=False)
    test_neg_idx = rng.choice(neg_idx, size=100, replace=False)
    
    test_idx = np.concatenate([test_pos_idx, test_neg_idx])
    
    mask = np.ones(len(labels), dtype=bool)
    mask[test_idx] = False
    
    test_set = {
        'image': jnp.array(images[test_idx]),
        'label': jnp.array(labels[test_idx])
    }
    
    pool = {
        'image': jnp.array(images[mask]),
        'label': jnp.array(labels[mask])
    }
    
    return test_set, pool
