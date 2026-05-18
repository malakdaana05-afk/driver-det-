import cv2
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224

def preprocess_image(image):
    # تحويل الصورة إلى RGB
    image = image.convert("RGB")

    # PIL -> NumPy
    img_np = np.array(image)

    # ======================================
    # CLAHE Enhancement
    # ======================================
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # RGB -> LAB
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    # تقسيم القنوات
    l, a, b = cv2.split(lab)

    # تطبيق CLAHE على قناة الإضاءة
    l_clahe = clahe.apply(l)

    # دمج القنوات
    lab_clahe = cv2.merge((l_clahe, a, b))

    # LAB -> RGB
    img_np = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # NumPy -> PIL
    image = Image.fromarray(img_np)

    # ======================================
    # Letterbox Resize
    # ======================================
    image = ImageOps.pad(
        image,
        (IMG_SIZE, IMG_SIZE),
        color=(0, 0, 0)
    )

    # ======================================
    # Normalize
    # ======================================
    image_array = np.array(image).astype(np.float32) / 255.0

    # إضافة Batch Dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
