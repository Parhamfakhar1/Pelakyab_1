# کتابخانه‌های مورد نیاز
import cv2
import pytesseract
import os
import numpy as np
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm

# تنظیم مسیر Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(blur)
    return enhanced

def detect_plate(image):
    edged = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 6 and w > 100:
            plate_candidate = image[y:y + h, x:x + w]
            text = pytesseract.image_to_string(plate_candidate, config='--psm 8')
            if len(text.strip()) >= 6:
                return text.strip()
    return None

def process_images(folder_path):
    plate_file = open('plak.txt', 'w')
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="در حال پردازش تصاویر"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        enhanced_image = enhance_image(image)
        plate_number = detect_plate(enhanced_image)

        if plate_number:
            plate_file.write(f'{filename}: {plate_number}\n')

        os.remove(image_path)  # حذف تصویر پردازش شده

    plate_file.close()
    messagebox.showinfo("پایان پردازش", "پردازش تصاویر به پایان رسید و نتایج در فایل 'plak.txt' ذخیره شد.")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="پوشه تصاویر را انتخاب کنید")

    if folder_selected:
        process_images(folder_selected)
    else:
        messagebox.showwarning("عدم انتخاب پوشه", "پوشه‌ای انتخاب نشد.")
