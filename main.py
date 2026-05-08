import os
import torch
import cv2
import numpy as np

# إعداد المسارات - مجلدات منفصلة تماماً
INPUT_DIR = "input_manga"   # هنا تضع فصول المانجا الأصلية
OUTPUT_DIR = "colored_results" # هنا سيضع المحرك النتائج الملونة

def setup_environment():
	# إنشاء المجلدات إذا لم تكن موجودة
	if not os.path.exists(INPUT_DIR):
		os.makedirs(INPUT_DIR)
		print(f"[*] تم إنشاء مجلد المدخلات: {INPUT_DIR}")
	
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)
		print(f"[*] تم إنشاء مجلد المخرجات: {OUTPUT_DIR}")

def process_manga_list():
	setup_environment()
	
	# جلب قائمة الصور من مجلد المدخلات فقط
	image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
	files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)]
	
	if not files:
		print("[!] لا توجد صور في مجلد المدخلات. يرجى رفع الصور في 'input_manga'.")
		return

	print(f"[*] تم العثور على {len(files)} صفحة. بدء التلوين...")

	for filename in files:
		input_path = os.path.join(INPUT_DIR, filename)
		output_path = os.path.join(OUTPUT_DIR, f"colored_{filename}")
		
		# استدعاء دالة التلوين
		run_ai_colorizer(input_path, output_path)
		
		# تنظيف الذاكرة بعد كل صورة لراحة المحرك
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

def run_ai_colorizer(path_in, path_out):
	print(f"--> جاري تلوين: {path_in}")
	
	# تحميل الصورة
	img = cv2.imread(path_in)
	if img is None: return

	# --- هنا سنضع كود مكتبة المانجا الحقيقي في الخطوة القادمة ---
	# حالياً سنقوم بعمل تأثير لوني للتأكد من أن الملفات تنتقل للمجلد الجديد
	result = cv2.applyColorMap(img, cv2.COLORMAP_DEEPGREEN) 
	
	# حفظ النتيجة في المجلد المنفصل
	cv2.imwrite(path_out, result)
	print(f"[OK] تم الحفظ في: {path_out}")

if __name__ == "__main__":
	process_manga_list()
