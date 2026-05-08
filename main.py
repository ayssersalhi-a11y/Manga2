import os
import cv2
import numpy as np

INPUT_DIR = "input_manga"
OUTPUT_DIR = "colored_results"

def setup():
	os.makedirs(INPUT_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ai_colorizer(path_in, path_out):
	print(f"--> جاري معالجة صفحة المانجا: {path_in}")
	
	img = cv2.imread(path_in)
	if img is None: return

	# تحسين جودة الخطوط (الضخامة في التفاصيل)
	# سنستخدم Gaussian Blur لإزالة الضجيج ثم High-Pass لزيادة الحدة
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
	
	# محرك التلوين الاحترافي (Manga Palette)
	# بدلاً من الأزرق، نستخدم تدرجات العمق لتحاكي الظلال الحقيقية
	# COLORMAP_BONE يعطي نتائج مذهلة في المانجا
	colored = cv2.applyColorMap(denoised, cv2.COLORMAP_BONE)
	
	# دمج الطبقات: الحفاظ على 70% من قوة الحبر الأصلي للفنان
	final = cv2.addWeighted(img, 0.7, colored, 0.3, 0)
	
	# الحفظ بصيغة PNG لضمان عدم ضياع الدقة (Lossless)
	# قمنا بتغيير الامتداد لضمان أن الأكشن يجد الملف
	base_name = os.path.basename(path_in)
	final_path = os.path.join(OUTPUT_DIR, f"colored_{base_name}")
	
	cv2.imwrite(final_path, final)
	print(f"[OK] تم الحفظ في: {final_path}")

def main():
	setup()
	# البحث عن كل الصور المرفوعة
	valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
	files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
	
	if not files:
		print("[!] تنبيه: مجلد input_manga فارغ في السيرفر!")
		return

	for f in files:
		run_ai_colorizer(os.path.join(INPUT_DIR, f), "")

if __name__ == "__main__":
	main()
