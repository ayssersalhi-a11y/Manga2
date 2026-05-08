import os
import cv2
import numpy as np

INPUT_DIR = "input_manga"
OUTPUT_DIR = "colored_results"

def setup():
	os.makedirs(INPUT_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ai_colorizer(path_in, path_out):
	# قراءة الصورة الأصلية
	img = cv2.imread(path_in)
	if img is None: return

	# تحويل لرمادي لانتزاع تفاصيل الظلال
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# الخطوة الضخمة: تطبيق تلوين احترافي بناءً على مستويات العمق
	# COLORMAP_JET يعطي تلوين كامل (أحمر، أزرق، أصفر) بناءً على الظلال
	# إذا أردت ألواناً "أهدأ" مثل المانجا الواقعية غيرها إلى COLORMAP_DEEPGREEN
	colored_layer = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
	
	# دمج "الرسم الأصلي" مع "طبقة الألوان"
	# 0.5 للرسم الأصلي يحافظ على سواد الخطوط
	# 0.5 للألوان يملأ المساحات البيضاء
	result = cv2.addWeighted(img, 0.5, colored_layer, 0.5, 0)
	
	# حفظ النتيجة النهائية
	cv2.imwrite(path_out, result)
	print(f"[+] تم التلوين والحفظ: {path_out}")

def main():
	setup()
	valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
	files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]
	
	if not files:
		print("[!] لا توجد صور في المجلد.")
		return

	for f in files:
		in_p = os.path.join(INPUT_DIR, f)
		out_p = os.path.join(OUTPUT_DIR, f"colored_{f}")
		run_ai_colorizer(in_p, out_p)

if __name__ == "__main__":
	main()
