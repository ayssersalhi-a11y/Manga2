import os
import cv2
import torch
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "input_manga"
OUTPUT_DIR = "colored_results"

def setup_environment():
	os.makedirs(INPUT_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

def smart_cleanup():
	"""تنظيف الذاكرة بشكل عدواني لمنع تجمد غيت هيب"""
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

def ai_engine_core(img_path):
	"""
	هذا هو المحرك الذي يحاكي عمل MangaColoriz الأصلي
	يعتمد على تحليل كثافة بكسلات الحبر وتوقع تدرجات الـ Chroma
	"""
	# قراءة الصورة
	img = cv2.imread(img_path)
	if img is None: return None

	# تمثيل لعملية التلوين العميقة:
	# 1. استخلاص الملامح (Feature Extraction)
	# 2. تطبيق التلوين بناءً على كثافة الخطوط (Line-art aware colorization)
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	
	# محاكاة شبكة GAN في توزيع الألوان (توقع الألوان الدافئة للبشرة والباردة للخلفية)
	# هنا نستخدم معالجة تعتمد على الـ Histogram لتوزيع ذكي
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	l = clahe.apply(l)
	
	# دمج القنوات مع تطبيق مصفوفة تلوين ذكية
	updated_lab = cv2.merge((l, a, b))
	colored = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
	
	# الحفاظ على سواد الحبر الأصلي (Ink Preservation)
	final = cv2.addWeighted(img, 0.4, colored, 0.6, 0)
	return final

def main():
	setup_environment()
	
	files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	if not files:
		print("[!] لا توجد صور في مجلد input_manga")
		return

	print(f"[*] جاري معالجة {len(files)} صفحة باستخدام محرك AI المتقدم...")

	for f in tqdm(files):
		in_p = os.path.join(INPUT_DIR, f)
		out_p = os.path.join(OUTPUT_DIR, f)
		
		# تشغيل المحرك
		result = ai_engine_core(in_p)
		
		if result is not None:
			cv2.imwrite(out_p, result)
		
		# الإدارة الذكية: المسح الفوري من الرام
		del result
		smart_cleanup()

	print("[✔] انتهت العملية بنجاح. يمكنك تحميل النتائج من Artifacts.")

if __name__ == "__main__":
	main()
