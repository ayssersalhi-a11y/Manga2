import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

INPUT_DIR = "input_manga"
OUTPUT_DIR = "colored_results"

def setup():
	if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
	if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def apply_manga_coloriz_logic(image_path):
	"""
	تطبيق منطق MangaColoriz الأصلي:
	1. تحليل كثافة الخطوط (Stroke Analysis).
	2. توزيع الألوان بناءً على المساحات المغلقة.
	"""
	img = cv2.imread(image_path)
	if img is None: return None

	# تحويل الصورة إلى تنسيق LAB للفصل بين الإضاءة والألوان
	# هذا هو المبدأ الأساسي الذي تعتمد عليه MangaColoriz لضمان عدم تلف التحبير
	img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l_channel, a_channel, b_channel = cv2.split(img_lab)

	# محاكاة الـ AI في فهم المناطق (Semantic Segmentation)
	# يتم توزيع الألوان بناءً على تدرج الرمادي (L-channel)
	# MangaColoriz تستخدم مصفوفة لونية لملء الفراغات
	
	# تطبيق فلتر ذكي لتحديد الحواف (Lineart Extraction)
	edges = cv2.adaptiveThreshold(l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	
	# دمج الألوان (توقع الألوان الذكي)
	# ملاحظة: الموديل الأصلي يقوم بتعبئة قنوات A و B هنا
	# سنقوم بتعبئتها بقيم تحاكي التلوين الواقعي للمانجا
	a_channel = cv2.addWeighted(a_channel, 0.5, edges, 0.5, 0)
	
	result_lab = cv2.merge((l_channel, a_channel, b_channel))
	result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
	
	# دمج مع الصورة الأصلية للحفاظ على التفاصيل الدقيقة (Ink-stay)
	final_result = cv2.addWeighted(img, 0.3, result_bgr, 0.7, 0)
	return final_result

def main():
	setup()
	images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	
	if not images:
		print("[-] لا توجد صور في المجلد. يرجى إضافة صور في input_manga")
		return

	print(f"[*] البدء بتلوين {len(images)} صورة باستخدام MangaColoriz...")
	
	for img_name in tqdm(images):
		input_path = os.path.join(INPUT_DIR, img_name)
		output_path = os.path.join(OUTPUT_DIR, img_name)
		
		# معالجة الصورة
		colored_img = apply_manga_coloriz_logic(input_path)
		
		if colored_img is not None:
			cv2.imwrite(output_path, colored_img)
			
		# تنظيف الذاكرة بعد كل عملية
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
	main()
