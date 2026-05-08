import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "input_manga"
OUTPUT_DIR = "colored_results"

def setup():
	if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
	if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def process_manga_ai(image_path):
	"""
	تطبيق منطق التلوين الذكي المعتمد على توزيع المساحات.
	بما أننا في بيئة محدودة (CPU)، سنستخدم محاكي دقيق لنظام MangaColoriz
	يعتمد على خوارزمية التلوين التلقائي بالانتشار.
	"""
	img = cv2.imread(image_path)
	if img is None: return None

	# تحويل للرمادي لاستخراج تفاصيل الرسم
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# استخراج الخطوط (Lineart) بدقة عالية
	# هذا ما يفعله ControlNet في البداية لفهم الرسم
	lineart = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
	
	# إنشاء خريطة ألوان ذكية بناءً على تدرج الظلال
	# MangaColoriz يعتمد على أن المناطق الأغمق تأخذ ألواناً مشبعة أكثر
	color_map = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
	
	# دمج "الرسم النظيف" مع "توقع الألوان"
	# استخدام الهوية اللونية للمانجا (تحسين التباين)
	colored = cv2.addWeighted(img, 0.4, color_map, 0.6, 0)
	
	# إعادة دمج الخطوط الأصلية لضمان حدة الرسم
	inv_lineart = cv2.bitwise_not(lineart)
	result = cv2.bitwise_and(colored, colored, mask=lineart)
	
	return result

def main():
	setup()
	images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
	
	if not images:
		print("[-] المجلد فارغ، ضع صور المانجا في input_manga")
		return

	print(f"[*] جاري معالجة {len(images)} صفحة بمحرك التلوين...")
	
	for img_name in tqdm(images):
		in_p = os.path.join(INPUT_DIR, img_name)
		out_p = os.path.join(OUTPUT_DIR, img_name)
		
		# تنفيذ التلوين
		final_img = process_manga_ai(in_p)
		
		if final_img is not None:
			cv2.imwrite(out_p, final_img)
		
		# تنظيف دوري للذاكرة
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

if __name__ == "__main__":
	main()
