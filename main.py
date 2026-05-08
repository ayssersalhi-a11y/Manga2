import os
import torch
import cv2
import numpy as np

# إعداد المجلدات
INPUT_FOLDER = "input_manga"
OUTPUT_FOLDER = "colored_manga"

def prepare_folders():
	if not os.path.exists(INPUT_FOLDER):
		os.makedirs(INPUT_FOLDER)
		print(f"تم إنشاء مجلد المدخلات: {INPUT_FOLDER}. ضع صورك فيه.")
	if not os.path.exists(OUTPUT_FOLDER):
		os.makedirs(OUTPUT_FOLDER)

def colorize_image(image_path, output_path):
	# هنا سنقوم باستدعاء النموذج لاحقاً 
	# حالياً سنضع منطقاً بسيطاً للتأكد من أن الأكشن يعمل
	print(f"جاري معالجة: {image_path}...")
	
	img = cv2.imread(image_path)
	if img is None:
		print("خطأ: لا يمكن قراءة الصورة.")
		return

	# تجربة: تحويل الصورة لدرجات رمادية ثم إضافة فلتر لوني (للتأكد من العمل)
	# في الخطوة القادمة سنضع كود الذكاء الاصطناعي الحقيقي هنا
	colored = cv2.applyColorMap(img, cv2.COLORMAP_JET)
	
	cv2.imwrite(output_path, colored)
	print(f"تم حفظ الصورة الملونة في: {output_path}")

def main():
	prepare_folders()
	
	images = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
	
	if not images:
		print("لا توجد صور في مجلد input_manga.")
		return

	for img_name in images:
		in_path = os.path.join(INPUT_FOLDER, img_name)
		out_path = os.path.join(OUTPUT_FOLDER, img_name)
		colorize_image(in_path, out_path)

if __name__ == "__main__":
	main()
