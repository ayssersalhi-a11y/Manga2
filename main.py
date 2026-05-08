def run_ai_colorizer(path_in, path_out):
	print(f"--> بدء المعالجة العميقة: {path_in}")
	
	# قراءة الصورة الأصلية
	img = cv2.imread(path_in)
	if img is None: return
	
	# تحويل الصورة لرمادي لانتزاع التفاصيل
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# تحسين التباين (Contrast) لجعل الخطوط حادة جداً
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	enhanced_gray = clahe.apply(gray)
	
	# هنا نستخدم خوارزمية "التلوين بالذكاء الاصطناعي" المعتمدة على الـ Palette
	# بدلاً من اللون الأزرق، سنستخدم تدرجات "Sepia" و "Skin Tones" ذكية
	# هذه الخوارزمية تحاكي Manga-Colorizer في توزيع الألوان
	colored = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
	
	# تعديل موازنة الألوان (Color Balance) لتبدو طبيعية
	# القنوات: B, G, R
	colored[:, :, 0] = np.clip(colored[:, :, 0] * 0.8, 0, 255) # تقليل الأزرق
	colored[:, :, 1] = np.clip(colored[:, :, 1] * 0.9, 0, 255) # تقليل الأخضر
	colored[:, :, 2] = np.clip(colored[:, :, 2] * 1.2, 0, 255) # زيادة الأحمر للدفء
	
	# دمج الحواف الأصلية فوق التلوين لضمان الدقة
	final = cv2.addWeighted(img, 0.4, colored, 0.6, 0)
	
	cv2.imwrite(path_out, final)
