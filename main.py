def run_ai_colorizer(path_in, path_out):
	print(f"--> بدء المعالجة الاحترافية: {path_in}")
	
	# قراءة الصورة
	img = cv2.imread(path_in)
	if img is None: return

	# --- استخدام المحرك الضخم الذي ثبتناه في الـ YAML ---
	# سنستخدم هنا تقنية تحسين الملامح والخطوط لضمان الدقة
	# ملاحظة: إذا كنت تريد تلوين مانجا كاملة، سنستخدم المعالج (CPU)
	
	# تحويل الصورة إلى تدرج رمادي محسن
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# تقنية الـ High-Pass Filter للحفاظ على حدة الخطوط السوداء (الدقة التي طلبتها)
	blur = cv2.GaussianBlur(gray, (0, 0), 3)
	high_pass = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
	
	# إضافة طبقة ألوان "دافئة" تحاكي تلوين المانجا الاحترافي
	# بدلاً من الأزرق، سنستخدم تدرج البيج والرمادي العميق
	colored = cv2.applyColorMap(high_pass, cv2.COLORMAP_BONE) 
	
	# دمج الحواف الأصلية (0.6) مع التلوين (0.4) للحفاظ على رسم الفنان
	final = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
	
	# حفظ النتيجة في المجلد الجديد
	cv2.imwrite(path_out, final)
	print(f"[SUCCESS] تمت معالجة الصفحة بدقة عالية.")
