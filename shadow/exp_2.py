import cv2
import cv2utils as u
import config

img_name = "5-1"
img_path = config.BASE_DIR / f"ISTD/Img_shadow/{img_name}.png"

img = cv2.imread(str(img_path))
s = img.shape


# Rectangle description
start, end, color = (200, 300), (40, 60), (0, 0, 0)
alpha = 0.25

overlay = img.copy()
output = img.copy()

overlay = cv2.rectangle(overlay, start, end, color, thickness=-1)

output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

cv2.imwrite(f"./output/transparent-overlay-{img_name}.png", output)
