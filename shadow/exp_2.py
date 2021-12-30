import cv2
import cv2utils as u
import config

img_path = config.BASE_DIR / "ISTD/Img_shadow/2-1.png"

img = cv2.imread(str(img_path))
s = img.shape


# Rectangle description
start, end, color = (200, 300), (40, 60), (0, 0, 0) 
alpha = 0.25

overlay = img.copy()
output = img.copy()

overlay = cv2.rectangle(overlay, start, end, color, thickness=-1)

output = cv2.addWeighted(
    overlay,
    alpha,
    output,
    1 - alpha,
    0,
    output
)

# u.view(img, output)

cv2.imwrite("trial-2-1.png", output)
