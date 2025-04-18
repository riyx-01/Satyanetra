
import cv2
import numpy as np
import os

# Load image
image_path = r"C:\Users\riyat\Downloads\foot.png"  # <- Change to your image path
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

clone = img.copy()
mask = np.zeros(img.shape[:2], dtype=np.uint8)
drawing = False

# Mouse callback for drawing
def draw_mask(event, x, y, flags, param):
    global drawing, mask, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), 10, 255, -1)
        cv2.circle(clone, (x, y), 10, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Setup window and mouse callback
cv2.namedWindow("Draw to Remove Object")
cv2.setMouseCallback("Draw to Remove Object", draw_mask)

print("✏️  Instructions:")
print(" - Draw on the object using the mouse")
print(" - Press 'd' to remove the object")
print(" - Press 'r' to reset the drawing")
print(" - Press 'q' to quit without saving")

while True:
    cv2.imshow("Draw to Remove Object", clone)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        clone = img.copy()

    elif key == ord('q'):
        print("❌ Quit without saving.")
        break

    elif key == ord('d'):
        inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imshow("Inpainted Result", inpainted)

        # Save the result
        output_dir = os.path.join(os.getcwd(), "removed_object_result.png")
        saved = cv2.imwrite(output_dir, inpainted)
        if saved:
            print(f"✅ Object removed. Image saved at:\n{output_dir}")
        else:
            print("❌ Failed to save image.")
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()
