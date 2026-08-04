import time
import cv2
from paddleocr import PaddleOCR

# ==========================================
# LOAD OCR MODEL
# ==========================================

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("OCR Loaded.\n")

# ==========================================
# LOAD IMAGE
# ==========================================

image_path = "test.jpg"      # <-- Put any image containing text here

frame = cv2.imread(image_path)

if frame is None:
    raise Exception(f"Cannot open image: {image_path}")

print("Image Shape:", frame.shape)

# ==========================================
# OPTIONAL RESIZE
# ==========================================

# Uncomment if you want to test resized image
#
# frame = cv2.resize(
#     frame,
#     None,
#     fx=0.5,
#     fy=0.5,
#     interpolation=cv2.INTER_AREA
# )

# ==========================================
# WARMUP
# ==========================================

print("Running warmup...")

ocr.predict(frame)

# ==========================================
# BENCHMARK
# ==========================================

times = []

for i in range(10):

    start = time.perf_counter()

    result = ocr.predict(frame)

    elapsed = time.perf_counter() - start

    times.append(elapsed)

    print(f"Run {i+1}: {elapsed:.3f} sec")

print("\n====================================")
print(f"Average : {sum(times)/len(times):.3f} sec")
print(f"Minimum : {min(times):.3f} sec")
print(f"Maximum : {max(times):.3f} sec")
print("====================================")

# ==========================================
# PRINT DETECTED TEXT
# ==========================================

for page in result:

    data = page.json

    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    print("\nDetected Text")

    for t, s in zip(texts, scores):
        print(f"{t} ({s:.2f})")