"""
Watermark embedding and detection demo.

Requires scikit-image (pip install scikit-image).
"""
import matplotlib.pyplot as plot

from skimage import data, color

from util.watermark import embed_watermark, detect_watermark, jpeg_compress


if __name__ == "__main__":
    # Use a sample grayscale image
    img = color.rgb2gray(data.astronaut())
    img = img[50:306, 50:306]  # crop to square
    img = (img - img.min()) / (img.max() - img.min())

    img_w, wm = embed_watermark(img, strength=8.0, seed=123)
    score_clean = detect_watermark(img_w, wm)

    # Robustness sweep: JPEG quality
    qualities = [90,70,50,30,10]
    scores = []

    for q in qualities:
        comp = jpeg_compress(img_w, quality=q)
        scores.append(detect_watermark(comp, wm))

    print("Clean score:", score_clean)

    for q,s in zip(qualities, scores):
        print(f"Quality {q}: score {s:.2f}")

    # Visualize
    plot.figure(figsize=(10,3))
    plot.subplot(1,3,1)
    plot.imshow(img, cmap="gray")
    plot.title("Original")
    plot.axis("off")
    plot.subplot(1,3,2)
    plot.imshow(img_w, cmap="gray")
    plot.title("Watermarked")
    plot.axis("off")
    plot.subplot(1,3,3)
    plot.plot(qualities, scores, marker="o")
    plot.gca().invert_xaxis()
    plot.title("Detection score vs JPEG quality")
    plot.xlabel("JPEG quality (lower worse)")
    plot.tight_layout()
    plot.show()
