import sys
import numpy as np
from PIL import Image


def to_grayscale(img):
    """RGB görüntüyü gri tonlamaya çevirir."""
    if img.ndim == 3:
        return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return img.astype(np.float64)


def convolve2d(image, kernel):
    """
    2B konvolüsyon.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    flipped = kernel[::-1, ::-1]
    out = np.zeros_like(image, dtype=np.float64)
    h, w = image.shape
    for i in range(kh):
        for j in range(kw):
            out += flipped[i, j] * padded[i:i + h, j:j + w]
    return out



# 1) Gürültü bastırma: 5x5 Gauss filtresi

def gaussian_kernel(size=5, sigma=1.4):
    """size x size boyutunda normalize edilmiş Gauss çekirdeği üretir."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image, size=5, sigma=1.4):
    """Görüntüyü 5x5 Gauss filtresiyle yumuşatır."""
    return convolve2d(image, gaussian_kernel(size, sigma))


# 2) Sobel ile gradyan hesaplama

def sobel_gradients(image):
    """
    Sobel filtreleriyle Gx, Gy hesaplar.
    
    """
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)

    gx = convolve2d(image, kx)
    gy = convolve2d(image, ky)

    magnitude = np.hypot(gx, gy)
    direction = np.rad2deg(np.arctan2(gy, gx)) % 180  # 0-180 aralığı yeterli
    return magnitude, direction



# 3) Non-maximum suppression

def non_maximum_suppression(magnitude, direction):
    """
    Her pikseli gradyan yönündeki iki komşusuyla karşılaştırır.
    """
    h, w = magnitude.shape
    out = np.zeros_like(magnitude)
    padded = np.pad(magnitude, 1, mode="constant")

    for y in range(h):
        for x in range(w):
            angle = direction[y, x]
            py, px = y + 1, x + 1  # dolgulu görüntüdeki konum

            if angle < 22.5 or angle >= 157.5:        # 0°
                n1, n2 = padded[py, px - 1], padded[py, px + 1]
            elif angle < 67.5:                         # 45°
                n1, n2 = padded[py + 1, px - 1], padded[py - 1, px + 1]
            elif angle < 112.5:                        # 90°
                n1, n2 = padded[py - 1, px], padded[py + 1, px]
            else:                                      # 135°
                n1, n2 = padded[py - 1, px - 1], padded[py + 1, px + 1]

            if magnitude[y, x] >= n1 and magnitude[y, x] >= n2:
                out[y, x] = magnitude[y, x]
    return out



# 4) Hysteresis thresholding

def hysteresis_threshold(nms, low, high):
    """
    Çift eşikleme + bağlantı analizi.
    """
    h, w = nms.shape
    strong = nms >= high
    weak = (nms >= low) & ~strong

    edges = np.zeros((h, w), dtype=np.uint8)
    edges[strong] = 255

    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and edges[ny, nx] == 0:
                    edges[ny, nx] = 255
                    stack.append((ny, nx))
    return edges



# Tüm boru hattı

def canny(image, low_ratio=0.05, high_ratio=0.15, sigma=1.4):
    """
    Eşikler, maksimum gradyan şiddetinin oranı olarak verilir; böylece
    farklı görüntülerde aynı değerler makul çalışır.
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, size=5, sigma=sigma)
    magnitude, direction = sobel_gradients(blurred)
    thin = non_maximum_suppression(magnitude, direction)

    max_val = thin.max() if thin.max() > 0 else 1.0
    edges = hysteresis_threshold(thin, low_ratio * max_val, high_ratio * max_val)
    return blurred, magnitude, thin, edges


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python canny.py gorsel.jpg")
        sys.exit(1)

    img = np.array(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64)
    blurred, magnitude, thin, edges = canny(img)

    import matplotlib.pyplot as plt
    titles = ["1) Gauss", "2) Sobel şiddeti", "3) NMS", "4) Hysteresis (sonuç)"]
    images = [blurred, magnitude, thin, edges]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("canny_sonuc.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()