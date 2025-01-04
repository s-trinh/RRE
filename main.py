import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

# filename=sys.argv[1]
filename = "img.png"
filename_we = filename.split('.')[0]

print(f"filename: {filename} ; filename_we: {filename_we}")
src = cv2.imread(filename)

# mask input
radius = 512/2
for i in range(src.shape[0]):
    v = i - 256
    for j in range(src.shape[1]):
        u = j - 256
        dist = np.sqrt(u**2 + v**2)
        if dist >= 255:
            src[i,j] = (0,0,0)
cv2.imwrite(f"{filename_we}_masked.png", src)

src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
print(src.shape)
dsize = [512, 512]
dsize2 = [512, int(512*np.pi)]
center = [256, 256]
center2 = [256, 256]
maxRadius = 256
base_flags =   cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS

radials = ["cv2.WARP_POLAR_LINEAR",
           "cv2.WARP_POLAR_LOG",
           "cv2.WARP_POLAR_EXP",
           "cv2.WARP_POLAR_SQRT",
           "cv2.WARP_POLAR_SQUARE"]

for i in range(5):
    flags = base_flags+eval(radials[i])
    radial = radials[i].split('_')[-1]

    dst = cv2.warpPolar(src, dsize2, center, maxRadius, flags)
    cv2.imwrite(f"{filename_we}_{radial}.png", dst)

    flags = base_flags+eval(radials[i]) + cv2.WARP_INVERSE_MAP
    rec = cv2.warpPolar(dst, dsize, center2, maxRadius, flags)

    error_ssd = (rec.astype(np.float64) - src.astype(np.float64))**2
    error_ssd = np.mean(error_ssd, axis=2)
    error_ssd_mean = error_ssd.mean()
    print(f"error_ssd_mean={error_ssd_mean}")

    # error_ssd = np.sum(error**2, axis=0)
    print(f"radial={radial} ; error_ssd={error_ssd.shape}")
    error_ssd_uint8 = cv2.normalize(error_ssd, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # error_ssd_uint8_bgr = cv2.cvtColor(error_ssd_uint8, cv2.COLOR_RGB2BGR)
    error_ssd_uint8_bgr = cv2.applyColorMap(error_ssd_uint8, cv2.COLORMAP_JET)
    error_ssd_uint8_filename = f"error_ssd_{radial}.png"
    print(f"error_ssd_uint8_filename={error_ssd_uint8_filename}")
    cv2.imwrite(error_ssd_uint8_filename, error_ssd_uint8_bgr)

    # error_ssd_uint8_gray = cv2.cvtColor(error_ssd_uint8, cv2.COLOR_RGB2GRAY)

    plt.subplot(4, 5, i+1)
    plt.imshow(src)
    plt.title('ori ' + radial)
    plt.axis('off')

    plt.subplot(4, 5, i+1+5)
    plt.imshow(dst)
    plt.title('polar ' + radial)
    plt.axis('off')

    plt.subplot(4, 5, i+1+10)
    plt.imshow(rec)
    plt.title('rec ' + radial)
    plt.axis('off')

    plt.subplot(4, 5, i+1+15)
    # plt.imshow(error_ssd_uint8)
    plt.imshow(error_ssd_uint8, cmap="jet")
    text_kwargs = dict(ha='center', va='center', fontsize=6, color='C1')
    plt.text(0.1, 0.1, f"Mean error: {error_ssd_mean:.2f}", **text_kwargs)
    plt.title('rec ' + radial)
    plt.axis('off')

plt.tight_layout()
# plt.show()
plt.savefig(filename.split('.')[0]+'_res.png', dpi=1200)
