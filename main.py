from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

image = Image.open("lenna.png")
print(type(image))

#원본 이미지 출력
plt.imshow(image)
plt.title("Original")
plt.show()
#좌우 반전 이미지 출력
image_reverse = image.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(image_reverse)
plt.title("Left and right reverse")
plt.show()
#180도 회전 이미지 출력
image_rotate = image.transpose(Image.ROTATE_180)
plt.imshow(image_rotate)
plt.title("180 rotation")
plt.show()
#가로 세로 길이가 2배로 축소된 이미지 출력
image_small = image.resize((int(image.size[0]/2),int(image.size[1]/2)))
plt.imshow(image_small)
plt.title("Small original")
plt.show()