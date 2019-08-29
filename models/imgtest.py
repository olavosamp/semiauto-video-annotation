import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

matplotlib.use("TkAgg")

imPath = "..\\datasets\\torch\\hymenoptera_data\\train\\ants\\0013035.jpg"
img = mpimg.imread(imPath)
# print(img)
plt.imshow(img)
# plt.show()