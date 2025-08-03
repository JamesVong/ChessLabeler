import cv2
import matplotlib.pyplot as plt

class PlotDrawer():
    def __init__(self):
        pass
    
    def show_image(self, image, title='Image', cmap=None):
        plt.figure(figsize=(6, 6))
        if image.ndim == 2:  # grayscale
            plt.imshow(image, cmap=cmap or 'gray')
        else:  # color
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def draw_contours(self, image, contours):
        image_copy = image.copy()
        for c in contours:
            cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 2)
        self.show_image(image_copy, "Contours")
    
    def draw_lines_on_image(self, image, lines):
        image_with_lines = image.copy()
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_image(image_with_lines, "Detected Lines")

    def draw_box(self,
                 image,
                 top_left: tuple[int, int],
                 bottom_right: tuple[int, int],
                 color: tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 title: str = "Detected Template"):
        img_copy = image.copy()
        cv2.rectangle(img_copy, top_left, bottom_right, color, thickness)
        self.show_image(img_copy, title)