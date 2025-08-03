import cv2
import numpy as np
import matplotlib.pyplot as plt

class ContourBoardDetector():
    def __init__(self, settings):
        self.settings = settings

    # Draw Functions
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

    # Detection Helper Functions
    def apply_edge_detection(self, image):
        T_LOWER, T_UPPER = 30, 100
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blurred, T_LOWER, T_UPPER)
        return edge

    def has_grid_pattern(self, edges):
        # Heuristic check (is there a board?)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=10)
    
        # Check that a good number of vertical + horizontal lines are present
        if lines is None or len(lines) < 10:
            return False
    
        vertical = 0
        horizontal = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi)
            if angle < 10:  # near-horizontal
                horizontal += 1
            elif angle > 80:  # near-vertical
                vertical += 1
    
        return vertical >= 6 and horizontal >= 6

    def find_squares(self, contours):
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
    
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
    
            if w == 0 or h == 0:
                continue
    
            # Test square aspect ratio with 10% error
            aspect_ratio = max(w, h) / min(w, h)
            if 0.9 < aspect_ratio < 1.1:
                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)
                candidates.append((box, area))
    
        return candidates

    def return_largest_contour(self, contour_candidates):
        largest_area = 0
        largest_box = None
        for box, area in contour_candidates:
            if area > largest_area:
                largest_area = area
                largest_box = box
        return largest_box

    def detect(self, image):
        edges = self.apply_edge_detection(image)
    
        if not self.has_grid_pattern(edges):
            self.show_image(image, "No grid")
            print("No grid pattern found.")
            return None
    
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.show_image(image, "No Contours")
            print("No contours found.")
            return None

        candidates = self.find_squares(contours)
        largest = self.return_largest_contour(candidates)

        self.draw_contours(image, [largest])
        return largest