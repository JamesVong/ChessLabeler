import cv2
import numpy as np

from PlotDrawer import PlotDrawer

class ContourBoardDetector():
    def __init__(self, settings):
        self.settings = settings
        self.plotter = PlotDrawer()

    # Detection Helper Functions
    def apply_edge_detection(self, image):
        T_LOWER, T_UPPER = 50, 100
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blurred, T_LOWER, T_UPPER)
        return edge

    def count_grid_lines(self, edges):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=10)
        
        vertical = 0
        horizontal = 0
    
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2((y2 - y1), (x2 - x1)) * 180.0 / np.pi)
                if angle < 10:  # near-horizontal
                    horizontal += 1
                elif angle > 80:  # near-vertical
                    vertical += 1
    
        return vertical, horizontal

    def compute_grid_confidence(self, board_crop, frame_shape, vertical, horizontal, angle) -> float:
        h, w = board_crop.shape[:2]
        frame_h, frame_w = frame_shape
    
        grid_score = min(horizontal / 7.0, 1.0) * min(vertical / 7.0, 1.0)
    
        aspect_ratio = max(w, h) / min(w, h)
        aspect_penalty = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
    
        area_ratio = (w * h) / (frame_w * frame_h)
        size_penalty = min(area_ratio / 0.05, 1.0)

        print( (
            0.4 * grid_score,
            0.4 * aspect_penalty,
            0.2 * size_penalty,
        ))
        
        total_conf = (
            0.4 * grid_score +
            0.4 * aspect_penalty +
            0.2 * size_penalty
        )
        return total_conf


    def find_squares(self, contours):
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.settings["resolution_area"] * 0.01:
                continue
    
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w, h), angle = rect
            
            if abs(angle % 180) != 0.0 and abs(angle != 90.0):
                continue

            if w == 0 or h == 0:
                continue
    
            # Test square aspect ratio with 10% error
            aspect_ratio = max(w, h) / min(w, h)
            if 0.9 < aspect_ratio < 1.1:
                box = cv2.boxPoints(rect)
                box = box.astype(np.intp)
                candidates.append((box, area, angle))
    
        return candidates

    def return_best_candidate(self, image, candidates):
        H, W = image.shape[:2]
        best_conf = 0
        best_box = None
    
        for box, area, angle in candidates:
            x, y, w, h = cv2.boundingRect(box)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            board_crop = image[y1:y2, x1:x2]
            if board_crop.size == 0:
                continue
    
            edges = self.apply_edge_detection(board_crop)
            vertical, horizontal = self.count_grid_lines(edges)
    
            conf = self.compute_grid_confidence(board_crop, (H, W), vertical, horizontal, angle)
            print(f"Candidate confidence: {conf:.2f} (v={vertical}, h={horizontal})")
    
            if conf > best_conf:
                best_conf = conf
                best_box = box
    
        if best_conf < 0.5:  # adjustable threshold
            print("No strong candidate found.")
            return None
    
        print(f"Best candidate confidence: {best_conf:.2f}")
        return best_box

    def detect(self, image):
        edges = self.apply_edge_detection(image)

        vertical, horizontal = self.count_grid_lines(edges)
        if vertical < 6 or horizontal < 6:
            self.plotter.show_image(image, "No grid")
            print("No grid pattern found.")
            return None
            
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.plotter.show_image(image, "No Contours")
            print("No contours found.")
            return None

        candidates = self.find_squares(contours)
        if not candidates:
            self.plotter.show_image(image, "No Candidates")
            print("No candidates found.")
            return None
            
        best_candidate = self.return_best_candidate(image, candidates)
        self.plotter.draw_contours(image, [best_candidate])
    
        return best_candidate