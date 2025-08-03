import cv2
import numpy as np

from PlotDrawer import PlotDrawer

class TemplateBoardDetector():
    def __init__(self, settings, template_image):
        self.settings = settings
        self.template_image = template_image
        self.SCALE_RANGE = np.linspace(0.05, 0.7, 75)
        self.THRESHOLD = 0.5  # Match quality threshold
        self.plotter = PlotDrawer()

    def match_template(self, test_gray, template_gray):
        template_h, template_w = template_gray.shape[:2]
        found = None
    
        for scale in self.SCALE_RANGE:
            resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
            if resized_template.shape[0] > test_gray.shape[0] or resized_template.shape[1] > test_gray.shape[1]:
                # Return if linear?
                continue
    
            result = cv2.matchTemplate(test_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
            if max_val > self.THRESHOLD:
                if found is None or max_val > found[0]:
                    found = (max_val, max_loc, scale)
    
        return found

    def detect(self, image):
        test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        
        found = self.match_template(test_gray, template_gray)
        
        if found:
            max_val, max_loc, best_scale = found
            best_template = cv2.resize(template_gray, None, fx=best_scale, fy=best_scale, interpolation=cv2.INTER_AREA)
            
            top_left = max_loc
            bottom_right = (top_left[0] + best_template.shape[1], top_left[1] + best_template.shape[0])
            # self.plotter.draw_box(image, top_left, bottom_right)
            return top_left, bottom_right
        # else:
        #     self.plotter.show_image(image, "No template match")

class OptimizedTemplateDetector():
    def __init__(self, settings, template_image):
        self.settings = settings
        self.template_image = template_image
        self.COARSE_SCALE_RANGE = np.linspace(0.05, 0.7, 50)
        self.THRESHOLD = 0.4  # Match quality threshold
        self.plotter = PlotDrawer()

    def match_template(self, test_gray, template_gray, scale_range):
        template_h, template_w = template_gray.shape[:2]
        found = None
    
        for scale in scale_range:
            resized_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
            if resized_template.shape[0] > test_gray.shape[0] or resized_template.shape[1] > test_gray.shape[1]:
                continue
    
            result = cv2.matchTemplate(test_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
            if max_val > self.THRESHOLD:
                if found is None or max_val > found[0]:
                    found = (max_val, max_loc, scale)
    
        return found

    def detect(self, image):
        ds_factor = 0.5
        frame_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray_full = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)

        # Improve speed by downscaling frames
        frame_small = cv2.resize(frame_gray_full, None,
                                    fx=ds_factor, fy=ds_factor,
                                    interpolation=cv2.INTER_AREA)

        template_gray_small = cv2.resize(template_gray_full, None,
                            fx=ds_factor, fy=ds_factor,
                            interpolation=cv2.INTER_AREA)
        
        found = self.match_template(frame_small, template_gray_small, self.COARSE_SCALE_RANGE)
        if not found:
            # self.plotter.show_image(image, "No template match")
            return None

        coarse_max_val, coarse_max_loc, coarse_best_scale = found

        # Improve accuracy by performing fine scaling range
        FINE_SCALE_RANGE = np.arange(coarse_best_scale - 0.01, coarse_best_scale + 0.01, 0.002)
        fine_found = self.match_template(frame_gray_full, template_gray_full, FINE_SCALE_RANGE)
        
        max_val, max_loc, best_scale = fine_found if fine_found else found
        best_template_length = int(template_gray_full.shape[0] * best_scale)
        best_template_width = int(template_gray_full.shape[1] * best_scale)
        
        top_left = max_loc
        bottom_right = (top_left[0] + best_template_length, top_left[1] + best_template_width)
        
        # self.plotter.draw_box(image, top_left, bottom_right, title=f"Match {max_val:.2f}, Course Scale: {coarse_best_scale}, Best Scale: {best_scale}")
        return top_left, bottom_right
            