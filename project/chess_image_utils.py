def crop_chessboard(image, match_dimensions):
    return image[match_dimensions[0]:match_dimensions[1], match_dimensions[2]:match_dimensions[3]]

def divide_chessboard(image):
    images = []
    height, width, _ = image.shape

    square_height = height // 8
    square_width = width // 8

    for row in range(8):
        for col in range(8):
            x1 = col * square_width
            x2 = (col + 1) * square_width
            y1 = row * square_height
            y2 = (row + 1) * square_height

            square = image[y1:y2, x1:x2]
            images.append((row, col, square))

    return images

def resize_image(image, target_size=(64, 64)):
    original_height, original_width, _ = image.shape
    target_height, target_width = target_size

    if target_height > original_height or target_width > original_width:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)