import cv2
import numpy as np
import os

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def create_kernels():
    open_red_kernel = np.ones((3, 4), np.uint8)
    close_kernel = np.ones((7, 7), np.uint8)
    blue_kernel = np.ones((11, 11), np.uint8)
    return open_red_kernel, close_kernel, blue_kernel

def create_masks(hsv_image):
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    lower_red1 = np.array([0, 50, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 30])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blue_mask, red_mask

def find_blue_contours(blue_mask, blue_kernel):
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, blue_kernel)
    cv2.imshow("Blue Mask", blue_mask)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return blue_contours

def find_red_contours(red_mask, open_red_kernel, close_kernel, image):
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow("Canny", edges)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    red_mask = cv2.bitwise_and(red_mask, edges)

    all_red_contours = []
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_red_contours.extend(red_contours)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, open_red_kernel)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_red_contours.extend(red_contours)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, close_kernel)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_red_contours.extend(red_contours)
    cv2.imshow("Red Mask", red_mask)

    return all_red_contours

def find_contours(image):
    hsv_image = convert_to_hsv(image)
    open_red_kernel, close_kernel, blue_kernel = create_kernels()
    blue_mask, red_mask = create_masks(hsv_image)
    blue_contours = find_blue_contours(blue_mask, blue_kernel)
    red_contours = find_red_contours(red_mask, open_red_kernel, close_kernel, image)
    return image, blue_contours, red_contours

def detect_traffic_signs(image, blue_contours, red_contours):
    for contour in blue_contours:
        area = cv2.contourArea(contour)
        if area > 100:
            epsilon = 0.025 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 4 and (h / 3 <= w <= 3 * h):
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 1)
                cv2.putText(image, "Indication", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 0, 0), 2)
            else:
                (x, y), _ = cv2.minEnclosingCircle(contour)
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                if circularity >= 0.6:
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
                    cv2.putText(image, "Obligation", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 0, 0), 2)

    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)

            if len(approx) == 3:
                pts = approx.reshape(3, 2)
                sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]
                if all(0.9 <= sides[i] / sides[(i + 1) % 3] <= 1.1 for i in range(3)):
                    cv2.drawContours(image, [approx], -1, (0, 255, 0), 1)
                    cv2.putText(image, "Danger", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                (x, y), _ = cv2.minEnclosingCircle(contour)
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2)
                if circularity >= 0.8:
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)
                    cv2.putText(image, "Prohibition", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return image

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No image found")
        return None

    image, blue_contours, red_contours = find_contours(image)
    final_image = detect_traffic_signs(image, blue_contours, red_contours)
    return final_image

if __name__ == "__main__":
    input_directory = "signals"
    for file in os.listdir(input_directory):
        image_path = os.path.join(input_directory, file)
        final_image = process_image(image_path)
        if final_image is not None:
            cv2.imshow("Detected signals", final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
