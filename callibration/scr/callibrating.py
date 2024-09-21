import cv2

# Load your image
image = cv2.imread('callibration/img/testing_img_1.jpg')

drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial coordinates

# Mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = image.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', image)
        calculate_conversion_factor(ix, iy, x, y)

# Create a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)

def calculate_conversion_factor(ix, iy, x, y):
    width_pixels = abs(x - ix)
    height_pixels = abs(y - iy)
    known_width_cm = 5.0  # Example known width
    known_height_cm = 5.0  # Example known height
    
    pixels_per_cm_x = width_pixels / known_width_cm
    pixels_per_cm_y = height_pixels / known_height_cm

    print(f'Pixels per cm (Width): {pixels_per_cm_x}')
    print(f'Pixels per cm (Height): {pixels_per_cm_y}')

while True:
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
