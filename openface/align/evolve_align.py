from PIL import Image
from detector import detect_faces
from visualization_utils import show_results

img = Image.open(
    "/home/chandra/tejasvi/facenet_arch/nested/10/0.jpg"
)  # modify the image path to yours
bounding_boxes, landmarks = detect_faces(
    img
)  # detect bboxes and landmarks for all faces in the image
img_aligned = show_results(img, bounding_boxes, landmarks)  # visualize the results
img_aligned
