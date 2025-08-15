from crack_detection_algorithm_steps.image_preprocessing import *
from crack_detection_algorithm_steps.area_of_interest_marking import *
from crack_detection_algorithm_steps.detect_cracks import *
from crack_detection_algorithm_steps.clip_cracks_to_area_of_interest import *
from crack_detection_algorithm_steps.measure_contour_circularity import *
from crack_detection_algorithm_steps.get_phyiscal_scale import *
from crack_detection_algorithm_steps.compute_crack_skeleton import *
from crack_detection_algorithm_steps.display_statistics import *
from crack_detection_algorithm_steps.save_and_display_results import *
from crack_detection_algorithm_steps.edit_contour_points import *
from crack_detection_algorithm_steps.input_dialogue import *


    

def main():
    args = get_user_inputs()
    image, gray, blurred = run_preprocess_gui(args["image_path"], args["suppress_instructions"])
    #cv2.imshow('Blurred Image', blurred)
    area_of_interest = get_polygon_from_user(image, args["suppress_instructions"])
    cracks, line_thickness = detect_cracks(image, blurred, area_of_interest, args['suppress_instructions'])
    cracks = clip_contours_with_polygon(gray.shape, cracks, area_of_interest)
    cracks, circularity = run_contour_editor(image, cracks, line_thickness, args["suppress_instructions"])
    crack_circularity = measure_contour_circularity(cracks)
    crack_and_crack_circularity = [(cnt, circ) for cnt, circ in zip(cracks, crack_circularity)]
    um_per_pixel = get_scale_from_user(image, line_thickness, args["suppress_instructions"])
    skeleton = compute_skeleton(crack_and_crack_circularity, gray.shape, circularity)
    display_statistics(crack_and_crack_circularity, um_per_pixel, skeleton, area_of_interest, circularity)
    save_and_display_results(image, crack_and_crack_circularity, skeleton, line_thickness, args["output_path"])
    print(f"Results saved to {args["output_path"]}")