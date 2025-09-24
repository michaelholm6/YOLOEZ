# Import your existing steps
from DefectDetect.labelling_workflow.image_preprocessing import *
from DefectDetect.labelling_workflow.area_of_interest_marking import *
from DefectDetect.labelling_workflow.detect_cracks import *
from DefectDetect.labelling_workflow.clip_cracks_to_area_of_interest import *
from DefectDetect.labelling_workflow.save_and_display_results import *
from DefectDetect.labelling_workflow.edit_contour_points import *
from DefectDetect.labelling_workflow.input_dialogue import *
from utils import show_instructions

def run_labeling_workflow(suppress_instructions=False):
    args = get_user_inputs()
    
    if not suppress_instructions:
        instructions = (
            "Instructions for using the Image Preprocessing GUI:\n\n"
            "- Use the sliders and spin boxes to adjust blur kernel size, CLAHE clip limit, and tile grid size.\n"
            "- The original and post-processed images are shown side by side.\n"
            "- Press ESC to quit the application.\n"
            "- Adjust parameters to achieve the desired image enhancement.\n"
            "- When finished, close the window.\n"
        )

        show_instructions(message=instructions)
        
    image, blurred = run_preprocess_gui(args["image_path"])
    
    if not suppress_instructions:
        # Your instructions text
        instructions = (
        "Instructions for Marking the Area of Interest:\n\n"
        "- You will now select your area of interest by drawing a polygon on the image.\n"
        "- The polygon should outline the area where you want to detect defects.\n"
        "- Left click to add points and outline your polygon.\n"
        "- Press 'C' to close the polygon and exit the window.\n"
        "- Press 'R' to reset points.\n"
        "- Press 'Esc' to cancel and close the application.\n"
        "- When finished, close the window.\n"
        "- To mark the entire image as the area of interest, just close the window without adding any points.\n"
    )
        show_instructions(message=instructions)
        
    area_of_interest = get_polygon_from_user(image)
    
    if not suppress_instructions:
        instructions = (
            "Use the sliders or spin boxes to adjust:\n"
            "- Confidence threshold: higher filters out weak edges.\n"
            "- Line thickness: changes contour line width.\n"
            "The image updates in real-time.\n"
            "Press ESC to exit and close the application.\n"
            "Close the window when done.\n"
        )
        show_instructions(message=instructions)
    
    contours, line_thickness = detect_cracks(image, blurred, area_of_interest)
    
    if not suppress_instructions:
        instructions = (
            "Instructions for Editing Contours:\n\n"
            "- You will now manually edit or create contours on the image.\n"
            "- Left click to select points by circling them with a lasso.\n"
            "- Right click and drag to pan the image.\n"
            "- Press 'C' to create a new contour by clicking to add points. Close the contour by pressing 'C' again.\n"
            "- Press 'D' to delete selected points.\n"
            "- Press 'U' to undo the last action.\n"
            "- Press 'S' to scale selected points (drag mouse relative to center).\n"
            "- Press 'M' to move selected points (drag mouse).\n"
            "- Press 'R' to rotate selected points (drag mouse relative to center).\n"
            "- Use the mouse wheel to zoom in and out.\n"
            "- Press 'Esc' to cancel and close the application.\n"
            "- Press the same key again or left click to exit a mode.\n\n"
            "When you're done, simply close the window to continue."
        )
        show_instructions(message=instructions)
    
    contours = run_contour_editor(image, contours, line_thickness)
    
    print('yay')
    save_and_display_results(image, contours, line_thickness, args["output_path"], area_of_interest)
    print(f"Results saved to {args['output_path']}")