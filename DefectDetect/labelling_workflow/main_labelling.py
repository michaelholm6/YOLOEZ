# Import your existing steps
from DefectDetect.labelling_workflow.image_gatherer import *
from DefectDetect.labelling_workflow.area_of_interest_marking import *
from DefectDetect.labelling_workflow.save_and_display_results import *
from DefectDetect.labelling_workflow.edit_contour_points import *
from DefectDetect.labelling_workflow.input_dialogue import *
from DefectDetect.labelling_workflow.edit_box_points import *
from utils import show_instructions

def run_labeling_workflow(suppress_instructions=False):
    args = get_user_inputs()
        
    images = args["image_paths"]
    
    if not suppress_instructions:
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
        
    areas_of_interest, line_thickness = annotate_images(args["image_paths"])
    
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
    
    if args['annotation_mode'] == 'segmentation':
    
        contours = run_contour_editor(images, line_thickness)
        
    else:
        contours = run_box_editor(images, line_thickness)
    
    save_and_display_results(images, contours, line_thickness, args["output_folder"], areas_of_interest)
    
    print(f"Results saved to {args['output_folder']}")