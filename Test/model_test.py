import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient

# Class names for reporting
CLASS_NAMES = {
    0: 'Bilirubin', 1: 'Blood', 2: 'Glucose', 3: 'Ketone',
    4: 'Leukocytes', 5: 'Nitrite', 6: 'Protein', 7: 'SpGravity',
    8: 'Urobilinogen', 9: 'background', 10: 'pH', 11: 'strip'
}

# Define color scheme for visualization
CLASS_COLORS = [
    (255, 0, 0),    # 0: Bilirubin (Red)
    (0, 0, 255),    # 1: Blood (Blue)
    (0, 255, 0),    # 2: Glucose (Green)
    (255, 255, 0),  # 3: Ketone (Yellow)
    (255, 0, 255),  # 4: Leukocytes (Magenta)
    (0, 255, 255),  # 5: Nitrite (Cyan)
    (128, 0, 0),    # 6: Protein (Maroon)
    (0, 128, 0),    # 7: SpGravity (Dark Green)
    (0, 0, 128),    # 8: Urobilinogen (Navy)
    (128, 128, 128),# 9: Background (Gray)
    (255, 165, 0),  # 10: pH (Orange)
    (75, 0, 130)    # 11: Strip (Indigo)
]

class RoboflowUrineAnalyzer:
    def __init__(self, api_key, workspace_name, workflow_id):
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=api_key
        )
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        
    def analyze_image(self, image_path, confidence_threshold=0.5):
        """Process an image using Roboflow API"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None
            
        print(f"Analyzing image: {image_path}")
        
        # Run inference through Roboflow
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={"image": image_path},
                use_cache=True
            )
            
            # Debug: Print the raw result structure
            print("\nRoboflow API Response Structure:")
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Keys in result: {list(result.keys())}")
            elif isinstance(result, list):
                print(f"List length: {len(result)}")
                if len(result) > 0:
                    if isinstance(result[0], dict):
                        print(f"Keys in first element: {list(result[0].keys())}")
                        
                        # If we have base64 image output, decode and use it
                        if 'output' in result[0] and isinstance(result[0]['output'], str):
                            try:
                                # Check if it looks like base64
                                if result[0]['output'].startswith('/9j/') or result[0]['output'].startswith('iVBOR'):
                                    print("Found base64 image output - decoding")
                                    import base64
                                    import io
                                    
                                    # Remove header if present (data:image/jpeg;base64,)
                                    img_data = result[0]['output']
                                    if ',' in img_data:
                                        img_data = img_data.split(',', 1)[1]
                                    
                                    # Decode the base64 image
                                    img_bytes = base64.b64decode(img_data)
                                    image_np = np.array(Image.open(io.BytesIO(img_bytes)))
                                    
                                    # Return both the original and the decoded result
                                    original_image = np.array(Image.open(image_path))
                                    return {
                                        'original_image': original_image,
                                        'result_image': image_np,
                                        'predictions': result,
                                    }
                            except Exception as e:
                                print(f"Error decoding base64 image: {e}")
            
            # Load original image for visualization
            image = Image.open(image_path)
            image_np = np.array(image)
            
            # Process and visualize results
            visualization = self.visualize_results(image_np, result, confidence_threshold)
            
            return {
                'original_image': image_np,
                'result_image': visualization,
                'predictions': result,
            }
        except Exception as e:
            print(f"Error calling Roboflow API: {e}")
            # Return a minimal result with just the original image
            image = Image.open(image_path)
            image_np = np.array(image)
            return {
                'original_image': image_np,
                'result_image': image_np.copy(),  # No modifications
                'predictions': None,
                'error': str(e)
            }
    
    def visualize_results(self, image_np, result, confidence_threshold=0.5):
        """Draw bounding boxes and labels on the image"""
        result_image = image_np.copy()
        
        # Check if we received a modified image output directly
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'output' in result[0]:
                # For base64 encoded images, we already handled this in analyze_image
                # and the processed image is already in result_image
                print("Using pre-processed image output from API response")
                return result_image
        
        # Check for empty result
        if result is None:
            print("Result is None")
            return result_image
            
        # Try different formats that Roboflow API might return
        try:
            # Format 1: List of tasks with individual tasks results
            if isinstance(result, list):
                for task_result in result:
                    if isinstance(task_result, dict):
                        # Debug print for each task
                        print(f"Task keys: {list(task_result.keys())}")
                        
                        # Try to find predictions in any of these common structures
                        if 'predictions' in task_result:
                            self._draw_predictions(result_image, task_result['predictions'], confidence_threshold)
                        elif 'Bounding Box' in task_result and 'predictions' in task_result['Bounding Box']:
                            self._draw_predictions(result_image, task_result['Bounding Box']['predictions'], confidence_threshold)
                        elif 'tasks' in task_result:
                            for subtask in task_result['tasks']:
                                if 'predictions' in subtask:
                                    self._draw_predictions(result_image, subtask['predictions'], confidence_threshold)
                                    
            # Format 2: Dictionary with 'tasks' or direct 'predictions'
            elif isinstance(result, dict):
                # Debug print for dictionary result
                print(f"Result dict keys: {list(result.keys())}")
                
                if 'predictions' in result:
                    self._draw_predictions(result_image, result['predictions'], confidence_threshold)
                elif 'tasks' in result:
                    for task in result['tasks']:
                        if 'predictions' in task:
                            self._draw_predictions(result_image, task['predictions'], confidence_threshold)
                
            else:
                print(f"Unexpected result type: {type(result)}")
        
        except Exception as e:
            print(f"Error during visualization: {e}")
            
        return result_image
        
    def _draw_predictions(self, image, predictions, confidence_threshold):
        """Helper method to draw predictions on image"""
        if not isinstance(predictions, list):
            print(f"Predictions is not a list: {type(predictions)}")
            return
            
        print(f"Processing {len(predictions)} predictions")
        
        for pred in predictions:
            if not isinstance(pred, dict):
                print(f"Prediction is not a dictionary: {type(pred)}")
                continue
                
            # Debug print first prediction to see its structure
            if predictions.index(pred) == 0:
                print(f"Example prediction: {pred}")
                
            # Check confidence threshold
            confidence = pred.get('confidence', 0)
            if confidence < confidence_threshold:
                continue
            
            # Extract coordinates - handle different possible formats
            if 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                # YOLO-style centerpoint format
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('width', 0)
                h = pred.get('height', 0)
                
                # Calculate absolute coordinates (top-left and bottom-right)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
            elif 'x' in pred and 'y' in pred and 'w' in pred and 'h' in pred:
                # Alternative naming
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                w = pred.get('w', 0)
                h = pred.get('h', 0)
                
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
            elif 'bbox' in pred:
                # Bbox array format [x1, y1, width, height] or [x1, y1, x2, y2]
                bbox = pred['bbox']
                if len(bbox) == 4:
                    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # Likely [x1,y1,x2,y2] format
                        x1, y1, x2, y2 = bbox
                    else:  # Likely [x1,y1,w,h] format
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h
            elif 'box' in pred:
                # Another common format
                box = pred['box']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            else:
                # Skip if no valid coordinates
                print(f"Skipping prediction - no valid coordinates: {pred}")
                continue
                
            # Get class information
            class_name = pred.get('class', pred.get('class_name', 'Unknown'))
            class_id = pred.get('class_id', 9)  # Default to background if not found
            
            # Get color based on class_id
            color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
            
            # Make sure coordinates are valid
            x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
            if x1 >= x2 or y1 >= y2 or x2 >= image.shape[1] or y2 >= image.shape[0]:
                print(f"Invalid coordinates: [{x1}, {y1}, {x2}, {y2}] for image shape {image.shape}")
                continue
            
            # Draw bounding box and label
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{class_name}: {confidence:.2f}", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except Exception as e:
                print(f"Error drawing on image: {e}")
    
    def save_results(self, results, save_path):
        """Save results to a file"""
        if results and 'result_image' in results:
            cv2.imwrite(save_path, cv2.cvtColor(results['result_image'], cv2.COLOR_RGB2BGR))
            print(f"Results saved to {save_path}")
    
    def extract_strip(self, image_np, predictions, confidence_threshold=0.5):
        """Extract the strip region if detected"""
        strip_image = None
        strip_bbox = None
        
        # Try to find strip in predictions
        try:
            if isinstance(predictions, list):
                for pred_list in predictions:
                    if isinstance(pred_list, dict) and 'predictions' in pred_list:
                        for pred in pred_list['predictions']:
                            if pred.get('class', '').lower() == 'strip' and pred.get('confidence', 0) >= confidence_threshold:
                                # Extract bbox coordinates
                                x = pred.get('x', 0)
                                y = pred.get('y', 0)
                                w = pred.get('width', 0)
                                h = pred.get('height', 0)
                                
                                # Calculate absolute coordinates
                                x1 = max(0, int(x - w/2))
                                y1 = max(0, int(y - h/2))
                                x2 = min(image_np.shape[1], int(x + w/2))
                                y2 = min(image_np.shape[0], int(y + h/2))
                                
                                # Extract the region
                                if x1 < x2 and y1 < y2:
                                    strip_image = image_np[y1:y2, x1:x2].copy()
                                    strip_bbox = (x1, y1, x2, y2)
                                    return strip_image, strip_bbox
            
            # Fallback: Try basic image processing to find the strip
            if strip_image is None:
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Find the largest contour (likely to be the strip)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest)
                    if w > 0 and h > 0:
                        strip_image = image_np[y:y+h, x:x+w].copy()
                        strip_bbox = (x, y, x+w, y+h)
        except Exception as e:
            print(f"Error extracting strip: {e}")
        
        return strip_image, strip_bbox


def update_image_on_canvas(image_np):
    """Update the image shown on the canvas"""
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_tk = ImageTk.PhotoImage(image_pil)
    canvas.itemconfig(image_on_canvas, image=image_tk)
    canvas.image = image_tk

def display_strip_in_plot(strip_image):
    """Display the cropped strip in a matplotlib plot"""
    if strip_image is None:
        print("No strip image to display")
        return
    
    # Clear previous plot
    strip_plot_ax.clear()
    
    # Display the image
    strip_plot_ax.imshow(strip_image)
    strip_plot_ax.set_title("Cropped Strip")
    strip_plot_ax.axis('on')  # Turn on axis
    
    # Add a grid for better visualization of the pads
    strip_plot_ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add color intensity profile
    if strip_image.shape[0] > 20:  # Only if the strip has reasonable height
        # Calculate average color intensity across the strip height
        intensity = np.mean(strip_image, axis=0)  # Average across height
        
        # Create a new axis
        color_profile_ax.clear()
        
        # Plot RGB channels
        x = np.arange(intensity.shape[0])
        color_profile_ax.plot(x, intensity[:, 0], 'r-', alpha=0.7, label='Red')
        color_profile_ax.plot(x, intensity[:, 1], 'g-', alpha=0.7, label='Green')
        color_profile_ax.plot(x, intensity[:, 2], 'b-', alpha=0.7, label='Blue')
        
        color_profile_ax.set_title("Color Intensity Profile")
        color_profile_ax.set_xlabel("Position along strip (pixels)")
        color_profile_ax.set_ylabel("Intensity")
        color_profile_ax.legend(loc='upper right')
        color_profile_ax.grid(True)
    
    # Update the canvas
    strip_plot_canvas.draw()

def crop_strip_button_click():
    """Handle crop button click"""
    global current_image_path, last_results
    
    if current_image_path and last_results:
        print("Cropping strip from image...")
        strip_image, strip_bbox = analyzer.extract_strip(
            last_results['original_image'], 
            last_results.get('predictions', []), 
            float(scale.get())
        )
        
        if strip_image is not None:
            # Display the cropped strip
            display_strip_in_plot(strip_image)
            strip_label_var.set(f"Strip: {strip_image.shape[1]}x{strip_image.shape[0]} pixels")
        else:
            strip_label_var.set("No strip detected")
    else:
        strip_label_var.set("No image loaded")

def process_and_display(image_path, confidence_threshold, save_results=False, save_path=None):
    """Process an image and display the results"""
    global last_results
    
    # Process the image
    results = analyzer.analyze_image(image_path, confidence_threshold)
    last_results = results  # Store results for later use
    
    if not results:
        status_var.set("Error processing image")
        return None
    
    # Update the display
    update_image_on_canvas(results['result_image'])
    
    # Update results text
    result_text = "Analysis Results:\n"
    
    if 'error' in results:
        result_text += f"\nError: {results['error']}\n"
    
    # Handle the specific response format from Roboflow
    if 'predictions' in results:
        predictions = results['predictions']
        
        # Special case for direct image output with classification data
        if isinstance(predictions, list) and len(predictions) > 0:
            if 'classification_label_visualization' in predictions[0]:
                result_text += "\nClassification Results:\n"
                label_vis = predictions[0].get('classification_label_visualization', {})
                
                if isinstance(label_vis, dict):
                    # Try to extract class labels and confidences
                    for class_name, data in label_vis.items():
                        if isinstance(data, dict) and 'confidence' in data:
                            confidence = data['confidence']
                            if confidence >= confidence_threshold:
                                result_text += f"{class_name}: {confidence:.2f}\n"
                        elif isinstance(data, float):
                            # Sometimes it's just a direct confidence value
                            if data >= confidence_threshold:
                                result_text += f"{class_name}: {data:.2f}\n"
        
        # Attempt to extract predictions from different formats
        if isinstance(predictions, dict):
            if 'tasks' in predictions:
                for task in predictions['tasks']:
                    if 'predictions' in task:
                        result_text += f"\nTask: {task.get('type', 'Unknown')}\n"
                        for pred in task['predictions']:
                            class_name = pred.get('class', 'Unknown')
                            confidence = pred.get('confidence', 0)
                            if confidence >= confidence_threshold:
                                result_text += f"{class_name}: {confidence:.2f}\n"
            elif 'predictions' in predictions:
                for pred in predictions['predictions']:
                    class_name = pred.get('class', 'Unknown')
                    confidence = pred.get('confidence', 0)
                    if confidence >= confidence_threshold:
                        result_text += f"{class_name}: {confidence:.2f}\n"
        
        # List format
        elif isinstance(predictions, list):
            for item in predictions:
                if isinstance(item, dict):
                    # Format can vary; try different common structures
                    if 'tasks' in item:
                        for task in item['tasks']:
                            if isinstance(task, dict) and 'predictions' in task:
                                result_text += f"\nTask: {task.get('type', 'Unknown')}\n"
                                for pred in task['predictions']:
                                    if isinstance(pred, dict):
                                        class_name = pred.get('class', 'Unknown')
                                        confidence = pred.get('confidence', 0)
                                        if confidence >= confidence_threshold:
                                            result_text += f"{class_name}: {confidence:.2f}\n"
    
    # Display raw JSON for debugging if no structured info was found
    if result_text == "Analysis Results:\n" and 'predictions' in results:
        result_text += "\nRaw Results (Debug):\n"
        result_text += str(results['predictions'])[:500] + '...'  # Truncate if too long
    
    results_display.config(state=tk.NORMAL)
    results_display.delete(1.0, tk.END)
    results_display.insert(tk.END, result_text)
    results_display.config(state=tk.DISABLED)
    
    # Save results if requested
    if save_results and save_path:
        analyzer.save_results(results, save_path)
        status_var.set(f"Results saved to {save_path}")
    else:
        status_var.set("Ready")
    
    return results

def update_confidence_threshold(val):
    """Update the confidence threshold and re-process the current image"""
    global current_image_path
    confidence_threshold = float(val)
    if current_image_path:
        process_and_display(current_image_path, confidence_threshold, save_results_var.get(), save_path_var.get())

def select_image():
    """Open a file dialog to select an image"""
    global current_image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        current_image_path = image_path
        confidence_threshold = float(scale.get())
        process_and_display(image_path, confidence_threshold, save_results_var.get(), save_path_var.get())


# Main GUI setup
if __name__ == "__main__":
    # Initialize Roboflow client
    API_KEY = "rmrJF4g00wIf5b5xyYht"  # Your Roboflow API key
    WORKSPACE_NAME = "urine-test-strips-qq9jx"  # Your Roboflow workspace name
    WORKFLOW_ID = "detect-and-classify"  # Your Roboflow workflow ID
    
    analyzer = RoboflowUrineAnalyzer(API_KEY, WORKSPACE_NAME, WORKFLOW_ID)
    
    # Initialize global variables
    current_image_path = None
    last_results = None
    
    # Create the main window
    root = tk.Tk()
    root.title("Urine Strip Analyzer")
    root.geometry("1200x700")  # Make window taller for the plot
    
    # Create frames for layout
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X)
    
    # Add a button to select an image
    select_button = tk.Button(top_frame, text="Select Image", command=select_image)
    select_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add confidence threshold slider
    threshold_label = tk.Label(top_frame, text="Confidence Threshold:")
    threshold_label.pack(side=tk.LEFT, padx=10, pady=10)
    scale = tk.Scale(top_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                    resolution=0.01, command=update_confidence_threshold, length=200)
    scale.set(0.5)  # Default value
    scale.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add checkbox to save results
    save_results_var = tk.BooleanVar()
    save_results_checkbox = tk.Checkbutton(top_frame, text="Save Results", variable=save_results_var)
    save_results_checkbox.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Add entry for save path
    save_path_var = tk.StringVar()
    save_path_entry = tk.Entry(top_frame, textvariable=save_path_var, width=30)
    save_path_entry.pack(side=tk.LEFT, padx=10, pady=10)
    save_path_var.set(r'D:\Programming\urine_interpret\results.jpg')  # Default save path
    
    # Add crop button
    crop_button = tk.Button(top_frame, text="Crop Strip", command=crop_strip_button_click)
    crop_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    # Create a canvas to display the image
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    canvas = tk.Canvas(canvas_frame, width=256, height=256, bd=2, relief=tk.SUNKEN)
    canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Initialize the image on the canvas
    image_np = np.zeros((256, 256, 3), dtype=np.uint8)
    image_pil = Image.fromarray(image_np)
    image_tk = ImageTk.PhotoImage(image_pil)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    
    # Text area for displaying results
    results_display = tk.Text(canvas_frame, height=10, width=30, state=tk.DISABLED)
    results_display.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Bottom frame for strip plot
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill=tk.BOTH, expand=True)
    
    # Label for strip info
    strip_label_var = tk.StringVar()
    strip_label_var.set("No strip detected")
    strip_label = tk.Label(bottom_frame, textvariable=strip_label_var)
    strip_label.pack(pady=5)
    
    # Matplotlib figure for strip visualization
    strip_fig = plt.figure(figsize=(10, 4))
    strip_plot_ax = strip_fig.add_subplot(2, 1, 1)
    color_profile_ax = strip_fig.add_subplot(2, 1, 2)
    strip_fig.tight_layout(pad=3.0)
    
    # Embed the matplotlib figure in the tkinter window
    strip_plot_canvas = FigureCanvasTkAgg(strip_fig, master=bottom_frame)
    strip_plot_canvas.draw()
    strip_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Status bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Run the application
    root.mainloop()
