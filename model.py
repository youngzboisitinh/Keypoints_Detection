import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

class AdvancedPoseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Estimation")
        self.root.geometry("1200x800")
        
        self.model = None
        self.current_image = None
        self.processed_image = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model loading frame
        model_frame = ttk.LabelFrame(main_frame, text="Model Setup", padding="5")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Model Path:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="yolov8n-pose.pt")
        ttk.Entry(model_frame, textvariable=self.model_path_var, width=40).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=(5, 0))
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=(5, 0))
        
        self.model_status = ttk.Label(model_frame, text="Model not loaded", foreground="red")
        self.model_status.grid(row=1, column=0, columnspan=4, sticky=tk.W)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(control_frame, text="Select Image", command=self.select_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Process Image", command=self.process_image).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(control_frame, text="Save Result", command=self.save_result).grid(row=0, column=2, padx=(0, 5))
        
        # Confidence threshold
        ttk.Label(control_frame, text="Confidence:").grid(row=0, column=3, padx=(10, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL, length=100)
        confidence_scale.grid(row=0, column=4, padx=(0, 5))
        self.confidence_label = ttk.Label(control_frame, text="0.5")
        self.confidence_label.grid(row=0, column=5)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Image display frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Original image
        self.original_frame = ttk.LabelFrame(image_frame, text="Original Image", padding="5")
        self.original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.original_label = ttk.Label(self.original_frame, text="No image selected")
        self.original_label.pack(expand=True)
        
        # Processed image
        self.processed_frame = ttk.LabelFrame(image_frame, text="Processed Image", padding="5")
        self.processed_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.processed_label = ttk.Label(self.processed_frame, text="No processed image")
        self.processed_label.pack(expand=True)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.results_text = tk.Text(results_frame, height=6, width=80)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value):.1f}")
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def load_model(self):
        try:
            model_path = self.model_path_var.get()
            self.model = YOLO(model_path)
            
            # In debug để kiểm tra head
            print("===== KIỂM TRA MÔ HÌNH =====")
            print(self.model)
            try:
                head = self.model.model.model[-1]
                print("Head module:", head)
            except:
                pass
            print("============================")
            
            self.model_status.config(text="Model loaded successfully", foreground="green")
            self.log_result("Model loaded successfully")
        except Exception as e:
            self.model_status.config(text=f"Failed to load model: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                self.display_original_image()
                self.log_result(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_original_image(self):
        if self.current_image is not None:
            display_image = self.resize_for_display(self.current_image)
            image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.original_label.config(image=photo, text="")
            self.original_label.image = photo
    
    def display_processed_image(self, processed_img):
        display_image = self.resize_for_display(processed_img)
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        self.processed_label.config(image=photo, text="")
        self.processed_label.image = photo
    
    def resize_for_display(self, image, max_size=400):
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            if h > w:
                new_h = max_size
                new_w = int(w * max_size / h)
            else:
                new_w = max_size
                new_h = int(h * max_size / w)
            return cv2.resize(image, (new_w, new_h))
        return image
    
    def process_image(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
        
        # Process in separate thread để GUI không bị treo
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        try:
            # 1) Chạy inference
            results = self.model(self.current_image, conf=0.25, verbose=False)

            # ======== Bắt đầu chèn debug ========
            for idx, result in enumerate(results):
                print(f"--- DEBUG: Result {idx} ---")
                print("Has attribute 'keypoints'? ", hasattr(result, 'keypoints'))
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kp_data = result.keypoints.data
                    print("  Type of kp_data:", type(kp_data))
                    print("  kp_data device:", kp_data.device)
                    print("  kp_data.shape:", kp_data.shape)   # Kỳ vọng: (n_person, 17, 3)
                    try:
                        print("  Sample keypoints[0]:", kp_data[0])  
                    except:
                        pass
                else:
                    print("  → No keypoints (result.keypoints is None hoặc attribute không tồn tại)")
            print("=====================================")
            # ======== Kết thúc debug ========

            # 2) Xử lý (vẽ, classify, v.v.)
            processed_img = self.current_image.copy()
            pose_results = []
            detection_info = []

            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    detection_info.append(f"Detected {len(result.boxes)} person(s)")

                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    detection_info.append(f"Found {len(keypoints_data)} keypoint set(s)")

                    for i, person_keypoints in enumerate(keypoints_data):
                        kpts = person_keypoints.cpu().numpy()
                        visible_kpts = sum(1 for kp in kpts if kp[2] > 0.3)
                        detection_info.append(f"Person {i+1}: {visible_kpts}/17 keypoints visible")

                        pose_label, confidence_msg = self.classify_pose_flexible(kpts)
                        pose_results.append(f"Person {i+1}: {pose_label} {confidence_msg}")

                        processed_img = self.draw_keypoints(processed_img, kpts, pose_label, i)
                else:
                    detection_info.append("No keypoints detected")

            if not pose_results:
                pose_results.append("No pose detected - try lowering confidence threshold")

            # 3) Cập nhật UI
            self.root.after(0, lambda: self._update_results(processed_img, pose_results, detection_info))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))

    def _update_results(self, processed_img, pose_results, detection_info):
        self.processed_image = processed_img
        self.display_processed_image(processed_img)
        
        # Update results text với debug info
        result_text = "Pose Detection Results:\n"
        result_text += "=" * 30 + "\n"
        
        # Add detection debug info
        result_text += "Detection Info:\n"
        for info in detection_info:
            result_text += f"• {info}\n"
        result_text += "\n"
        
        # Add pose results
        result_text += "Pose Classification:\n"
        for r in pose_results:
            result_text += f"• {r}\n"
        
        self.log_result(result_text)
    
    def classify_pose_flexible(self, keypoints):
        """Improved flexible pose classification that works with partial keypoints"""
        if len(keypoints) != 17:
            return "Unknown", "(invalid keypoint format)"
        
        visibility_threshold = 0.25
        visible_kpts = [i for i, kp in enumerate(keypoints) if kp[2] > visibility_threshold]
        visible_count = len(visible_kpts)
        
        if visible_count < 3:
            return "Insufficient data", f"(only {visible_count} keypoints)"
        
        visible_points = [(keypoints[i][0], keypoints[i][1]) for i in visible_kpts]
        all_x = [p[0] for p in visible_points]
        all_y = [p[1] for p in visible_points]
        
        bbox_width = max(all_x) - min(all_x)
        bbox_height = max(all_y) - min(all_y)
        aspect_ratio = bbox_width / (bbox_height + 1e-8)
        
        classification_scores = {
            'lying': 0,
            'standing': 0,
            'sitting': 0,
            'kneeling': 0
        }
        confidence_details = []
        
        # RULE 1: Aspect Ratio
        if aspect_ratio > 1.4:
            classification_scores['lying'] += 3
            confidence_details.append(f"Wide AR:{aspect_ratio:.2f}")
        elif aspect_ratio > 1.1:
            classification_scores['lying'] += 1
            confidence_details.append(f"Med AR:{aspect_ratio:.2f}")
        elif aspect_ratio < 0.6:
            classification_scores['standing'] += 2
            confidence_details.append(f"Tall AR:{aspect_ratio:.2f}")
        elif aspect_ratio < 0.8:
            classification_scores['standing'] += 1
            classification_scores['sitting'] += 1
            confidence_details.append(f"Norm AR:{aspect_ratio:.2f}")
        
        # RULE 2: Body Orientation
        torso_points = self.get_torso_keypoints(keypoints, visibility_threshold)
        if torso_points:
            torso_angle = self.calculate_torso_angle(torso_points)
            if torso_angle is not None:
                if torso_angle > 60:
                    classification_scores['lying'] += 2
                    confidence_details.append(f"H-torso:{torso_angle:.0f}°")
                elif torso_angle < 20:
                    classification_scores['standing'] += 1
                    confidence_details.append(f"V-torso:{torso_angle:.0f}°")
                else:
                    classification_scores['sitting'] += 1
                    confidence_details.append(f"A-torso:{torso_angle:.0f}°")
        
        # RULE 3: Leg Analysis
        leg_analysis = self.analyze_legs(keypoints, visibility_threshold)
        for pose, score in leg_analysis['scores'].items():
            classification_scores[pose] += score
        if leg_analysis['details']:
            confidence_details.extend(leg_analysis['details'])
        
        # RULE 4: Vertical Position
        head_body_analysis = self.analyze_head_body_positions(keypoints, visibility_threshold)
        for pose, score in head_body_analysis['scores'].items():
            classification_scores[pose] += score
        if head_body_analysis['details']:
            confidence_details.extend(head_body_analysis['details'])
        
        # RULE 5: Few Keypoints
        if visible_count < 8:
            if aspect_ratio > 1.2:
                classification_scores['lying'] += 2
            elif aspect_ratio < 0.7:
                classification_scores['standing'] += 1
            confidence_details.append(f"Few-pts:{visible_count}")
        
        max_score = max(classification_scores.values())
        if max_score == 0:
            return "Unknown", f"(no clear indicators, {visible_count} pts)"
        
        best_pose = max(classification_scores.items(), key=lambda x: x[1])[0]
        confidence_level = "high" if max_score >= 3 else "medium" if max_score >= 2 else "low"
        detail_str = ", ".join(confidence_details) if confidence_details else "basic detection"
        
        return best_pose.capitalize(), f"({confidence_level}, {detail_str})"
    
    def get_torso_keypoints(self, keypoints, threshold):
        """Extract visible torso keypoints"""
        torso_indices = [5, 6, 11, 12]
        visible_torso = []
        for idx in torso_indices:
            if keypoints[idx][2] > threshold:
                visible_torso.append((keypoints[idx][0], keypoints[idx][1], idx))
        return visible_torso if len(visible_torso) >= 2 else None
    
    def calculate_torso_angle(self, torso_points):
        """Calculate torso orientation angle"""
        if len(torso_points) < 2:
            return None
        shoulders = [p for p in torso_points if p[2] in [5, 6]]
        hips = [p for p in torso_points if p[2] in [11, 12]]
        if not shoulders or not hips:
            return None
        
        shoulder_center = np.mean([[p[0], p[1]] for p in shoulders], axis=0)
        hip_center = np.mean([[p[0], p[1]] for p in hips], axis=0)
        torso_vector = hip_center - shoulder_center
        vertical_vector = np.array([0, 1])
        
        if np.linalg.norm(torso_vector) == 0:
            return None
        cos_angle = np.dot(torso_vector, vertical_vector) / np.linalg.norm(torso_vector)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        return angle
    
    def analyze_legs(self, keypoints, threshold):
        """Analyze leg positions for pose classification"""
        scores = {'lying': 0, 'standing': 0, 'sitting': 0, 'kneeling': 0}
        details = []
        
        left_leg = [(11, 13, 15), keypoints[11], keypoints[13], keypoints[15]]
        right_leg = [(12, 14, 16), keypoints[12], keypoints[14], keypoints[16]]
        leg_angles = []
        
        for leg_name, hip, knee, ankle in [("left", *left_leg[1:]), ("right", *right_leg[1:])]:
            if all(kp[2] > threshold for kp in [hip, knee, ankle]):
                angle = self.calculate_angle(hip[:2], knee[:2], ankle[:2])
                leg_angles.append(angle)
                
                if angle < 90:
                    scores['kneeling'] += 1
                    details.append(f"{leg_name}-bent:{angle:.0f}°")
                elif angle < 130:
                    scores['sitting'] += 1
                    details.append(f"{leg_name}-sit:{angle:.0f}°")
                else:
                    scores['standing'] += 1
                    details.append(f"{leg_name}-ext:{angle:.0f}°")
        
        if leg_angles:
            avg_angle = np.mean(leg_angles)
            if avg_angle < 100:
                scores['kneeling'] += 1
            elif avg_angle < 140:
                scores['sitting'] += 1
        
        return {'scores': scores, 'details': details}
    
    def analyze_head_body_positions(self, keypoints, threshold):
        """Analyze head and body vertical positions"""
        scores = {'lying': 0, 'standing': 0, 'sitting': 0, 'kneeling': 0}
        details = []
        
        head_y = keypoints[0][1] if keypoints[0][2] > threshold else None
        shoulders_y = []
        hips_y = []
        
        for idx in [5, 6]:
            if keypoints[idx][2] > threshold:
                shoulders_y.append(keypoints[idx][1])
        
        for idx in [11, 12]:
            if keypoints[idx][2] > threshold:
                hips_y.append(keypoints[idx][1])
        
        if head_y is not None and shoulders_y and hips_y:
            shoulder_avg = np.mean(shoulders_y)
            hip_avg = np.mean(hips_y)
            if abs(head_y - shoulder_avg) < abs(hip_avg - shoulder_avg) * 0.3:
                scores['lying'] += 1
                details.append("compact-vertical")
            elif head_y < shoulder_avg < hip_avg:
                scores['standing'] += 1
                details.append("stacked-vertical")
        
        return {'scores': scores, 'details': details}
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 180
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        except:
            return 180
    
    def draw_keypoints(self, img, keypoints, pose_label, person_id):
        """Draw keypoints and skeleton with better visibility"""
        img_copy = img.copy()
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[person_id % len(colors)]
        draw_threshold = 0.25
        
        # Draw skeleton
        for connection in skeleton:
            kpt_a, kpt_b = connection
            kpt_a_idx, kpt_b_idx = kpt_a - 1, kpt_b - 1
            if (kpt_a_idx < len(keypoints) and kpt_b_idx < len(keypoints) and
                keypoints[kpt_a_idx][2] > draw_threshold and 
                keypoints[kpt_b_idx][2] > draw_threshold):
                x_a, y_a = int(keypoints[kpt_a_idx][0]), int(keypoints[kpt_a_idx][1])
                x_b, y_b = int(keypoints[kpt_b_idx][0]), int(keypoints[kpt_b_idx][1])
                cv2.line(img_copy, (x_a, y_a), (x_b, y_b), color, 3)
        
        # Draw keypoint circles
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > draw_threshold:
                cv2.circle(img_copy, (int(x), int(y)), 6, color, -1)
                cv2.circle(img_copy, (int(x), int(y)), 6, (255, 255, 255), 2)
                if conf > 0.5:
                    cv2.putText(img_copy, f"{conf:.2f}", (int(x)+8, int(y)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw bounding box + aspect ratio for debug
        visible_points = [(x, y) for x, y, conf in keypoints if conf > draw_threshold]
        if len(visible_points) > 2:
            x_coords = [p[0] for p in visible_points]
            y_coords = [p[1] for p in visible_points]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            cv2.rectangle(img_copy, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 2)
            width = max_x - min_x
            height = max_y - min_y
            aspect_ratio = width / (height + 1e-8)
            debug_text = f"AR: {aspect_ratio:.2f}"
            cv2.putText(img_copy, debug_text, (int(min_x), int(min_y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw pose label
        if len(keypoints) > 0 and keypoints[0][2] > draw_threshold:
            text_pos = (int(keypoints[0][0]-50), int(keypoints[0][1] - 30))
        else:
            text_pos = (50, 50 + person_id * 30)
        
        text = f"Person {person_id+1}: {pose_label}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_copy, (text_pos[0]-5, text_pos[1]-text_height-5), 
                     (text_pos[0]+text_width+5, text_pos[1]+5), (255, 255, 255), -1)
        cv2.putText(img_copy, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img_copy
    
    def save_result(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(
                title="Save Processed Image",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                self.log_result(f"Image saved to: {file_path}")
                messagebox.showinfo("Success", "Image saved successfully!")
        else:
            messagebox.showwarning("Warning", "No processed image to save!")
    
    def log_result(self, message):
        """Add message to results text area"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPoseGUI(root)
    
    # Thêm menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # File menu
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Load Model", command=app.load_model)
    file_menu.add_command(label="Select Image", command=app.select_image)
    file_menu.add_command(label="Save Result", command=app.save_result)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    # Help menu
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="About", 
                         command=lambda: messagebox.showinfo("About", 
                                                            "YOLOv8 Pose Estimation\n"
                                                            "Phân loại tư thế: Đứng, Ngồi, Nằm, Quỳ\n"
                                                            "Sử dụng Ultralytics YOLOv8\n"
                                                            "Improved flexible keypoint detection"))
    
    root.mainloop()
