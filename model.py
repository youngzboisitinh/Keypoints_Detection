import cv2
import numpy as np
import math
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading

# =======================
# 1) Đặt hàm standalone classify_pose_flexible lên đây
# =======================
def classify_pose_flexible(keypoints):
    """
    Phân loại tư thế dựa trên keypoints người (x,y). Trả về 'standing', 'lying' hoặc 'unknown'.
    keypoints: array có kích thước (N, 3) theo chuẩn YOLOv8 (x, y, confidence).
    Chúng ta chỉ dùng x,y để tính toán.
    """
    # Lọc keypoints hợp lệ (loại bỏ giá trị None hoặc NaN)
    data = []
    for p in keypoints:
        if p is None:
            continue
        # p có dạng (x, y, conf)
        try:
            x, y, _ = p  # chỉ lấy x,y, bỏ conf
        except:
            continue
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            continue
        data.append((x, y))
    
    if len(data) < 2:
        # Quá ít điểm để xác định tư thế
        return "unknown"
    data = np.array(data, dtype=np.float32)
    
    # Tính khung giới hạn (bounding box) và aspect ratio
    xs = data[:, 0]
    ys = data[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min() + 1e-6  # tránh chia 0
    aspect_ratio = width / height
    
    # Tính phương sai theo các chiều X, Y
    var_x = np.var(xs)
    var_y = np.var(ys)
    var_ratio = var_x / (var_y + 1e-6)
    
    # PCA để tìm trục chính và góc với trục dọc
    cov = np.cov(data.T)
    if cov.shape == (2, 2):
        eig_vals, eig_vecs = np.linalg.eig(cov)
    else:
        eig_vals = np.array([var_x, var_y])
        eig_vecs = np.array([[1, 0], [0, 1]])
    idx = np.argmax(eig_vals)
    principal_vec = eig_vecs[:, idx]
    norm_vec = principal_vec / (np.linalg.norm(principal_vec) + 1e-6)
    angle_to_vertical = math.degrees(math.acos(abs(norm_vec[1])))
    
    # Tính góc thân (nếu có vai và hông)
    trunk_angle = None
    mp_indices = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_hip': 23, 'right_hip': 24
    }
    if isinstance(keypoints, (list, tuple)) and len(keypoints) > max(mp_indices.values()):
        try:
            ls = keypoints[mp_indices['left_shoulder']]
            rs = keypoints[mp_indices['right_shoulder']]
            lh = keypoints[mp_indices['left_hip']]
            rh = keypoints[mp_indices['right_hip']]
            def get_xy(pt):
                try:
                    return (pt[0], pt[1])
                except:
                    return None
            ls_xy = get_xy(ls)
            rs_xy = get_xy(rs)
            lh_xy = get_xy(lh)
            rh_xy = get_xy(rh)
            if ls_xy and rs_xy and lh_xy and rh_xy:
                shoulder_mid = np.array([(ls_xy[0] + rs_xy[0]) * 0.5, (ls_xy[1] + rs_xy[1]) * 0.5])
                hip_mid = np.array([(lh_xy[0] + rh_xy[0]) * 0.5, (lh_xy[1] + rh_xy[1]) * 0.5])
                trunk_vec = hip_mid - shoulder_mid
                trunk_vec = trunk_vec / (np.linalg.norm(trunk_vec) + 1e-6)
                trunk_angle = math.degrees(math.acos(abs(trunk_vec[1])))
        except Exception:
            trunk_angle = None
    
    # Điều kiện quyết định tư thế
    lying = False
    if angle_to_vertical > 60:
        lying = True
    if var_ratio > 1.0:
        lying = True
    if aspect_ratio > 0.8:
        lying = True
    if trunk_angle is not None and trunk_angle > 45:
        lying = True
    
    return "lying" if lying else "standing"


# =======================
# 2) Class GUI giữ nguyên, nhưng loại bỏ phần classify_pose_flexible cũ và gọi hàm standalone
# =======================
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
        
        # Xử lý trong thread riêng để GUI không bị treo
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        try:
            # 1) Chạy inference
            results = self.model(self.current_image, conf=float(self.confidence_var.get()), verbose=False)

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
                    keypoints_data = result.keypoints.data  # tensor hình (n_person, 17, 3)
                    detection_info.append(f"Found {len(keypoints_data)} keypoint set(s)")

                    for i, person_keypoints in enumerate(keypoints_data):
                        kpts = person_keypoints.cpu().numpy()  # shape (17, 3)
        
                        # Chuyển từ normalized (0–1) sang pixel
                        h, w = self.current_image.shape[:2]
                        kpts_pixel = kpts.copy()
                        kpts_pixel[:, 0] *= w
                        kpts_pixel[:, 1] *= h
                        
                        # Gọi classifier với keypoints đã chuyển đổi
                        pose_label = classify_pose_flexible(kpts_pixel)
                        pose_results.append(f"Person {i+1}: {pose_label}")

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
    
    # Phương thức classify_pose_flexible cũ trong class đã được loại bỏ/comment nếu có
    
    def draw_keypoints(self, img, keypoints, pose_label, person_id):
        """Draw keypoints và skeleton lên ảnh, kèm nhãn tư thế."""
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
        
        # Draw bounding box + aspect ratio cho debug
        visible_points = [(x, y) for x, y, c in keypoints if c > draw_threshold]
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
        """Thêm message vào textbox kết quả"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)


# =======================
# 3) Phần chạy chính
# =======================
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
                                                            "Phân loại tư thế: Đứng hoặc Nằm\n"
                                                            "Sử dụng Ultralytics YOLOv8\n"
                                                            "Hàm phân loại tư thế đơn giản"))
    
    root.mainloop()
