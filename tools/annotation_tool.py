#!/usr/bin/env python3
"""
SpineCheck-AI Annotation Tool
This tool allows for manual annotation of vertebrae in X-ray images.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import json


class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("SpineCheck-AI Annotation Tool")

        # Setup variables
        self.image_path = None
        self.image = None
        self.tk_image = None
        self.points = []
        self.vertebrae = []
        self.current_vertebra = []
        self.canvas_image = None
        self.zoom_factor = 1.0

        # Create necessary directories
        self.create_directories()

        # Setup UI
        self.setup_ui()

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_dir = os.path.join(script_dir, "dataset", "raw")
        self.annotations_dir = os.path.join(script_dir, "dataset", "annotations")
        self.masks_dir = os.path.join(script_dir, "dataset", "masks")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

    def setup_ui(self):
        """Initialize the user interface."""
        # Create frames
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control panel
        tk.Label(self.control_frame, text="SpineCheck-AI Annotator", font=("Arial", 14, "bold")).pack(pady=10)

        file_frame = tk.Frame(self.control_frame)
        file_frame.pack(fill=tk.X, pady=10)

        tk.Button(file_frame, text="Load Image", command=self.load_image, bg="#4CAF50", fg="white", padx=10).pack(
            side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Browse Raw Dir", command=self.browse_raw_dir, bg="#2196F3", fg="white",
                  padx=10).pack(side=tk.LEFT, padx=5)

        # Vertebra controls
        tk.Label(self.control_frame, text="Vertebra Controls", font=("Arial", 12, "bold")).pack(pady=(20, 5))

        annotation_frame = tk.Frame(self.control_frame)
        annotation_frame.pack(fill=tk.X, pady=5)

        tk.Button(annotation_frame, text="Add Vertebra", command=self.add_vertebra, bg="#FF9800", fg="white",
                  padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(annotation_frame, text="Clear Current", command=self.clear_current, bg="#F44336", fg="white",
                  padx=10).pack(side=tk.LEFT, padx=5)

        save_frame = tk.Frame(self.control_frame)
        save_frame.pack(fill=tk.X, pady=10)

        tk.Button(save_frame, text="Save Annotation", command=self.save_annotation, bg="#673AB7", fg="white",
                  padx=10).pack(side=tk.LEFT, padx=5)
        tk.Button(save_frame, text="Generate Mask", command=self.generate_mask, bg="#009688", fg="white", padx=10).pack(
            side=tk.LEFT, padx=5)

        # Clear all button
        tk.Button(self.control_frame, text="Clear All", command=self.clear_all, bg="#E91E63", fg="white").pack(
            fill=tk.X, pady=10)

        # Zoom controls
        zoom_frame = tk.Frame(self.control_frame)
        zoom_frame.pack(fill=tk.X, pady=10)

        tk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                               padx=2)

        # Instructions
        instructions = (
            "Instructions:\n\n"
            "1. Load an X-ray image\n"
            "2. Click to mark points around a vertebra\n"
            "3. Press 'Add Vertebra' when done with one\n"
            "4. Repeat for all vertebrae\n"
            "5. Save the annotation\n"
            "6. Generate the binary mask"
        )
        instr_label = tk.Label(self.control_frame, text=instructions, justify=tk.LEFT,
                               bg="#f0f0f0", padx=10, pady=10, relief=tk.GROOVE)
        instr_label.pack(fill=tk.X, pady=10)

        # Vertebrae list
        tk.Label(self.control_frame, text="Annotated Vertebrae:", font=("Arial", 12, "bold")).pack(pady=(20, 5))

        self.vertebra_listbox = tk.Listbox(self.control_frame, height=10, bg="#f9f9f9", borderwidth=1)
        self.vertebra_listbox.pack(fill=tk.X, pady=5)

        tk.Button(self.control_frame, text="Remove Selected Vertebra", command=self.remove_selected_vertebra,
                  bg="#607D8B", fg="white").pack(fill=tk.X, pady=5)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to annotate")
        status_bar = tk.Label(self.control_frame, textvariable=self.status_var,
                              bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas for image display
        self.canvas = tk.Canvas(self.canvas_frame, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.root.bind("<Key>", self.on_key_press)

    def load_image(self):
        """Open file dialog to load an image."""
        self.image_path = filedialog.askopenfilename(
            initialdir=self.raw_dir,
            title="Select X-ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not self.image_path:
            return

        self.image = Image.open(self.image_path)
        self.original_image = self.image.copy()
        self.display_image()
        self.clear_all()

        # Update status
        filename = os.path.basename(self.image_path)
        self.status_var.set(f"Loaded image: {filename}")

    def browse_raw_dir(self):
        """Open the raw images directory."""
        try:
            if sys.platform == 'win32':
                os.startfile(self.raw_dir)
            elif sys.platform == 'darwin':  # macOS
                os.system(f'open "{self.raw_dir}"')
            else:  # Linux
                os.system(f'xdg-open "{self.raw_dir}"')
        except Exception as e:
            messagebox.showerror("Error", f"Could not open directory: {str(e)}")

    def display_image(self):
        """Display the loaded image on the canvas."""
        if self.image is None:
            return

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Resize image to fit canvas
        img_width, img_height = self.image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale * self.zoom_factor)
        new_height = int(img_height * scale * self.zoom_factor)

        resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # Display image on canvas
        if self.canvas_image:
            self.canvas.delete(self.canvas_image)

        self.canvas_image = self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.tk_image, anchor=tk.CENTER
        )

        # Redraw points
        self.redraw_points()

    def on_canvas_resize(self, event):
        """Handle canvas resize event."""
        self.display_image()

    def on_canvas_click(self, event):
        """Handle canvas click event to add points."""
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Get canvas size and image position
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate image position and size
        img_width, img_height = self.image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_factor
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        img_x = (canvas_width - new_width) // 2
        img_y = (canvas_height - new_height) // 2

        # Check if click is on the image
        if (img_x <= event.x <= img_x + new_width and
                img_y <= event.y <= img_y + new_height):
            # Convert to original image coordinates
            original_x = int((event.x - img_x) / scale)
            original_y = int((event.y - img_y) / scale)

            # Add point
            self.current_vertebra.append((original_x, original_y))

            # Update status
            self.status_var.set(f"Added point at ({original_x}, {original_y}). Points: {len(self.current_vertebra)}")

            # Redraw
            self.redraw_points()

    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.char == "z" and self.current_vertebra:  # Undo last point with 'z'
            self.current_vertebra.pop()
            self.redraw_points()
            self.status_var.set(f"Removed last point. Points: {len(self.current_vertebra)}")

    def add_vertebra(self):
        """Add current vertebra to the list."""
        if not self.current_vertebra:
            messagebox.showwarning("Warning", "No points selected for current vertebra.")
            return

        if len(self.current_vertebra) < 3:
            messagebox.showwarning("Warning", "Need at least 3 points to define a vertebra.")
            return

        self.vertebrae.append(self.current_vertebra)
        self.vertebra_listbox.insert(tk.END, f"Vertebra {len(self.vertebrae)}: {len(self.current_vertebra)} points")
        self.status_var.set(f"Added vertebra {len(self.vertebrae)}. Total vertebrae: {len(self.vertebrae)}")
        self.current_vertebra = []
        self.redraw_points()

    def remove_selected_vertebra(self):
        """Remove the selected vertebra from the list."""
        selection = self.vertebra_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "No vertebra selected.")
            return

        index = selection[0]
        self.vertebrae.pop(index)
        self.vertebra_listbox.delete(index)

        # Renumber the vertebrae in the listbox
        self.vertebra_listbox.delete(0, tk.END)
        for i, vertebra in enumerate(self.vertebrae):
            self.vertebra_listbox.insert(tk.END, f"Vertebra {i + 1}: {len(vertebra)} points")

        self.status_var.set(f"Removed vertebra. Total vertebrae: {len(self.vertebrae)}")
        self.redraw_points()

    def clear_current(self):
        """Clear the current vertebra being annotated."""
        if not self.current_vertebra:
            messagebox.showwarning("Warning", "No current vertebra to clear.")
            return

        self.current_vertebra = []
        self.redraw_points()
        self.status_var.set("Cleared current vertebra.")

    def clear_all(self):
        """Clear all vertebrae and reset the annotation."""
        self.vertebrae = []
        self.current_vertebra = []
        self.vertebra_listbox.delete(0, tk.END)
        self.canvas.delete("point", "line", "polygon")
        self.status_var.set("Cleared all annotations.")

    def redraw_points(self):
        """Redraw all points, lines, and polygons on the canvas."""
        if self.image is None:
            return

        self.canvas.delete("point", "line", "polygon")

        # Get canvas size and image position
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate image position and size
        img_width, img_height = self.image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * self.zoom_factor
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        img_x = (canvas_width - new_width) // 2
        img_y = (canvas_height - new_height) // 2

        # Draw stored vertebrae
        for i, vertebra in enumerate(self.vertebrae):
            # Draw polygon
            polygon_points = []
            for point in vertebra:
                x = int(point[0] * scale) + img_x
                y = int(point[1] * scale) + img_y
                polygon_points.extend([x, y])

            if len(polygon_points) >= 6:  # At least 3 points (6 coordinates)
                self.canvas.create_polygon(
                    polygon_points,
                    outline="#00BCD4",
                    fill="#BCD480",
                    width=2,
                    tags="polygon"
                )

            # Draw points
            for point in vertebra:
                x = int(point[0] * scale) + img_x
                y = int(point[1] * scale) + img_y

                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4,
                    fill="#2196F3", outline="white", tags="point"
                )

        # Draw current vertebra being annotated
        current_points = []
        for point in self.current_vertebra:
            x = int(point[0] * scale) + img_x
            y = int(point[1] * scale) + img_y
            current_points.extend([x, y])

            # Draw the point
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="#F44336", outline="white", tags="point"
            )

        # Draw lines between points of current vertebra
        if len(self.current_vertebra) >= 2:
            for i in range(len(self.current_vertebra) - 1):
                p1 = self.current_vertebra[i]
                p2 = self.current_vertebra[i + 1]

                x1 = int(p1[0] * scale) + img_x
                y1 = int(p1[1] * scale) + img_y
                x2 = int(p2[0] * scale) + img_x
                y2 = int(p2[1] * scale) + img_y

                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#FF9800", width=2, tags="line"
                )

            # Connect last point to first if at least 3 points
            if len(self.current_vertebra) >= 3:
                p1 = self.current_vertebra[-1]
                p2 = self.current_vertebra[0]

                x1 = int(p1[0] * scale) + img_x
                y1 = int(p1[1] * scale) + img_y
                x2 = int(p2[0] * scale) + img_x
                y2 = int(p2[1] * scale) + img_y

                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="#FF9800", width=2, dash=(4, 4), tags="line"
                )

    def zoom_in(self):
        """Zoom in on the image."""
        self.zoom_factor *= 1.2
        self.display_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.1f}x")

    def zoom_out(self):
        """Zoom out of the image."""
        self.zoom_factor /= 1.2
        self.zoom_factor = max(0.1, self.zoom_factor)  # Prevent too much zoom out
        self.display_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.1f}x")

    def reset_zoom(self):
        """Reset zoom to default level."""
        self.zoom_factor = 1.0
        self.display_image()
        self.status_var.set("Zoom reset to 1.0x")

    def save_annotation(self):
        """Save vertebrae annotations as JSON."""
        if not self.vertebrae and not self.current_vertebra:
            messagebox.showwarning("Warning", "No vertebrae to save.")
            return

        if self.image_path is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return

        # Make sure we add the current vertebra if it has points
        if self.current_vertebra and len(self.current_vertebra) >= 3:
            self.add_vertebra()

        # Get image filename without extension
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        # Save annotation as JSON
        annotation_path = os.path.join(self.annotations_dir, f"{base_filename}.json")

        annotation_data = {
            "image_path": self.image_path,
            "image_width": self.original_image.width,
            "image_height": self.original_image.height,
            "vertebrae": self.vertebrae
        }

        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)

        self.status_var.set(f"Annotation saved: {base_filename}.json")
        messagebox.showinfo("Success", f"Annotation saved to {annotation_path}")

    def generate_mask(self):
        """Generate binary mask from vertebrae annotations."""
        if not self.vertebrae:
            messagebox.showwarning("Warning", "No vertebrae annotated.")
            return

        if self.image_path is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return

        # Get image filename without extension
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        # Create a blank mask
        mask = Image.new('L', self.original_image.size, 0)
        draw = ImageDraw.Draw(mask)

        # Draw vertebrae on the mask
        for vertebra in self.vertebrae:
            # Convert list of points to polygon format
            polygon = []
            for point in vertebra:
                polygon.append(point)

            # Draw the polygon
            if len(polygon) >= 3:  # Need at least 3 points for a polygon
                draw.polygon(polygon, fill=255)

        # Save the mask
        mask_path = os.path.join(self.masks_dir, f"{base_filename}.png")
        mask.save(mask_path)

        self.status_var.set(f"Mask generated: {base_filename}.png")
        messagebox.showinfo("Success", f"Mask generated and saved to {mask_path}")

        # Display preview
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Mask Preview")

        # Create a frame for the images
        frame = tk.Frame(preview_window)
        frame.pack(padx=10, pady=10)

        # Display original image
        orig_img = self.original_image.copy()
        orig_img.thumbnail((400, 400))
        orig_tk = ImageTk.PhotoImage(orig_img)

        orig_label = tk.Label(frame, text="Original Image")
        orig_label.grid(row=0, column=0, padx=10, pady=5)

        orig_canvas = tk.Canvas(frame, width=orig_img.width, height=orig_img.height)
        orig_canvas.grid(row=1, column=0, padx=10, pady=5)
        orig_canvas.create_image(0, 0, anchor=tk.NW, image=orig_tk)

        # Display mask image
        mask_img = mask.copy()
        mask_img.thumbnail((400, 400))
        mask_tk = ImageTk.PhotoImage(mask_img)

        mask_label = tk.Label(frame, text="Binary Mask")
        mask_label.grid(row=0, column=1, padx=10, pady=5)

        mask_canvas = tk.Canvas(frame, width=mask_img.width, height=mask_img.height)
        mask_canvas.grid(row=1, column=1, padx=10, pady=5)
        mask_canvas.create_image(0, 0, anchor=tk.NW, image=mask_tk)

        # Keep references to images
        preview_window.orig_tk = orig_tk
        preview_window.mask_tk = mask_tk

        # Close button
        close_btn = tk.Button(preview_window, text="Close", command=preview_window.destroy)
        close_btn.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x800")
    app = AnnotationTool(root)
    root.mainloop()