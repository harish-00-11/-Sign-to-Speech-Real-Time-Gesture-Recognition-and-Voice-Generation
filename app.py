import tkinter as tk
from tkinter import messagebox, ttk
import os
import subprocess
import random
import string
import sys


class ModernApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gesture Control Application")
        self.geometry("600x500")
        self.configure(bg='#f5f5f5')
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#03A9F4',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#f44336',
            'light': '#f5f5f5',
            'dark': '#333333',
            'text': '#212121'
        }

        # Create header
        self.header_frame = tk.Frame(self, bg=self.colors['primary'], height=60)
        self.header_frame.pack(fill='x')

        header_label = tk.Label(
            self.header_frame,
            text="Gesture Control System",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            pady=10
        )
        header_label.pack()

        # Create main container with padding
        self.container = tk.Frame(self, bg=self.colors['light'])
        self.container.pack(fill='both', expand=True, padx=20, pady=20)

        # Main menu frame
        self.main_frame = tk.Frame(self.container, bg=self.colors['light'])
        self.main_frame.pack(pady=20, fill='both', expand=True)

        # Title for main menu
        main_title = tk.Label(
            self.main_frame,
            text="Select an Option",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['light'],
            fg=self.colors['text']
        )
        main_title.pack(pady=(0, 20))

        # Button frame for better alignment
        button_frame = tk.Frame(self.main_frame, bg=self.colors['light'])
        button_frame.pack()

        # Create modern styled buttons
        self.create_styled_button(button_frame, "Hand Gesture Recognition", self.colors['primary'],
                                  "👋", self.show_gesture_menu)

        self.create_styled_button(button_frame, "Sign Language Detection", self.colors['primary'],
                                  "✌️", self.show_sign_menu)

        self.create_styled_button(button_frame, "Air Gesture Control", self.colors['primary'],
                                  "🖐️", self.show_air_menu)

        self.create_styled_button(button_frame, "Exit Application", self.colors['danger'],
                                  "❌", self.confirm_exit)

        # Footer with version info
        footer_frame = tk.Frame(self, bg='#e0e0e0', height=30)
        footer_frame.pack(fill='x', side='bottom')

        version_label = tk.Label(
            footer_frame,
            text="v1.2.0",
            font=('Helvetica', 8),
            bg='#e0e0e0',
            fg='#757575'
        )
        version_label.pack(side='right', padx=10, pady=5)

        # Initialize submenu frames
        self.gesture_frame = None
        self.sign_frame = None
        self.air_frame = None
        self.captcha_window = None
        self.current_module = ""

    def create_styled_button(self, parent, text, color, icon, command):
        """Create a modern button with icon and text"""
        frame = tk.Frame(parent, bg=self.colors['light'], pady=10)
        frame.pack(fill='x')

        button = tk.Button(
            frame,
            text=f" {icon} {text}",
            font=('Helvetica', 12),
            bg=color,
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            width=25,
            cursor="hand2",
            activebackground=self.darken_color(color),
            activeforeground='white',
            command=command
        )
        button.pack()

        # Add hover effect
        button.bind("<Enter>", lambda e: button.config(bg=self.darken_color(color)))
        button.bind("<Leave>", lambda e: button.config(bg=color))

    def darken_color(self, hex_color, factor=0.8):
        """Darken a hex color by the given factor"""
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)

        # Darken
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'

    def confirm_exit(self):
        """Show confirmation before exiting"""
        response = messagebox.askyesno(
            "Confirm Exit",
            "Are you sure you want to exit the application?",
            icon=messagebox.QUESTION
        )
        if response:
            self.destroy()

    def clear_frame(self, frame):
        if frame:
            frame.pack_forget()
        self.main_frame.pack(pady=20, fill='both', expand=True)

    def show_submenu(self, title, options, frame_var):
        """Generic function to display a submenu"""
        # Hide main menu
        self.main_frame.pack_forget()

        # Create new frame if it doesn't exist
        if not getattr(self, frame_var):
            new_frame = tk.Frame(self.container, bg=self.colors['light'])
            setattr(self, frame_var, new_frame)

            # Add submenu title
            submenu_title = tk.Label(
                new_frame,
                text=title,
                font=('Helvetica', 14, 'bold'),
                bg=self.colors['light'],
                fg=self.colors['text']
            )
            submenu_title.pack(pady=(0, 20))

            # Add buttons from options
            for opt in options:
                if opt['text'] == "Back":
                    color = self.colors['warning']
                    icon = "⬅️"
                elif opt['text'] == "Close":
                    color = self.colors['danger']
                    icon = "❌"
                else:
                    color = self.colors['primary']
                    icon = "▶️"

                self.create_styled_button(new_frame, opt['text'], color, icon, opt['command'])

        # Show the submenu
        getattr(self, frame_var).pack(pady=20, fill='both', expand=True)

    def show_gesture_menu(self):
        options = [
            {"text": "Training Module", "command": lambda: self.verify_captcha("Training.py")},
            {"text": "Gesture GUI", "command": lambda: self.run_module("gui.py")},
            {"text": "Diagnostics & Analysis", "command": lambda: self.run_module("dia&ana.py")},
            {"text": "Back", "command": lambda: self.clear_frame(self.gesture_frame)},
            {"text": "Close", "command": self.confirm_exit}
        ]
        self.show_submenu("Hand Gesture Recognition", options, "gesture_frame")

    def show_sign_menu(self):
        options = [
            {"text": "Sign Language Training", "command": lambda: self.verify_captcha("sign_training.py")},
            {"text": "Alert System", "command": lambda: self.run_module("gesture_alert.py")},
            {"text": "Back", "command": lambda: self.clear_frame(self.sign_frame)},
            {"text": "Close", "command": self.confirm_exit}
        ]
        self.show_submenu("Sign Language Detection", options, "sign_frame")

    def show_air_menu(self):
        options = [
            {"text": "Air Gesture Control", "command": lambda: self.run_module("air_ges.py")},
            {"text": "Back", "command": lambda: self.clear_frame(self.air_frame)},
            {"text": "Close", "command": self.confirm_exit}
        ]
        self.show_submenu("Air Gesture Control", options, "air_frame")

    def verify_captcha(self, module_name):
        self.current_module = module_name

        if self.captcha_window:
            self.captcha_window.destroy()

        # Create modern captcha window
        self.captcha_window = tk.Toplevel(self)
        self.captcha_window.title("Security Verification")
        self.captcha_window.geometry("400x250")
        self.captcha_window.resizable(False, False)
        self.captcha_window.configure(bg=self.colors['light'])
        self.captcha_window.transient(self)
        self.captcha_window.grab_set()

        # Add header to captcha window
        captcha_header = tk.Frame(self.captcha_window, bg=self.colors['primary'], height=40)
        captcha_header.pack(fill='x')

        captcha_title = tk.Label(
            captcha_header,
            text="CAPTCHA Verification",
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            pady=8
        )
        captcha_title.pack()

        captcha_content = tk.Frame(self.captcha_window, bg=self.colors['light'], padx=30, pady=20)
        captcha_content.pack(fill='both', expand=True)

        # Instructions
        instructions = tk.Label(
            captcha_content,
            text="Please enter the text shown below:",
            font=('Helvetica', 10),
            bg=self.colors['light'],
            fg=self.colors['text']
        )
        instructions.pack(pady=(0, 10))

        # Generate CAPTCHA text
        self.captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

        # Create styled captcha display
        captcha_display_frame = tk.Frame(
            captcha_content,
            bg='#e0e0e0',
            highlightbackground=self.colors['primary'],
            highlightthickness=1,
            padx=15,
            pady=8
        )
        captcha_display_frame.pack(pady=10)

        captcha_display = tk.Label(
            captcha_display_frame,
            text=self.captcha_text,
            font=('Courier New', 22, 'bold'),
            bg='#e0e0e0',
            fg=self.colors['dark']
        )
        captcha_display.pack()

        # Entry field with border
        entry_frame = tk.Frame(
            captcha_content,
            bg=self.colors['light']
        )
        entry_frame.pack(pady=10)

        self.captcha_entry = ttk.Entry(
            entry_frame,
            font=('Helvetica', 12),
            width=15,
            justify='center'
        )
        self.captcha_entry.pack()
        self.captcha_entry.focus()

        # Button frame
        button_frame = tk.Frame(captcha_content, bg=self.colors['light'])
        button_frame.pack(pady=10)

        # Verify button
        verify_btn = tk.Button(
            button_frame,
            text="Verify",
            bg=self.colors['success'],
            fg='white',
            font=('Helvetica', 10, 'bold'),
            padx=15,
            pady=5,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.check_captcha
        )
        verify_btn.pack(side='left', padx=5)

        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            bg='#9e9e9e',
            fg='white',
            font=('Helvetica', 10),
            padx=15,
            pady=5,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.captcha_window.destroy
        )
        cancel_btn.pack(side='left', padx=5)

        # Bind enter key to check_captcha
        self.captcha_window.bind('<Return>', lambda event: self.check_captcha())

    def check_captcha(self):
        user_input = self.captcha_entry.get().strip().upper()
        if user_input == self.captcha_text:
            self.captcha_window.destroy()
            self.run_module(self.current_module)
        else:
            messagebox.showerror(
                "Verification Failed",
                "Incorrect verification code. Please try again.",
                parent=self.captcha_window
            )
            self.captcha_entry.delete(0, tk.END)

            # Generate new CAPTCHA
            self.captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

            # Find and update the label containing the CAPTCHA
            for widget in self.captcha_window.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Frame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, tk.Label) and len(grandchild.cget("text")) == 6:
                                    grandchild.config(text=self.captcha_text)

    def run_module(self, module_name):
        try:
            full_path = os.path.join(os.getcwd(), module_name)
            print(f"Attempting to run module: {full_path}")
            print("Python executable:", sys.executable)

            if os.path.exists(full_path):
                subprocess.Popen([sys.executable, full_path])
                messagebox.showinfo(
                    "Module Launched",
                    f"Successfully launched {module_name}",
                    parent=self
                )
            else:
                messagebox.showerror(
                    "Module Not Found",
                    f"The module '{module_name}' could not be found in the current directory.",
                    parent=self
                )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Error launching {module_name}:\n{str(e)}",
                parent=self
            )
            print(f"Error running {module_name}: {e}")


if __name__ == "__main__":
    app = ModernApplication()
    app.mainloop()