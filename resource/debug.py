import subprocess
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

# Path to your texdiag.exe (edit if needed)
TEXDIAG_PATH = r"texdiag.exe"

def pick_file():
    filepath = filedialog.askopenfilename(
        title="Select DDS File",
        filetypes=[("DDS Textures", "*.dds"), ("All Files", "*.*")]
    )
    if not filepath:
        return

    try:
        # Must call one of: info, analyze, compare, diff, dumpbc, or dumpdds
        result = subprocess.run(
            [TEXDIAG_PATH, "info", filepath],
            capture_output=True,
            text=True
        )
    except FileNotFoundError:
        messagebox.showerror("Error", f"Could not find {TEXDIAG_PATH}\nCheck your path.")
        return

    # Display the result
    output_box.delete(1.0, tk.END)
    if result.stdout.strip():
        output_box.insert(tk.END, result.stdout)
    if result.stderr.strip():
        output_box.insert(tk.END, "\n[Errors/Warnings]\n" + result.stderr)


# ---- UI ----
root = tk.Tk()
root.title("DDS Metadata Viewer (texdiag info)")
root.geometry("900x650")

open_button = tk.Button(root, text="Select DDS File", command=pick_file, height=2)
open_button.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
output_box.pack(expand=True, fill="both")

root.mainloop()
