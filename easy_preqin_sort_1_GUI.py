import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

def process_file():
    # Get the file path from the entry widget
    file_path = file_path_entry.get()
    
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Process the DataFrame (example: selecting and filtering columns)
        df = df[["ASSET CLASS", "FUND ID", "FUND SIZE (USD MN)", "GEOGRAPHIC FOCUS"]]
        df = df[df['GEOGRAPHIC FOCUS'] != 'Global']  # Example filter
        
        # Group and aggregate data
        df_grouped = df.groupby("ASSET CLASS").agg({
            "FUND ID": 'count',
            "FUND SIZE (USD MN)": 'sum'
        }).reset_index()
        
        # Clear the treeview
        for row in tree.get_children():
            tree.delete(row)
        
        # Insert DataFrame into the treeview
        tree["columns"] = list(df_grouped.columns)
        tree["show"] = "headings"
        
        for col in df_grouped.columns:
            tree.heading(col, text=col)
        
        for _, row in df_grouped.iterrows():
            tree.insert("", "end", values=list(row))
    
    except Exception as e:
        # Show an error message if something goes wrong
        messagebox.showerror("Error", f"Failed to process the file.\n\n{str(e)}")

# Create the main Tkinter window
root = tk.Tk()
root.title("Excel File Processor")

# Add an entry field to paste the file path
tk.Label(root, text="Enter Excel File Path:").pack(pady=5)
file_path_entry = tk.Entry(root, width=50)
file_path_entry.pack(pady=5)

# Add a button to process the file
process_button = tk.Button(root, text="Process File", command=process_file)
process_button.pack(pady=10)

# Add a treeview to display the DataFrame
tree = ttk.Treeview(root)
tree.pack(pady=10, fill="both", expand=True)

# Start the Tkinter main loop
root.mainloop()
