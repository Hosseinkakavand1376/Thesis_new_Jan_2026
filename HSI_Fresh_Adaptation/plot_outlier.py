import fitz  # PyMuPDF
import os

# Define input and output paths
img1_path = "imgs/10_SG_MSC/outlier_detection_10_SG_MSC.pdf"
img2_path = "imgs/10_SG_SVN/outlier_detection_10_SG_SVN.pdf"
img3_path = "imgs/10_SG1_MSC/outlier_detection_10_SG1_MSC.pdf"
img4_path = "imgs/10_SG1_SVN/outlier_detection_10_SG1_SVN.pdf"
output_dir = "plots"
output_path = os.path.join(output_dir, "combined_outlier_detection.pdf")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the first pages of the PDFs as images
doc1 = fitz.open(img1_path)
doc2 = fitz.open(img2_path)
doc3 = fitz.open(img3_path)
doc4 = fitz.open(img4_path)

page1 = doc1.load_page(0)  # first page
page2 = doc2.load_page(0)
page3 = doc3.load_page(0)
page4 = doc4.load_page(0)

pix1 = page1.get_pixmap()
pix2 = page2.get_pixmap()
pix3 = page3.get_pixmap()
pix4 = page4.get_pixmap()


# Calculate grid size for 2 rows x 2 columns
row1_height = max(pix1.height, pix2.height)
row2_height = max(pix3.height, pix4.height)
col1_width = max(pix1.width, pix3.width)
col2_width = max(pix2.width, pix4.width)
combined_width = col1_width + col2_width
combined_height = row1_height + row2_height

# Create a new blank page
merged_doc = fitz.open()
merged_page = merged_doc.new_page(width=combined_width, height=combined_height)

# Insert images in 2x2 grid
# Top-left (img1)
merged_page.insert_image(fitz.Rect(0, 0, pix1.width, pix1.height), pixmap=pix1)
# Top-right (img2)
merged_page.insert_image(fitz.Rect(col1_width, 0, col1_width + pix2.width, pix2.height), pixmap=pix2)
# Bottom-left (img3)
merged_page.insert_image(fitz.Rect(0, row1_height, pix3.width, row1_height + pix3.height), pixmap=pix3)
# Bottom-right (img4)
merged_page.insert_image(fitz.Rect(col1_width, row1_height, col1_width + pix4.width, row1_height + pix4.height), pixmap=pix4)

# Save the result
merged_doc.save(output_path)
merged_doc.close()

print(f"Merged PDF saved to: {output_path}")