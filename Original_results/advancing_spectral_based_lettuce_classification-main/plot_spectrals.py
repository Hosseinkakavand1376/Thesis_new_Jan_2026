import fitz  # PyMuPDF
import os

# Input paths
top_image_path = "imgs/10_SG_MSC/all_raw_spectra.pdf"
msc_path = "imgs/10_SG_MSC/cleaned_spectra_10_SG_MSC.pdf"
svn_path = "imgs/10_SG_SVN/cleaned_spectra_10_SG_SVN.pdf"
sg1_msc_path = "imgs/10_SG1_MSC/cleaned_spectra_10_SG1_MSC.pdf"
sg1_svn_path = "imgs/10_SG1_SVN/cleaned_spectra_10_SG1_SVN.pdf"

# Output path
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "spectral_signals.pdf")

# Load pages
top_doc = fitz.open(top_image_path)
msc_doc = fitz.open(msc_path)
svn_doc = fitz.open(svn_path)
sg1_msc_doc = fitz.open(sg1_msc_path)
sg1_svn_doc = fitz.open(sg1_svn_path)

top_pix = top_doc.load_page(0).get_pixmap()
msc_pix = msc_doc.load_page(0).get_pixmap()
svn_pix = svn_doc.load_page(0).get_pixmap()
sg1_msc_pix = sg1_msc_doc.load_page(0).get_pixmap()
sg1_svn_pix = sg1_svn_doc.load_page(0).get_pixmap()

# Dimensions
margin = 20
font_size = 16

row2_height = max(msc_pix.height, svn_pix.height)
row3_height = max(sg1_msc_pix.height, sg1_svn_pix.height)

row2_width = msc_pix.width + svn_pix.width + margin
row3_width = sg1_msc_pix.width + sg1_svn_pix.width + margin

total_width = max(top_pix.width, row2_width, row3_width)
total_height = top_pix.height + 2 * margin + row2_height + row3_height + 2 * font_size + 10

# Create output document and page
doc = fitz.open()
page = doc.new_page(width=total_width, height=total_height)

# Top row (centered)
top_x = (total_width - top_pix.width) // 2
page.insert_image(fitz.Rect(top_x, 0, top_x + top_pix.width, top_pix.height), pixmap=top_pix)

# Second row (MSC, SVN)
row2_y = top_pix.height + margin
msc_x = (total_width - (msc_pix.width + svn_pix.width + margin)) // 2
svn_x = msc_x + msc_pix.width + margin
page.insert_image(fitz.Rect(msc_x, row2_y, msc_x + msc_pix.width, row2_y + msc_pix.height), pixmap=msc_pix)
page.insert_image(fitz.Rect(svn_x, row2_y, svn_x + svn_pix.width, row2_y + svn_pix.height), pixmap=svn_pix)

# Third row (SG1_MSC, SG1_SVN)
row3_y = row2_y + row2_height + font_size + 10 + margin
sg1_msc_x = (total_width - (sg1_msc_pix.width + sg1_svn_pix.width + margin)) // 2
sg1_svn_x = sg1_msc_x + sg1_msc_pix.width + margin
page.insert_image(fitz.Rect(sg1_msc_x, row3_y, sg1_msc_x + sg1_msc_pix.width, row3_y + sg1_msc_pix.height), pixmap=sg1_msc_pix)
page.insert_image(fitz.Rect(sg1_svn_x, row3_y, sg1_svn_x + sg1_svn_pix.width, row3_y + sg1_svn_pix.height), pixmap=sg1_svn_pix)
# Labels
page.insert_text(fitz.Point(sg1_msc_x, row3_y + sg1_msc_pix.height + 5), f"SG1_MSC: {sg1_msc_path}", fontsize=font_size, color=(0, 0, 0))
page.insert_text(fitz.Point(sg1_svn_x, row3_y + sg1_svn_pix.height + 5), f"SG1_SVN: {sg1_svn_path}", fontsize=font_size, color=(0, 0, 0))

# Save the result
doc.save(output_path)
doc.close()

print(f"Spectral signals PDF saved to: {output_path}")