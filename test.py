import os
import comtypes.client
from tkinter import Tk, filedialog, Button, Label

# 将PPT转为PDF的函数
def ppt_to_pdf(ppt_path, pdf_path):
    powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
    powerpoint.Visible = 1
    presentation = powerpoint.Presentations.Open(ppt_path)
    presentation.SaveAs(pdf_path, 32)  # 32代表 PDF 格式
    presentation.Close()
    powerpoint.Quit()

# GUI程序
def select_ppt():
    ppt_path = filedialog.askopenfilename(title="选择PPT文件", filetypes=[("PPT files", "*.pptx")])
    if ppt_path:
        pdf_path = os.path.splitext(ppt_path)[0] + ".pdf"
        ppt_to_pdf(ppt_path, pdf_path)
        label.config(text=f"成功转换为: {pdf_path}")

# 创建GUI
root = Tk()
root.title("PPT to PDF Converter")

label = Label(root, text="请选择PPT文件")
label.pack(pady=10)

btn = Button(root, text="选择PPT文件", command=select_ppt)
btn.pack(pady=10)

root.mainloop()
