
from PIL import Image

# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"])

def is_image_file(filename):
    image_extensions = {"png", "jpg", "jpeg", "bmp", "gif", "webp", "svg", "ico", "tiff", "tif", "jfif"}
    ext = filename.split('.')[-1].lower()  # 获取最后一个小数点后的扩展名并转小写[2,4](@ref)
    return ext in image_extensions


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img
