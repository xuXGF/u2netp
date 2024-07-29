import os

from PIL import Image

def mask_cut_image(image_path):
    # 加载原图和掩码图像
    aaa=image_path.split(os.sep)[-1].split(".")
    bbb = aaa[0:-1]
    img_name = bbb[0]
    for i in range(1, len(bbb)):
        img_name = img_name + "." + bbb[i]

    image = Image.open(image_path).convert('RGBA')
    mask = Image.open(f'results/{img_name}.png').convert('L')  # 转换为灰度模式

    # 确保掩码图像与原图尺寸一致
    mask = mask.resize(image.size)

    # 创建一个完全透明的图像
    transparent_image = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # 使用掩码将透明图像与原图合成
    composite = Image.composite(image, transparent_image, mask)

    # 保存结果图像
    composite.save(f'results/{img_name}_vut.png')
