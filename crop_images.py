import os
import argparse
from PIL import Image, ImageOps

def process_images(input_dir, output_dir, size=(200, 200)):
    """
    Crop và resize tất cả ảnh trong input_dir thành size (width, height)
    và lưu vào output_dir. Giữ nguyên tỷ lệ và crop từ trung tâm.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    count = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                with Image.open(input_path) as img:
                    # Chuyển đổi sang hệ màu RGB nếu ảnh đang ở mode RGBA hoặc Palette
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # ImageOps.fit tự động resize và crop từ tâm ảnh để lấp đầy kích thước 200x200
                    processed_img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
                    
                    # Lưu ảnh với chất lượng cao
                    processed_img.save(output_path, quality=95)
                    print(f"✅ Đã xử lý: {filename}")
                    count += 1
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {filename}: {e}")
                
    print(f"\n🎉 Đã hoàn thành! Xử lý thành công {count} ảnh. Ảnh mới nằm trong thư mục '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script crop và resize ảnh hàng loạt.")
    parser.add_argument("-i", "--input", default="input_images", help="Thư mục chứa ảnh gốc (Mặc định: input_images)")
    parser.add_argument("-o", "--output", default="output_images", help="Thư mục lưu ảnh đã xử lý (Mặc định: output_images)")
    parser.add_argument("-s", "--size", type=int, default=200, help="Kích thước ảnh vuông muốn cắt (Mặc định: 200)")
    
    args = parser.parse_args()
    
    input_directory = args.input
    output_directory = args.output
    target_size = (args.size, args.size)
    
    # Tạo thư mục input nếu chưa có để người dùng biết chỗ copy ảnh vào
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        print(f"⚠️ Đã tạo thư mục '{input_directory}'. Vui lòng copy ảnh của bạn vào thư mục này rồi chạy lại script.")
    else:
        print(f"🚀 Bắt đầu crop ảnh từ '{input_directory}' về kích thước {target_size[0]}x{target_size[1]}...")
        process_images(input_directory, output_directory, target_size)
