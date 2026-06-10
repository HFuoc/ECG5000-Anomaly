import pandas as pd
import os

def process_kaggle_data():
    print("🔄 Đang xử lý dữ liệu từ Kaggle...")
    # Điều chỉnh tên file trong ngoặc kép nếu file bạn tải về có tên khác
    train_path = "data/raw/ECG5000_TRAIN.txt"
    test_path = "data/raw/ECG5000_TEST.txt"
    
    # Đọc dữ liệu (ECG5000 thường phân tách các cột bằng khoảng trắng)
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)
    
    # Gộp toàn bộ thành 1 DataFrame duy nhất (5000 dòng)
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Lưu ra file CSV
    csv_path = "data/raw/ECG5000_full.csv"
    full_df.to_csv(csv_path, index=False, header=False)
    
    print(f"✅ Đã gộp và lưu thành công file CSV tại: {csv_path}")
    print(f"📊 Kích thước bộ dữ liệu: {full_df.shape}")

if __name__ == "__main__":
    process_kaggle_data()