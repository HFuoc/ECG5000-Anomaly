# Deploy to Render

## 1. Create Render account

1. Đăng ký hoặc đăng nhập tại https://render.com
2. Chọn "New" → "Web Service"

## 2. Kết nối GitHub

1. Kết nối tài khoản GitHub của bạn với Render
2. Chọn repository `ECG5000-Anomaly`

## 3. Cấu hình service

- Name: `ecg-demo`
- Environment: `Python`
- Branch: `main`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn demo.app:app --host 0.0.0.0 --port $PORT --workers 1`

## 4. Xác nhận

Render sẽ build và deploy tự động khi bạn push `main`.

## 5. Kiểm tra app

Mở URL do Render cung cấp và test endpoint:

```bash
curl https://<your-service-name>.onrender.com/api/health
```
