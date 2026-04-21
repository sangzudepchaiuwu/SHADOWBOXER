<h1 align="center">SHADOW BOXER
 ỨNG DỤNG CỦA THỊ GIÁC MÁY TÍNH TRONG TRÒ CHƠI</h1>

<div align="center">

<p align="center">
  <img src="static/images/logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="static/images/LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

<h2 align="center">SHADOW BOXER</h2>

---
## 🌟 Giới thiệu
# Shadow Boxer là một tựa game đối kháng góc nhìn thứ nhất (First-Person Perspective - FPP) thông qua sức mạnh của AI và Thị giác máy tính (Computer Vision).

Người chơi sẽ nhập vai vào một võ sĩ dưới góc nhìn thứ nhất. Vung các nắm đấm để tấn công đối thủ !

## 🚀 Core gameplay

* **Hệ thống chiến đấu:**  
  - Sử dụng webcam để nhận diện chuyển động tay thông qua **MediaPipe Pose**  
  - Phát hiện cú đấm dựa trên:
    - Tốc độ tay (velocity)
    - Độ sâu (Z-axis)
    - Góc duỗi tay (shoulder–elbow–wrist)  
  - Hỗ trợ:
    - Đấm trái / phải / combo
    - Đỡ đòn (blocking pose)
  - Hệ thống sát thương:
    - Đấm đầu: damage cao
    - Đấm thân: damage trung bình
    - Đỡ đòn giảm hoặc chặn sát thương

* **AI đối thủ:**  
  - Tự động:
    - Tấn công theo thời gian ngẫu nhiên
    - Chặn đòn (block) với xác suất  
  - Thích ứng theo stage:
    - Stage 1: dễ
    - Stage 2: block nhiều
    - Stage 3: nhanh + damage cao  
  - Sát thương enemy có thể scale theo người chơi

* **HUD & Feedback:**  
  - Thanh máu Player / Enemy  
  - Hiệu ứng hit, block, damage  
  - Notification động (floating text, fade animation)  
  - Camera debug hiển thị skeleton realtime  

* **Cốt truyện (Story Mode):**  
  - Intro video + hội thoại (dialogue system)  
  - Bối cảnh: đấu trường ngầm “Shadow Ring” trong nhà tù  
  - Progression theo từng stage  

* **Các màn chơi:**  
  - 🥊 Stage 1 – Rookie Brawl  
  - 💪 Stage 2 – Prison Enforcer  
  - 👑 Stage 3 – The Champion  
  - 🏆 Kết thúc: Champion of the Shadow Ring  

---

## 📁 Cấu trúc Dự án

---

## ⚙️ Công nghệ sử dụng

- **Python**
- **OpenCV** – xử lý camera  
- **MediaPipe** – nhận diện pose (AI)  
- **Pygame** – render game + UI  
- **NumPy** – xử lý toán học  

---

## 🖥️ Cài đặt Môi trường

### 1. Clone project
```bash
git clone <repo-url>
cd Shadow-Boxer
```
### 2. Tạo môi trường ảo
```
python -m venv .venv
```
### 3. Kích hoạt môi trường
```
# Windows
.venv\Scripts\activate

# Linux / MacOS
source .venv/bin/activate
```
### 4. Cài thư viện
```
pip install requirements.txt
```
### 5. Chạy chương trình
```
python shadow_boxer_ai.py
```








