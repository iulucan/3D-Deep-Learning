import face_recognition
import cv2

# Görüntüyü yükle
image_path = "people.jpg"  # analiz edilecek görsel
image = face_recognition.load_image_file(image_path)

# Yüz konumlarını bul
face_locations = face_recognition.face_locations(image)

print(f"{len(face_locations)} yüz bulundu.")

# Görseli OpenCV formatına çevir
image_cv2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Yüzlerin etrafına dikdörtgen çiz
for i, face in enumerate(face_locations):
    top, right, bottom, left = face
    cv2.rectangle(image_cv2, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_cv2, f"Face {i+1}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Sonucu göster
cv2.imshow("Yüz Tanıma", image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
