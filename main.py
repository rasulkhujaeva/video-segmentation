import cv2
import numpy as np
from sklearn.cluster import KMeans

# Video dosyasını aç
video = cv2.VideoCapture("traffic_video.mp4")

frame_count = 0

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Her 10 karede bir işlem yap
    if frame_count % 10 == 0:

        # Görüntüyü yeniden boyutlandır
        frame = cv2.resize(frame, (640, 480))

        # Gürültü azaltma
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # RGB -> Lab dönüşümü
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # Piksel verisini yeniden şekillendir
        pixel_values = lab.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # K-Means segmentasyonu
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(pixel_values)

        centers = np.uint8(kmeans.cluster_centers_)
        segmented_data = centers[labels.flatten()]

        segmented_image = segmented_data.reshape(frame.shape)

        # Hareket tespiti için gri tonlama
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hareket analizi
        edges = cv2.Canny(gray, 100, 200)

        # Sonuçları göster
        cv2.imshow("Original Video", frame)
        cv2.imshow("Segmented Video", segmented_image)
        cv2.imshow("Motion Detection", edges)

    frame_count += 1

    # Çıkış için q tuşu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
cv2.imwrite("segmented_output.png", segmented_image)
cv2.imwrite("motion_detection.png", edges)
cv2.imwrite("original_frame.png", frame)
