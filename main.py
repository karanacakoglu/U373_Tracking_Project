import torch
import cv2
import numpy as np
import os
from model import UNet
from tracker import CentroidTracker
from segmenter import get_image_paths, preprocess_frame

# Ayarlar
MODEL_PATH = "unet_cell_seg.pth"
DATA_PATH = "data/PhC-C2DH-U373/01"
SAVE_PATH = "hucre_takip_sonuc.mp4"  # Kaydedilecek dosya adı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model başarıyla yüklendi!")
    except:
        print("HATA: Model dosyası bulunamadı!")
        return

    model.eval()
    tracker = CentroidTracker(max_disappeared=30)
    image_files = get_image_paths(DATA_PATH)
    traces = {}

    # --- VİDEO KAYIT AYARLARI ---
    # İlk kareyi okuyup boyutlarını alalım
    first_frame = preprocess_frame(image_files[0])
    height, width = first_frame.shape

    # Video kaydediciyi başlat (FPS: 10, Codec: mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_PATH, fourcc, 10.0, (width, height))

    print(f"Kayıt başladı: {SAVE_PATH}")

    for path in image_files:
        frame = preprocess_frame(path)
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Normalizasyon ve Model Tahmini
        frame_float = frame.astype(np.float32)
        frame_norm = (frame_float - np.min(frame_float)) / (np.max(frame_float) - np.min(frame_float) + 1e-7)
        input_tensor = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            output = model(input_tensor)
            mask = output.squeeze().cpu().numpy()

        # Maske ve Morfoloji
        mask_binary = (mask > 0.9).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centroids.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

        objects = tracker.update(centroids)

        # Çizim ve İzleme (Senin kodun)
        for (object_id, centroid) in objects.items():
            if object_id not in traces: traces[object_id] = []
            traces[object_id].append(tuple(centroid))

            if len(traces[object_id]) > 2:
                points = np.array(traces[object_id][-20:], np.int32)
                cv2.polylines(display_frame, [points], False, (0, 255, 0), 1)

            cv2.putText(display_frame, f"ID:{object_id}", (centroid[0] - 15, centroid[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.circle(display_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv2.putText(display_frame, f"Canli Hucre Sayisi: {len(objects)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- KAREYİ VİDEOYA YAZ ---
        out.write(display_frame)

        cv2.imshow("Kayıt Ediliyor...", display_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # İşlem bittiğinde serbest bırak
    out.release()
    cv2.destroyAllWindows()
    print(f"Video başarıyla kaydedildi: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    main()