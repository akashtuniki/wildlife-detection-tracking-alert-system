import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# -----------------------------
# 1. Load trained model
# -----------------------------
model = YOLO("runs/detect/train/weights/best.pt")

# -----------------------------
# 2. Input video
# -----------------------------
video_path = "videos/lion.mp4"

cap = cv2.VideoCapture(video_path)

# -----------------------------
# 3. Video properties
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------
# 4. Create output folders
# -----------------------------
alert_folder = "outputs/alerts"
os.makedirs(alert_folder, exist_ok=True)

# -----------------------------
# 5. Create CSV event log
# -----------------------------
csv_file = "outputs/alerts.csv"

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Time",
            "Animal",
            "Track_ID",
            "Confidence",
            "Event"
        ])

# -----------------------------
# 6. Output video
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "runs/detect/zone_alert_output.mp4",
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# 7. Restricted zone
# -----------------------------
zone_x1 = int(width * 0.55)
zone_y1 = int(height * 0.25)

zone_x2 = int(width * 0.95)
zone_y2 = int(height * 0.85)

# Keep track of animals that already triggered
alerted_ids = set()

# -----------------------------
# 8. Process video
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    # -----------------------------
    # Draw restricted zone
    # -----------------------------
    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # Process detections
    # -----------------------------
    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Class
            cls_id = int(box.cls[0])

            # Confidence
            confidence = float(box.conf[0])

            # Tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            # Animal name
            animal_name = model.names[cls_id]

            # -----------------------------
            # Calculate center
            # -----------------------------
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # -----------------------------
            # Check restricted zone
            # -----------------------------
            inside_zone = (
                zone_x1 <= center_x <= zone_x2
                and
                zone_y1 <= center_y <= zone_y2
            )

            # -----------------------------
            # Draw normal bounding box
            # -----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = (
                f"{animal_name} "
                f"{confidence:.2f} "
                f"ID:{track_id}"
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # -----------------------------
            # ALERT
            # -----------------------------
            if inside_zone:

                # Red bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                # Alert message
                cv2.putText(
                    frame,
                    "ALERT! ANIMAL IN ZONE",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                # -----------------------------
                # Save screenshot only once
                # for each tracking ID
                # -----------------------------
                if track_id != -1 and track_id not in alerted_ids:

                    alerted_ids.add(track_id)

                    # Current time
                    current_time = datetime.now()

                    time_string = current_time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Screenshot filename
                    filename = (
                        f"alert_{track_id}_"
                        f"{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    )

                    screenshot_path = os.path.join(
                        alert_folder,
                        filename
                    )

                    # Save screenshot
                    cv2.imwrite(
                        screenshot_path,
                        frame
                    )

                    # -----------------------------
                    # Save event to CSV
                    # -----------------------------
                    with open(
                        csv_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow([
                            time_string,
                            animal_name,
                            track_id,
                            round(confidence, 2),
                            "Zone Entry"
                        ])

                    print(
                        f"ALERT: {animal_name} "
                        f"ID:{track_id} entered zone"
                    )

                    print(
                        f"Screenshot saved: "
                        f"{screenshot_path}"
                    )

    # -----------------------------
    # Show video
    # -----------------------------
    cv2.imshow(
        "Wildlife Detection - Alert System",
        frame
    )

    # Save processed video
    out.write(frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# 9. Release resources
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nProcessing completed!")
print("Alert screenshots: outputs/alerts/")
print("Event log: outputs/alerts.csv")
print("Output video: runs/detect/zone_alert_output.mp4")