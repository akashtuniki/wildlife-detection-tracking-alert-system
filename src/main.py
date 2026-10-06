import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# 1. LOAD MODEL
# ==========================================

MODEL_PATH = "runs/detect/train/weights/best.pt"

model = YOLO(MODEL_PATH)


# ==========================================
# 2. INPUT VIDEO
# ==========================================

VIDEO_PATH = "videos/elephant1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


# ==========================================
# 3. VIDEO INFORMATION
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Video opened successfully!")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")


# ==========================================
# 4. CREATE FOLDERS
# ==========================================

os.makedirs("outputs/alerts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ==========================================
# 5. CSV EVENT LOG
# ==========================================

CSV_PATH = "outputs/alerts.csv"

if not os.path.exists(CSV_PATH):

    with open(CSV_PATH, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Time",
            "Animal",
            "Track_ID",
            "Confidence",
            "Event"
        ])


# ==========================================
# 6. OUTPUT VIDEO
# ==========================================

OUTPUT_VIDEO = "runs/detect/final_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ==========================================
# 7. RESTRICTED ZONE
# ==========================================

zone_x1 = int(width * 0.55)
zone_y1 = int(height * 0.25)

zone_x2 = int(width * 0.95)
zone_y2 = int(height * 0.85)


# ==========================================
# 8. PREVENT DUPLICATE ALERTS
# ==========================================

alerted_ids = set()


# ==========================================
# 9. PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------
    # YOLO + BYTETRACK
    # --------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )


    # ======================================
    # DRAW RESTRICTED ZONE
    # ======================================

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


    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            # --------------------------------
            # Bounding box
            # --------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            # --------------------------------
            # Class
            # --------------------------------

            cls_id = int(box.cls[0])


            # --------------------------------
            # Confidence
            # --------------------------------

            confidence = float(box.conf[0])


            # --------------------------------
            # Tracking ID
            # --------------------------------

            if box.id is not None:

                track_id = int(box.id[0])

            else:

                track_id = -1


            # --------------------------------
            # Animal name
            # --------------------------------

            animal_name = model.names[cls_id]


            # =================================
            # CENTER OF ANIMAL
            # =================================

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2


            # =================================
            # CHECK RESTRICTED ZONE
            # =================================

            inside_zone = (

                zone_x1 <= center_x <= zone_x2

                and

                zone_y1 <= center_y <= zone_y2

            )


            # =================================
            # NORMAL BOUNDING BOX
            # =================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # =================================
            # LABEL
            # =================================

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


            # =================================
            # ALERT
            # =================================

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


                # =================================
                # SAVE ALERT ONLY ONCE PER ID
                # =================================

                if (
                    track_id != -1
                    and
                    track_id not in alerted_ids
                ):

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
                        "outputs/alerts",
                        filename
                    )


                    # Save screenshot

                    cv2.imwrite(
                        screenshot_path,
                        frame
                    )


                    # =================================
                    # SAVE EVENT TO CSV
                    # =================================

                    with open(
                        CSV_PATH,
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


                    print()
                    print("================================")
                    print("🚨 ALERT GENERATED")
                    print("================================")
                    print(f"Animal     : {animal_name}")
                    print(f"Track ID   : {track_id}")
                    print(f"Confidence : {confidence:.2f}")
                    print(f"Time       : {time_string}")
                    print(f"Screenshot : {screenshot_path}")
                    print("================================")


    # ======================================
    # DISPLAY VIDEO
    # ======================================

    cv2.imshow(
        "Wildlife Detection - Final System",
        frame
    )


    # ======================================
    # SAVE OUTPUT VIDEO
    # ======================================

    out.write(frame)


    # ======================================
    # PRESS Q TO STOP
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# 10. CLEANUP
# ==========================================

cap.release()

out.release()

cv2.destroyAllWindows()


# ==========================================
# 11. FINAL MESSAGE
# ==========================================

print()
print("================================")
print("FINAL SYSTEM COMPLETED")
print("================================")

print(f"Output video : {OUTPUT_VIDEO}")
print(f"Alert images : outputs/alerts/")
print(f"Event log    : {CSV_PATH}")

print("================================")