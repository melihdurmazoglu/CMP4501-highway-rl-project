import cv2
import numpy as np
import os


def merge_videos() -> None:
    """3 videoyu yan yana birleştirip evolution.mp4 oluşturur."""
    paths = [
        ("videos/untrained.mp4", "Untrained"),
        ("videos/half_trained.mp4", "Half Trained"),
        ("videos/full_trained.mp4", "Full Trained"),
    ]

    caps = [cv2.VideoCapture(p) for p, _ in paths]
    labels = [label for _, label in paths]

    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    min_frames = min(frame_counts)

    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "assets/evolution.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width * 3, height),
    )

    for _ in range(min_frames):
        frames = []
        for cap, label in zip(caps, labels):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frames.append(frame)
        combined = np.concatenate(frames, axis=1)
        out.write(combined)

    for cap in caps:
        cap.release()
    out.release()
    print("Evolution video kaydedildi: assets/evolution.mp4")


if __name__ == "__main__":
    merge_videos()