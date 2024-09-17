import cv2

video_path = 'traffic.mp4'
capture = cv2.VideoCapture(video_path)

line1 = [(40, 200), (500, 200)]
line2 = [(40, 500), (500, 500)]

count = 0
object_ids = set()

object_detector = cv2.createBackgroundSubtractorMOG2()

def _intersect(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    
    return cross_product < 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = ((x + x + w) // 2, (y + y + h) // 2)

            object_id = hash(centroid)

            if object_id not in object_ids:
                if _intersect(centroid, line1[0], line1[1]):
                    count += 1
                    object_ids.add(object_id)

                if _intersect(centroid, line2[0], line2[1]):
                    count -= 1
                    object_ids.discard(object_id)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.line(frame, line1[0], line1[1], (0, 0, 255), 2)
    cv2.line(frame, line2[0], line2[1], (0, 0, 255), 2)

    cv2.putText(frame, f'Count: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Vehicle Density', frame)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

print(count)
