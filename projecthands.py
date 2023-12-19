import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.1
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Bỏ qua khung máy ảnh trống.")
            # Nếu tải video, thay vì break thì contiune.
            continue

        # Để cải thiện hiệu suất, tùy ý đánh dấu hình ảnh là không thể ghi vào
        # chuyển qua tham chiếu.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Vẽ chú thích bằng tay trên ảnh.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Lấy chỉ số tay để kiểm tra (trái hoặc phải)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Đặt biến để giữ vị trí mốc (x và y)
                hand_landmarks_list = []

                # Điền vào danh sách vị trí x và y của từng mốc
                for landmark in hand_landmarks.landmark:
                    hand_landmarks_list.append([landmark.x, landmark.y])

                # Tính góc giữa hai đường thẳng a và b
                a = hand_landmarks_list[4]  # Mẹo ngón tay cái
                b = hand_landmarks_list[3]  # IP ngón tay cái
                c = hand_landmarks_list[2]  # MCP ngón tay cái

                angle = math.degrees(
                    math.atan2(c[1] - b[1], c[0] - b[0])
                    - math.atan2(a[1] - b[1], a[0] - b[0])
                )

                if handLabel == "Left":
                    if (
                        hand_landmarks_list[4][0] > hand_landmarks_list[13][0]
                        and angle <= 180.0
                    ):
                        finger_count += 1
                    elif (
                        hand_landmarks_list[4][0] < hand_landmarks_list[13][0]
                        and angle > 180.0
                        ):
                        finger_count += 1
                elif handLabel == "Right":
                    if (
                        hand_landmarks_list[4][0] < hand_landmarks_list[13][0]
                        and angle > 180.0
                    ):
                        finger_count += 1
                    elif (
                        hand_landmarks_list[4][0] > hand_landmarks_list[13][0]
                        and angle <= 180.0
                    ):
                        finger_count += 1

                # Các ngón khác: Vị trí TIP y phải thấp hơn vị trí PIP y,
                # vì nguồn gốc hình ảnh nằm ở góc trên bên trái.
                if hand_landmarks_list[8][1] < hand_landmarks_list[7][1]:  # Ngón trỏ
                    finger_count += 1
                if hand_landmarks_list[12][1] < hand_landmarks_list[11][1]:  # Ngón giữa
                    finger_count += 1
                if (
                    hand_landmarks_list[16][1] < hand_landmarks_list[15][1]
                ):  # Ngón áp út 
                    finger_count += 1
                if hand_landmarks_list[20][1] < hand_landmarks_list[19][1]:  # Ngón út
                    finger_count += 1

                # Vẽ mốc bằng tay
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

            # Hiển thị số ngón tay
            cv2.putText(
                image,
                f"So ngon tay: {finger_count}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Dem ngon tay", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()