import dlib, cv2, sys
import numpy as np


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/einst/Toy/dlib_face_recognition/shape_predictor_68_face_landmarks.dat"
)

cap = cv2.VideoCapture("C:/Users/einst/Toy/dlib_face_recognition/production ID_4770560.mp4")
overay_face = cv2.imread(
    "C:/Users/einst/Toy/dlib_face_recognition/lengokoo.png", cv2.IMREAD_UNCHANGED
)  # UNCHANGED 로 투명(알파재널) 까지 표현sssssssss


def overlay_transparent(
    background_img, img_to_overlay_t, x, y, overlay_size=None
):  # 오버레이 하는 소스 - 이미지를 센터 중심으로 오버레이 사이즈만큼 리사이즈 해서 원본에 넣음
    try:  # 오버레이 사이즈가 넘어가면 에러뜬는걸 예외처리해줌

        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)]
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)] = cv2.add(
            img1_bg, img2_fg
        )

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img

    except Exception:
        return background_img


while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
    ori = img.copy()
    result = ori

    faces = detector(img)  # 일단 단일로 얼굴 검출
    if len(faces) == 0:  # 여기도 얼굴 인식 못할 경우 예외처리(얼굴이 없는 프레임이 스킵됨..프레임 기준으로 인식해서)
        continue

    # result = np.array(faces)
    for n in faces:
        face = n

        dlib_shape = predictor(img, face)  # 얼굴 특징점 주출(이 잡은 점들의 위치를 활용해 다양한 것 계산 가능)
        shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])  # 이를 위해 연산을(넘파이로)

        top_left = np.min(shape_2d, axis=0)  # 열의 최소값을 구함 // 좌상단과 우하단의 위치점을 잡아 얼굴 사이즈로 구함
        bottom_right = np.max(shape_2d, axis=0)

        face_size = int(max(bottom_right - top_left) * 1.8)

        center_x, center_y = np.mean(
            shape_2d, axis=0
        ).astype(  # 얼굴의 중심값을 구함 , 소수가 될 수 있으므로 astype 으로 int형의 캐스트함
            np.int
        )
        # result = np.array([])
        # result = np.append(
        #    result,
        #    overlay_transparent(
        #        ori, overay_face, center_x, center_y, overlay_size=(face_size, face_size) # 개 뻘짓
        #    ),
        # )

        result = overlay_transparent(
            result, overay_face, center_x, center_y, overlay_size=(face_size, face_size)
        )

        print(result)
        for i in shape_2d:
            img = cv2.rectangle(
                img,
                pt1=(face.left(), face.top()),
                pt2=(face.right(), face.bottom()),
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        for n in shape_2d:
            cv2.circle(
                img,
                center=tuple(n),
                radius=1,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )  # shape_2d안에는 어차피 좌표형식의 배열이 있으므로 tuple로 바로 중심지정함.

        cv2.circle(
            img,
            center=tuple(top_left),
            radius=1,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            img,
            center=tuple(bottom_right),
            radius=1,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            img,
            center=(center_x, center_y),
            radius=1,
            color=(255, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    cv2.imshow("img", img)  # 인식 범위 확인용도
    cv2.imshow("result", result)
    cv2.waitKey(1)
