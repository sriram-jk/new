import cv2
import numpy as np
import mediapipe as mp
import math

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        # Left eyes indices 
        LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
        LEFT_IRIS = [474,475, 476, 477] 
        # right eyes indices
        RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
        RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
        RIGHT_IRIS = [469, 470, 471, 472]
        # variables 
        CEF_COUNTER =0
        TOTAL_BLINKS =0
        # constants
        CLOSED_EYES_image =3
        FONTS =cv2.FONT_HERSHEY_COMPLEX
        #mediapipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_faces=1,refine_landmarks=True)
        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        # Euclidean distance 
        def euclideanDistance(point, point1):
            x, y = point
            x1, y1 = point1
            distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
            return distance
        # Blinking Ratio
        def blinkRatio(img, landmarks, right_indices, left_indices):
            # RIGHT_EYE
            # horizontal line 
            rh_right = landmarks[right_indices[0]]
            rh_left = landmarks[right_indices[8]]
            # vertical line 
            rv_top = landmarks[right_indices[12]]
            rv_bottom = landmarks[right_indices[4]]
            # LEFT_EYE 
            # horizontal line 
            lh_right = landmarks[left_indices[0]]
            lh_left = landmarks[left_indices[8]]
            # vertical line 
            lv_top = landmarks[left_indices[12]]
            lv_bottom = landmarks[left_indices[4]]

            rhDistance = euclideanDistance(rh_right, rh_left)
            rvDistance = euclideanDistance(rv_top, rv_bottom)
            lvDistance = euclideanDistance(lv_top, lv_bottom)
            lhDistance = euclideanDistance(lh_right, lh_left)

            reRatio = rhDistance/rvDistance
            leRatio = lhDistance/lvDistance

            ratio = (reRatio+leRatio)/2
            return ratio 
        # Eyes Extractor function
        def eyesExtractor(img, right_eye_coords, left_eye_coords):
            # converting color image to  scale image 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # getting the dimension of image 
            dim = gray.shape
            # creating mask from gray scale dim
            mask = np.zeros(dim, dtype=np.uint8)
            # drawing Eyes Shape on mask with white color 
            cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
            cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
            # draw eyes image on mask, where white shape is 
            eyes = cv2.bitwise_and(gray, gray, mask=mask)
            eyes[mask==0]=155
            
            # getting minium and maximum x and y  for right and left eyes 
            
            # For Right Eye 
            r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
            r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
            r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
            r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

            # For LEFT Eye
            l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
            l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
            l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
            l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

            # croping the eyes from mask 
            cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
            cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

            # returning the cropped eyes 
            return cropped_right, cropped_left

        # Eyes Postion Estimator 
        def positionEstimator(cropped_eye):
            # getting height and width of eye 
            h, w =cropped_eye.shape
            # remove the noise from images
            gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
            median_blur = cv2.medianBlur(gaussain_blur, 3)
            # applying thrsholding to convert binary_image
            ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)
            # create fixd part for eye with 
            piece = int(w/3) 
            # slicing the eyes into three parts 
            right_piece = threshed_eye[0:h, 0:piece]
            center_piece = threshed_eye[0:h, piece: piece+piece]
            left_piece = threshed_eye[0:h, piece +piece:w]
            # calling pixel counter function
            eye_position= pixelCounter(right_piece, center_piece, left_piece)
            return eye_position
        # creating pixel counter function 
        def pixelCounter(first_piece, second_piece, third_piece):
            # counting black pixel in each part 
            right_part = np.sum(first_piece==0)
            center_part = np.sum(second_piece==0)
            left_part = np.sum(third_piece==0)
            # creating list of these values
            eye_parts = [right_part, center_part, left_part]
            # getting the index of max values in the list 
            max_index = eye_parts.index(max(eye_parts))
            pos_eye ='' 
            if max_index==0:
                pos_eye="RIGHT"
            elif max_index==1:
                pos_eye = 'CENTER'
            elif max_index ==2:
                pos_eye = 'LEFT'
            else:
                pos_eye="Closed"
            return pos_eye

        success, image = self.video.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False
        
        # Get the result
        results = face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            mesh_coord = [(int(point.x * img_w), int(point.y * img_h)) for point in results.multi_face_landmarks[0].landmark]
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        # Get the 2D Coordinates
                        face_2d.append([x, y])
                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            
                ratio = blinkRatio(image, mesh_coord, RIGHT_EYE, LEFT_EYE)
                cv2.putText(image, f'ratio: {ratio}', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

                if ratio >5.5:
                    CEF_COUNTER +=1
                    cv2.putText(image, 'Blink', (20,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
                    
                else:
                    if CEF_COUNTER>CLOSED_EYES_image:
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                cv2.putText(image, f'Total_Blinks: {TOTAL_BLINKS}', (20, 60),  cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
                cv2.polylines(image,  [np.array([mesh_coord[p] for p in LEFT_IRIS ], dtype=np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.polylines(image,  [np.array([mesh_coord[p] for p in RIGHT_IRIS ], dtype=np.int32)], True,(0, 0, 255), 1, cv2.LINE_AA)

                # Blink Detector Counter Completed
                right_coords = [mesh_coord[p] for p in RIGHT_EYE]
                left_coords = [mesh_coord[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(image, right_coords, left_coords)
                eye_position = positionEstimator(crop_right)
                cv2.putText(image, f'Right_eye_pos: {eye_position}', (20, 80),  cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
                eye_position_left = positionEstimator(crop_left)
                cv2.putText(image, f'Left_eye_pos: {eye_position}', (20, 100),  cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
                
                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
            
                # Add the text on the image
                cv2.putText(image, f'Facing_side: {text}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)
                cv2.putText(image, "Pitch: " + str(np.round(x,2)), (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, "Yaw: " + str(np.round(y,2)), (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, "Roll: " + str(np.round(z,2)), (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        # Calculating the fps
        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()