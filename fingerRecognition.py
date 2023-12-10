import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# Initialize the mediapipe hands class
mp_hands = mp.solutions.hands

# Set up the Hands function for images
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2, min_detection_confidence=0.5)


# Initialize the mediapipe drawing class
mp_drawing = mp.solutions.drawing_utils


def rotate_image_90(image):
    # Rotate the image by 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return rotated_image


def detectHandsLandmarks(image, hands, draw=True, display=True):

    # Create a copy of the input image to draw landmarks on
    output_image = image.copy()

    # convert the image from BGR to RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection
    results = hands.process(imageRGB)

    # Check if the landmarks are found and are specified to be drawn
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the copy of the input image
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(
                                          color=(255, 255, 255), thickness=2, circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                     thickness=2, circle_radius=2))

    # check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output")
        plt.axis('off')
        plt.show()

        # Return the output image and results of hands landmarks detection.
    return output_image, results


def countFingers(image, results, draw=True, display=True):

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Create a copy of the input image to write the count of fingers on
    output_image = image.copy()

    # Initialize a dictionary to store the count of fingers of both hands
    count = {'RIGHT': 0, 'LEFT': 0}

    # Store the indexes of the tips landmarks of each finger of a hand in a list
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False, 'RIGHT_PINKY': False,
                        'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False, 'LEFT_RING': False, 'LEFT_PINKY': False}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand. (Right hand or Left Hand)
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the landmarks of the found hand.
            finger_name = tip_index.name.split("_")[0]

            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):

                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                # Increament the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks
        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):

            # Update the status of the thumb in the dictionary to true
            fingers_statuses[hand_label.upper() + "_THUMB"] = True

            # Increament the count of the fingers up of the hand by 1
            count[hand_label.upper()] += 1

    # Check if the total count of the fingers of the both hands are specified to be written on the output image.
    if draw:
        print(hand_label)

        cv2.putText(output_image, str(sum(count.values())), (width//2-150,
                    240), cv2.FONT_HERSHEY_SIMPLEX, 8.9, (20, 255, 155), 10, 10)
        cv2.putText(output_image, hand_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 255, 155), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image:
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')
        plt.show()

    else:

        # Return the output image, the status of each finger and the count of the fingers up of both hands.
        return output_image, fingers_statuses, count


image = cv2.imread("1.jpg")
rotate_image = rotate_image_90(image)
output_image, results = detectHandsLandmarks(rotate_image, hands, display=True)
output_image, fingers_statuses, count = countFingers(
    output_image, results, display=True)
