import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.SerialModule import SerialObject
import numpy as np
import pyfirmata

comport = 'COM8'
board = pyfirmata.Arduino(comport)

led_1 = board.get_pin('d:8:o')
led_2 = board.get_pin('d:9:o')
led_3 = board.get_pin('d:10:o')
led_4 = board.get_pin('d:11:o')
led_5 = board.get_pin('d:12:o')

# Initialize the buzzer pin
buzzer_pin = board.get_pin('d:7:o')

def led_and_buzzer(fingerUp, is_red_detected):
    if fingerUp == [0, 0, 0, 0, 0]:
        led_1.write(0)
        led_2.write(0)
        led_3.write(0)
        led_4.write(0)
        led_5.write(0)
        buzzer_pin.write(0)

    elif fingerUp == [0, 1, 0, 0, 0]:
        led_1.write(1)
        led_2.write(0)
        led_3.write(0)
        led_4.write(0)
        led_5.write(0)
        buzzer_pin.write(1) if is_red_detected else buzzer_pin.write(0)
    elif fingerUp == [0, 1, 1, 0, 0]:
        led_1.write(1)
        led_2.write(1)
        led_3.write(0)
        led_4.write(0)
        led_5.write(0)
        buzzer_pin.write(1) if is_red_detected else buzzer_pin.write(0)
    elif fingerUp == [0, 1, 1, 1, 0]:
        led_1.write(1)
        led_2.write(1)
        led_3.write(1)
        led_4.write(0)
        led_5.write(0)
        buzzer_pin.write(1) if is_red_detected else buzzer_pin.write(0)
    elif fingerUp == [0, 1, 1, 1, 1]:
        led_1.write(1)
        led_2.write(1)
        led_3.write(1)
        led_4.write(1)
        led_5.write(0)
        buzzer_pin.write(1) if is_red_detected else buzzer_pin.write(0)
    elif fingerUp == [1, 1, 1, 1, 1]:
        led_1.write(1)
        led_2.write(1)
        led_3.write(1)
        led_4.write(1)
        led_5.write(1)
        buzzer_pin.write(1) if is_red_detected else buzzer_pin.write(0)

# Initialize video capture, hand detector, and Arduino communication
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)
arduino = SerialObject('/dev/ttyUSB0')

# Define hand gesture and color recognition parameters
my_array = [[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
my_numbers = [0, 1, 2, 3, 4, 5, 6]
color_ranges = {
    "red": ([170, 180], [0, 100], [0, 100]),
    "green": ([36, 80], [0, 100], [0, 100]),
    "blue": ([110, 130], [0, 100], [0, 100]),
    # Add more color ranges as needed
}

# Define min_contour_area outside the loop
min_contour_area = 1000  # Adjust this value based on your needs

try:
    while True:
        # Read video frame and find hands
        success, image = cap.read()
        hands, bboxInfo = detector.findHands(image)

        # Hand gesture recognition
        # Hand gesture recognition
        is_red_detected = False  # Assume red color is not detected
        fingers_up = [0, 0, 0, 0, 0]

        if len(hands) == 1:
             fingers_up = detector.fingersUp(hands[0])
             num_fingers_up = sum(fingers_up)
             if num_fingers_up > 0:
              arduino.sendData([num_fingers_up])
             cv2.putText(image, f"{num_fingers_up} Finger(s) Up", (70, 110), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

             for i in range(0, 6):
               if fingers_up == my_array[i]:
                gesture_number = my_numbers[i]
                cv2.putText(image, str(gesture_number), (70, 110), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
                arduino.sendData([gesture_number])
        elif fingers_up == [0, 0, 0, 0, 0]:
                arduino.sendData([0])


        # Color recognition for red only
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_lower_bound = np.array([0, 100, 100])
        red_upper_bound = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv_image, red_lower_bound, red_upper_bound)
        cnts_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts_red:
            largest_contour_red = max(cnts_red, key=cv2.contourArea)
            if cv2.contourArea(largest_contour_red) > min_contour_area:
                x, y, w, h = cv2.boundingRect(largest_contour_red)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                signal_sent = fingers_up  # Assuming the signal sent is the fingers_up array
                color_detected = "Red"
                is_red_detected = True
                print(f"Signal sent to Arduino: {signal_sent}, Color detected: {color_detected}")

        # Call the led_and_buzzer function with the updated is_red_detected value
        led_and_buzzer(fingers_up, is_red_detected)

        cv2.imshow('image', image)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Keyboard interrupt. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
