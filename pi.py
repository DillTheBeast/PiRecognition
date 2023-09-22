import RPi.GPIO as GPIO
import time

# Set up GPIO mode
GPIO.setmode(GPIO.BCM)

# Pin setup
BUTTON_PIN = 17
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def button_callback(channel):
    print("Hi")

# Set up an interrupt-driven event on button press
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)

try:
    while True:
        # Just sleep and wait for button presses
        time.sleep(0.1)

except KeyboardInterrupt:
    # If someone presses CTRL+C then cleanup and exit
    GPIO.cleanup()

