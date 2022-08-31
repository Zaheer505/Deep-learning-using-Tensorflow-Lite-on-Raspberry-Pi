import RPi.GPIO as GPIO
import time

led_1 = 21
led_2 = 20

GPIO.setmode(GPIO.BCM)
GPIO.setup(led_1,GPIO.OUT)
GPIO.setup(led_2,GPIO.OUT)

while(1):
    GPIO.output(led_1,GPIO.HIGH)
    GPIO.output(led_2,GPIO.HIGH)

    time.sleep(1)

    GPIO.output(led_1,GPIO.LOW)
    GPIO.output(led_2,GPIO.LOW)

    time.sleep(1)