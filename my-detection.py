import jetson.inference
import jetson.utils

# My_detect_ID
Need_detect_ID = [
    "1",
    "2",
    "3",
    "4",
    "6",
    "8",
]  # person bicycle car motorcyle bus truck

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")
display = jetson.utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(image=img, overlay="labels,conf")

    for detection in detections:
        
        My_class_ID = str(detection).split()[4]
        if My_class_ID in Need_detect_ID:

            if My_class_ID == "1":
                print("person")

            if My_class_ID == "2":
                print("bicycle")

            if My_class_ID == "3":
                print("car")

            if My_class_ID == "4":
                print("motorcyle")

            if My_class_ID == "6":
                print("bus")

            if My_class_ID == "8":
                print("truck")
        

    display.Render(img)
    display.SetStatus(
        "Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())
    )
