import wordninja
import wordsegment

wordsegment.load()

sample = "We are building IoT devices with C++andFreeRTOS, using AWSIoTCore for cloudconnectivity and InfluxDBfor time-series data storage."

print("wordninja:", wordninja.split(sample))
print("wordsegment:", wordsegment.segment(sample))
