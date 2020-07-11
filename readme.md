## Dartboard detection

- Environment: Java 1.8
- Dependency: `groupId: org.openpnp`
- Artifact: opencv
- Version: 3.4.2-1

Main classes:
- `Main.java` (where detection takes place)
- `OpenCVHelper.java` (helper functions to aid detection)

This is a project of detecting dartboards in an image.
It utilises a trained Viola-Jones detector(with a high false-positive rate) together with Hough Transform on lines and circles.

Below is a flow diagram illustrating the procedure of the detection:
![alt text][flow]

[flow]: readme_pic/flow_diagram.png "Flow diagram"

Here is what happens at each step:
1. The original image:

![alt text][originalpic]

[originalpic]: readme_pic/dart8.jpg "The original image containing dartboards"

2. Detection result from a viola-jones detector with high false-positive rate:

![alt text][vjpic]

[vjpic]: readme_pic/vj.png "The original image containing dartboards"

Green boxes are labelled by the VJ detector. Red boxes are labelled by hand, showing where the dartboards are. From the above picture we can see that the vj detector is able to detect the dartboards but also labelled lots of other things as dartboards. Hence I improved it with image processing techniques - Hough Transform.

3. To perform Hough Transform, the target image should first have edges extracted. Here I used Canny edge detection:

![alt text][cannypic]

[cannypic]: readme_pic/cannypic.png "Target image after edge extraction"

4. Perform Hough Transform on the extracted image.

Circle Hough Transform:

![alt text][circle]

[circle]: readme_pic/circle.png "Target image after edge extraction"

Line Hough Transform:

![alt text][line]

[line]: readme_pic/line.png "Target image after edge extraction"

5. Combined detection result:

![alt text][combined]

[combined]: readme_pic/combined.png "Target image after edge extraction"

The Dartboard on the right was labelled correctly after the combination of VJ and Hough Transform. However, in this example, the dartboard on the left was not able to be detected, because it is an oval instead of a circle.
