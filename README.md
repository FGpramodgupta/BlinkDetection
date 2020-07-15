<p>Our eyes are one of the primary sensing tools we use to learn and react to the environment. Eye movements can tell us a lot about human behavior. The field of measuring human eye movements is called eye-tracking. It has various applications in gaming, medical diagnostics, market research, and psychology.
In this article, we explore blinking, which is one of the easiest eye movements to detect and has numerous applications in the fields of medical diagnostics and human computer interaction. Blinking can be involuntary or voluntary.</p>

<p>Involuntary blinking is used as criteria to diagnose medical conditions. For example, if a person blinks excessively it may indicate the onset of Tourette syndrome, strokes, or disorders of the nervous system. A reduced rate of blinking is associated with Parkinson’s disease.
Voluntary blinking can be used as a means of communication. The interface to a computing device has traditionally been a keyboard or a mouse, and more recently, a touchscreen. If blinking can be detected, it can serve as an additional interface, with appropriate responses for a specified action.
There have been many approaches to detecting blinks. In this article, we show you how to detect blinks using a commodity webcam and python and a simplified algorithm. Ready to get started? Let’s look at the prerequisites.</p>
Prerequisites

<b>Install python</b>

<p>Python version 3.5 is recommended for compatibility with dlib and OpenCV.</p>

<b>Install OpenCV</b>

<p>OpenCV is a library of programming functions mainly aimed at real-time computer vision. We recommend version 3.3 for better compatibility with dlib. Here, OpenCV is used to capture frames by accessing the webcam in real time.</p>

<b>Install dlib</b>

<p>dlib is an open-source library used for face detection. We recommend installing dlib version 19.4 for better compatibility with OpenCV. Given a face, dlib can extract features from the face like eyes, nose, lips, and jaw using facial landmarks. The required facial landmark file can be downloaded from the following link. This file should be placed in the folder which has the rest of the code.</p>
