import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
import plotly.graph_objects as go
import sys
sys.path.append('D:\GIT\AI-Fitness-Trainer\models\PoseDetector.py') 
from PoseDetector import PoseDetector

# Streamlit UI
st.markdown("<h2 style='text-align:center; color:white; background-color:#025246; padding:10px;'>Train Here</h2>", unsafe_allow_html=True)

# Sidebar selection
app_mode = st.sidebar.selectbox("Choose the exercise", ["About", "Left Dumbbell", "Right Dumbbell", "Squats", "Pushups", "Shoulder Press"])

if app_mode == "About":
    st.markdown("## Welcome to the Training Arena")
    st.markdown("Choose the workout from the sidebar")
    st.write("""
    - Allow webcam access.
    - Avoid crowded areas.
    - Ensure proper lighting.
    - Keep the camera focused on you.
    """)

else:
    st.markdown(f"## {app_mode} Exercise")

    weight1 = st.slider('Enter your weight (kg)', 20, 130, 40)
    goal_calorie1 = st.slider('Set a goal calorie to burn', 1, 200, 15)

    st.write("Click Start to begin the exercise.")

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils

    # Angle calculation function
    def calculate_angle(a, b, c):
        a = np.array(a)  # First joint
        b = np.array(b)  # Middle joint
        c = np.array(c)  # Last joint

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return int(angle)

    # Button handlers
    if 'type' not in st.session_state:
        st.session_state.type = None

    def handle_click_start():
        st.session_state.type = "Start"
        st.session_state.counter1 = 0

    def handle_click_stop():
        st.session_state.type = "Stop"

    st.button('Start', on_click=handle_click_start)
    st.button('Stop', on_click=handle_click_stop)

    counter, direction = 0, 0
    frame_placeholder = st.empty()

    if st.session_state['type'] == 'Start':
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Detect joints based on the selected exercise
                    if app_mode == "Left Dumbbell":
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
                    elif app_mode == "Right Dumbbell":
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST
                    elif app_mode == "Squats":
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE
                    elif app_mode == "Pushups":
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST
                    elif app_mode == "Shoulder Press":
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP

                    # Get joint positions
                    p1 = [landmarks[joint1.value].x * frame.shape[1], landmarks[joint1.value].y * frame.shape[0]]
                    p2 = [landmarks[joint2.value].x * frame.shape[1], landmarks[joint2.value].y * frame.shape[0]]
                    p3 = [landmarks[joint3.value].x * frame.shape[1], landmarks[joint3.value].y * frame.shape[0]]

                    # Calculate angle
                    angle = calculate_angle(p1, p2, p3)

                    # Draw line connecting joints
                    cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 4)
                    cv2.line(frame, tuple(map(int, p2)), tuple(map(int, p3)), (0, 255, 0), 4)

                    # Draw circles on joints
                    cv2.circle(frame, tuple(map(int, p1)), 8, (255, 0, 255), -1)
                    cv2.circle(frame, tuple(map(int, p2)), 8, (255, 0, 255), -1)
                    cv2.circle(frame, tuple(map(int, p3)), 8, (255, 0, 255), -1)

                    # Rep counting logic
                    if angle >= 90 and direction == 0:
                        counter += 0.5
                        st.session_state.counter1 = counter
                        direction = 1
                    elif angle <= 70 and direction == 1:
                        counter += 0.5
                        st.session_state.counter1 = counter
                        direction = 0

                    # Display rep count
                    cv2.rectangle(frame, (10, 10), (150, 80), (255, 0, 0), -1)
                    cv2.putText(frame, str(int(counter)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                    # Draw progress bar
                    progress = np.interp(angle, [30, 130], [480, 200])
                    cv2.rectangle(frame, (580, 200), (630, 480), (0, 0, 255), 5)
                    cv2.rectangle(frame, (580, int(progress)), (630, 480), (0, 0, 255), -1)

                # Convert frame to RGB for Streamlit display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, "RGB")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    elif st.session_state['type'] == 'Stop':
        st.write(f"## Analytics\nYou did {st.session_state.counter1} reps.")

        # Calories calculation
        calories_burned = 0.25 * st.session_state.counter1
        st.write(f"You burned {calories_burned:.2f} kcal")

        if calories_burned < goal_calorie1:
            st.write("You have not achieved your goal. Try again!")
        else:
            st.write("You have achieved your goal. Congratulations! 🎉")

        # Plotting calories burned
        fig = go.Figure(data=[go.Bar(x=[app_mode], y=[calories_burned], name='Calories Burned')])
        fig.add_trace(go.Bar(x=[app_mode], y=[goal_calorie1], name='Goal Calorie'))
        fig.update_layout(title='Calories Burned', xaxis_title='Exercise', yaxis_title='Calories Burned')
        st.plotly_chart(fig)

                   
            
        

            
    

    elif app_mode == "Right Dumbbell":
        st.markdown("## Right Dumbbell Curl")

        weight = st.slider('Enter your weight (kg)', 20, 130, 40, key="weight_right")
        goal_calorie = st.slider('Set a goal calorie to burn', 1, 200, 15, key="goal_right")

        st.write("Click Start to begin the exercise.")

        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        # Angle calculation function
        def calculate_angle(a, b, c):
            a = np.array(a)  # First joint
            b = np.array(b)  # Middle joint
            c = np.array(c)  # Last joint

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return int(angle)

        # Button handlers
        if 'type' not in st.session_state:
            st.session_state.type = None

        def handle_click_start():
            st.session_state.type = "Start_Right"
            st.session_state.counter_right = 0

        def handle_click_stop():
            st.session_state.type = "Stop_Right"

        st.button('Start', on_click=handle_click_start, key="start_right")
        st.button('Stop', on_click=handle_click_stop, key="stop_right")

        counter, direction = 0, 0
        frame_placeholder = st.empty()

        if st.session_state['type'] == 'Start_Right':
            cap = cv2.VideoCapture(0)

            with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        # Right arm joints
                        joint1, joint2, joint3 = mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST

                        # Get joint positions
                        p1 = [landmarks[joint1.value].x * frame.shape[1], landmarks[joint1.value].y * frame.shape[0]]
                        p2 = [landmarks[joint2.value].x * frame.shape[1], landmarks[joint2.value].y * frame.shape[0]]
                        p3 = [landmarks[joint3.value].x * frame.shape[1], landmarks[joint3.value].y * frame.shape[0]]

                        # Calculate angle
                        angle = calculate_angle(p1, p2, p3)

                        # Draw line connecting joints
                        cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 0), 4)
                        cv2.line(frame, tuple(map(int, p2)), tuple(map(int, p3)), (0, 255, 0), 4)

                        # Draw circles on joints
                        cv2.circle(frame, tuple(map(int, p1)), 8, (255, 0, 255), -1)
                        cv2.circle(frame, tuple(map(int, p2)), 8, (255, 0, 255), -1)
                        cv2.circle(frame, tuple(map(int, p3)), 8, (255, 0, 255), -1)

                        # Rep counting logic
                        if angle >= 90 and direction == 0:
                            counter += 0.5
                            st.session_state.counter_right = counter
                            direction = 1
                        elif angle <= 70 and direction == 1:
                            counter += 0.5
                            st.session_state.counter_right = counter
                            direction = 0

                        # Display rep count
                        cv2.rectangle(frame, (10, 10), (150, 80), (255, 0, 0), -1)
                        cv2.putText(frame, str(int(counter)), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                        # Draw progress bar
                        progress = np.interp(angle, [30, 130], [480, 200])
                        cv2.rectangle(frame, (580, 200), (630, 480), (0, 0, 255), 5)
                        cv2.rectangle(frame, (580, int(progress)), (630, 480), (0, 0, 255), -1)

                    # Convert frame to RGB for Streamlit display
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame, "RGB")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        elif st.session_state['type'] == 'Stop_Right':
            st.write(f"## Analytics\nYou did {st.session_state.counter_right} reps.")

            # Calories calculation
            calories_burned = 0.25 * st.session_state.counter_right
            st.write(f"You burned {calories_burned:.2f} kcal")

            if calories_burned < goal_calorie:
                st.write("You have not achieved your goal. Try again!")
            else:
                st.write("You have achieved your goal. Congratulations! 🎉")

            # Plotting calories burned
            fig = go.Figure(data=[go.Bar(x=["Right Dumbbell Curl"], y=[calories_burned], name='Calories Burned')])
            fig.add_trace(go.Bar(x=["Right Dumbbell Curl"], y=[goal_calorie], name='Goal Calorie'))
            fig.update_layout(title='Calories Burned', xaxis_title='Exercise', yaxis_title='Calories Burned')
            st.plotly_chart(fig)
            

        elif app_mode == "Squats":
            st.markdown("## Squats")
            weight3 = st.slider('What is your weight?', 20, 130, 40)
            st.write("I'm ", weight3, 'kgs')

            st.write("-------------")

            goal_calorie3 = st.slider('Set a goal calorie to burn', 1, 200, 15)
            st.write("I want to burn", goal_calorie3, 'kcal')
            
            st.write("-------------")


            st.write(" Click on the Start button to start the live video feed.")
            st.write("##")


            # Creating Angle finder class
            class angleFinder:
                def __init__(self,lmlist,p1,p2,p3,p4,p5,p6,drawPoints):
                    self.lmlist = lmlist
                    self.p1 = p1
                    self.p2 = p2
                    self.p3 = p3
                    self.p4 = p4
                    self.p5 = p5
                    self.p6 = p6
                    self.drawPoints = drawPoints
                #    finding angles

                def angle(self):
                    if len(self.lmlist) != 0:
                        point1 = self.lmlist[self.p1]
                        point2 = self.lmlist[self.p2]
                        point3 = self.lmlist[self.p3]
                        point4 = self.lmlist[self.p4]
                        point5 = self.lmlist[self.p5]
                        point6 = self.lmlist[self.p6]

                        x1,y1 = point1[1:-1]
                        x2, y2 = point2[1:-1]
                        x3, y3 = point3[1:-1]
                        x4, y4 = point4[1:-1]
                        x5, y5 = point5[1:-1]
                        x6, y6 = point6[1:-1]

                        # calculating angle for left leg
                        leftLegAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                                    math.atan2(y1 - y2, x1 - x2))

                        leftLegAngle = int(np.interp(leftLegAngle, [42,143], [100, 0]))
                        

                        # drawing circles and lines on selected points
                        if self.drawPoints == True:
                            cv2.circle(img, (x1, y1), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x1, y1), 15, (0, 255, 0), 6)
                            cv2.circle(img, (x2, y2), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x2, y2), 15, (0, 255, 0), 6)
                            cv2.circle(img, (x3, y3), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x3, y3), 15, (0, 255, 0), 6)
                            cv2.circle(img, (x4, y4), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x4, y4), 15, (0, 255, 0), 6)
                            cv2.circle(img, (x5, y5), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x5, y5), 15, (0, 255, 0), 6)
                            cv2.circle(img, (x6, y6), 10, (0, 255, 255), 5)
                            cv2.circle(img, (x6, y6), 15, (0, 255, 0), 6)

                            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),4)
                            cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                            cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                            cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                            cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 4)

                        return leftLegAngle
                    
            if 'type' not in st.session_state:
                st.session_state.type = None


            def handle_click_start():
                st.session_state.type = "Start3"

            def handle_click_stop():
                st.write(st.session_state.counter3)
                st.session_state.type = "Stop3"
            
            start_button = st.button('Start', on_click=handle_click_start)
            stop_button = st.button('Stop',  on_click=handle_click_stop)

            # defining some variables
            counter = 0
            direction = 0

            frame_placeholder = st.empty()

            detector = PoseDetector(detectionCon=0.7,trackCon=0.7)


            if st.session_state['type']=='Start3':
                cap = cv2.VideoCapture(0)
                while cap.isOpened():
                    ret, img = cap.read()
                    img = cv2.resize(img,(640,480))

                    detector.findPose(img,draw=0)
                    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=0,draw=False)

                    angle1 = angleFinder(lmList,24,26,28,23,25,27,drawPoints=True)
                    left = angle1.angle()
                    
                    if left==None:
                        left=0

                    # Counting number of shoulder ups
                    if left >= 90:
                        if direction == 0:
                            counter += 0.5
                            st.session_state.counter3 = counter
                            direction = 1
                    if left <= 70:
                        if direction == 1:
                            counter += 0.5
                            st.session_state.counter3 = counter
                            direction = 0



                    #putting scores on the screen
                    cv2.rectangle(img,(0,0),(120,120),(255,0,0),-1)
                    cv2.putText(img,str(int(counter)),(1,70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1.6,(0,0,255),6)

                    # Converting values for rectangles
                    leftval = np.interp(left,[0,100],[480,280])


                    # Drawing left rectangle and putting text
                    cv2.rectangle(img, (582, 280), (632, 480), (0, 0, 255), 5)
                    cv2.rectangle(img, (582, int(leftval)), (632, 480), (0, 0, 255), -1)


                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(img, "RGB")
                    
                    cv2.waitKey(1)
                    
            elif st.session_state['type']=='Stop3': 
                st.write("The video capture has ended")

                st.write("---------")
                st.write("## Analytics") 
                st.write("You did ",st.session_state.counter3," reps")   
                
                # calories3=6.0*weight3/st.session_state.counter3
                calories3=0.3*st.session_state.counter3
                if calories3<goal_calorie3:
                    st.write("You have burned ",calories3,"kcal of calories")
                    st.write("You have not achieved your goal. Try again")

                else:
                    st.write("You have burned ",calories3,"kcal of calories")
                    st.write("You have achieved your goal. Congratulations")
                
                fig = go.Figure(data=[go.Bar(x=['Bicep Curls'], y=[calories3], name='Calories Burned')])

                fig.add_trace(go.Bar(x=['Bicep Curls'], y=[goal_calorie3], name='Goal Calorie'))

                # Set chart layout
                fig.update_layout(
                    title='Calories Burned for Bicep Curls',
                    xaxis_title='Exercise',
                    yaxis_title='Calories Burned'
                )

                # Display the chart using Streamlit
                st.plotly_chart(fig)

            

    elif app_mode == "Pushups":
        st.markdown("## Pushups")
        weight4 = st.slider('What is your weight?', 20, 130, 40)
        st.write("I'm ", weight4, 'kgs')

        st.write("-------------")

        goal_calorie4 = st.slider('Set a goal calorie to burn', 1, 200, 15)
        st.write("I want to burn", goal_calorie4, 'kcal')
        
        st.write("-------------")


        st.write(" Click on the Start button to start the live video feed.")
        st.write("##")


        #cap = cv2.VideoCapture('vid1.mp4')
        

        def angles(lmlist,p1,p2,p3,p4,p5,p6,drawpoints):
                global counter
                global direction

                if len(lmlist)!= 0:
                    point1 = lmlist[p1]
                    point2 = lmlist[p2]
                    point3 = lmlist[p3]
                    point4 = lmlist[p4]
                    point5 = lmlist[p5]
                    point6 = lmlist[p6]

                    x1,y1 = point1[1:-1]
                    x2, y2 = point2[1:-1]
                    x3, y3 = point3[1:-1]
                    x4, y4 = point4[1:-1]
                    x5, y5 = point5[1:-1]
                    x6, y6 = point6[1:-1]

                    if drawpoints == True:
                        cv2.circle(img,(x1,y1),10,(255,0,255),5)
                        cv2.circle(img, (x1, y1), 15, (0,255, 0),5)
                        cv2.circle(img, (x2, y2), 10, (255, 0, 255), 5)
                        cv2.circle(img, (x2, y2), 15, (0, 255, 0), 5)
                        cv2.circle(img, (x3, y3), 10, (255, 0, 255), 5)
                        cv2.circle(img, (x3, y3), 15, (0, 255, 0), 5)
                        cv2.circle(img, (x4, y4), 10, (255, 0, 255), 5)
                        cv2.circle(img, (x4, y4), 15, (0, 255, 0), 5)
                        cv2.circle(img, (x5, y5), 10, (255, 0, 255), 5)
                        cv2.circle(img, (x5, y5), 15, (0, 255, 0), 5)
                        cv2.circle(img, (x6, y6), 10, (255, 0, 255), 5)
                        cv2.circle(img, (x6, y6), 15, (0, 255, 0), 5)

                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),6)
                        cv2.line(img, (x2,y2), (x3, y3), (0, 0, 255), 6)
                        cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 6)
                        cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 6)
                        cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 6)

                    lefthandangle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                                math.atan2(y1 - y2, x1 - x2))

                    righthandangle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                                math.atan2(y4 - y5, x4 - x5))

                    # print(lefthandangle,righthandangle)

                    leftHandAngle = int(np.interp(lefthandangle, [-30, 180], [100, 0]))
                    rightHandAngle = int(np.interp(righthandangle, [34, 173], [100, 0]))

                    left, right = leftHandAngle, rightHandAngle

                    if left >= 60 and right >= 60:
                        if direction == 0:
                            counter += 0.5
                            st.session_state.counter4 = counter
                            direction = 1
                    if left <= 60 and right <= 60:
                        if direction == 1:
                            counter += 0.5
                            st.session_state.counter4 = counter
                            direction = 0

                    cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
                    cv2.putText(img, str(int(counter)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

                    leftval  = np.interp(right,[0,100],[400,200])
                    rightval = np.interp(right, [0, 100], [400, 200])

                    cv2.putText(img,'R', (24, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
                    cv2.rectangle(img,(8,200),(50,400),(0,255,0),5)
                    cv2.rectangle(img, (8, int(rightval)), (50, 400), (255,0, 0), -1)

                    cv2.putText(img, 'L', (962, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 7)
                    cv2.rectangle(img, (952, 200), (995, 400), (0, 255, 0), 5)
                    cv2.rectangle(img, (952, int(leftval)), (995, 400), (255, 0, 0), -1)


                    if left > 70:
                        cv2.rectangle(img, (952, int(leftval)), (995, 400), (0, 0, 255), -1)

                    if right > 70:
                        cv2.rectangle(img, (8, int(leftval)), (50, 400), (0, 0, 255), -1)


        if 'type' not in st.session_state:
            st.session_state.type = None


        def handle_click_start():
            st.session_state.type = "Start4"

        def handle_click_stop():
            st.write(st.session_state.counter4)
            st.session_state.type = "Stop4"
        
        start_button = st.button('Start', on_click=handle_click_start)
        stop_button = st.button('Stop',  on_click=handle_click_stop)

        counter = 0
        direction = 0
        
        frame_placeholder = st.empty()

        pd = PoseDetector(detectionCon=0.7,trackCon=0.7)


        if st.session_state['type']=='Start4':
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret,img = cap.read()
                # if not ret:
                #     cap = cv2.VideoCapture('vid1.mp4')
                #     continue

                img = cv2.resize(img,(1000,500))
                #cvzone.putTextRect(img,'AI Push Up Counter',[345,30],thickness=2,border=2,scale=2.5)
                pd.findPose(img,draw=0)
                lmlist ,bbox = pd.findPosition(img ,draw=0,bboxWithHands=0)


                angles(lmlist,11,13,15,12,14,16,drawpoints=1)



                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                frame_placeholder.image(img, "RGB")

                cv2.waitKey(1)
                
        elif st.session_state['type']=='Stop4': 
            st.write("The video capture has ended")

            st.write("---------")
            st.write("## Analytics") 
            st.write("You did ",st.session_state.counter4," reps")   
            
            # calories4=8.0*weight4/st.session_state.counter4
            calories4=0.32*st.session_state.counter4
            if calories4<goal_calorie4:
                st.write("You have burned ",calories4,"kcal of calories")
                st.write("You have not achieved your goal. Try again")

            else:
                st.write("You have burned ",calories4,"kcal of calories")
                st.write("You have achieved your goal. Congratulations")
            
            fig = go.Figure(data=[go.Bar(x=['Bicep Curls'], y=[calories4], name='Calories Burned')])

            fig.add_trace(go.Bar(x=['Bicep Curls'], y=[goal_calorie4], name='Goal Calorie'))

            # Set chart layout
            fig.update_layout(
                title='Calories Burned for Bicep Curls',
                xaxis_title='Exercise',
                yaxis_title='Calories Burned'
            )

            # Display the chart using Streamlit
            st.plotly_chart(fig)



    elif app_mode == "Shoulder press":
        st.markdown("## Shoulder Press")
        weight5 = st.slider('What is your weight?', 20, 130, 40)
        st.write("I'm ", weight5, 'kgs')

        st.write("-------------")

        goal_calorie5 = st.slider('Set a goal calorie to burn', 1, 200, 15)
        st.write("I want to burn", goal_calorie5, 'kcal')

        st.write("-------------")

        st.write(" Click on the Start button to start the live video feed.")
        st.write("##")

        # Creating Angle Finder Class
        class AngleFinder:
            def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawPoints):
                self.lmlist = lmlist
                self.p1, self.p2, self.p3 = p1, p2, p3
                self.p4, self.p5, self.p6 = p4, p5, p6
                self.drawPoints = drawPoints

            def angle(self):
                if len(self.lmlist) != 0:
                    x1, y1 = self.lmlist[self.p1][1:-1]
                    x2, y2 = self.lmlist[self.p2][1:-1]
                    x3, y3 = self.lmlist[self.p3][1:-1]
                    x4, y4 = self.lmlist[self.p4][1:-1]
                    x5, y5 = self.lmlist[self.p5][1:-1]
                    x6, y6 = self.lmlist[self.p6][1:-1]

                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                    leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                    rightHandAngle = int(np.interp(rightHandAngle, [-170, 180], [100, 0]))

                    return [leftHandAngle, rightHandAngle]
                return None

        if 'type' not in st.session_state:
            st.session_state.type = None

        def handle_click_start():
            st.session_state.type = "Start5"

        def handle_click_stop():
            st.write(st.session_state.counter5)
            st.session_state.type = "Stop5"

        start_button = st.button('Start', on_click=handle_click_start)
        stop_button = st.button('Stop', on_click=handle_click_stop)

        counter = 0
        direction = 0
        frame_placeholder = st.empty()
        detector = PoseDetector(detectionCon=0.7, trackCon=0.7)

        if st.session_state['type'] == 'Start5':
            cap = cv2.VideoCapture(0)

            while cap.isOpened():
                ret, img = cap.read()
                img = cv2.resize(img, (1000, 600))

                detector.findPose(img, draw=False)
                lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

                if lmList:
                    angle1 = AngleFinder(lmList, 11, 13, 15, 12, 14, 16, drawPoints=True)
                    hands = angle1.angle()

                    if hands:
                        left, right = hands

                        print(f"Left: {left}, Right: {right}")  # Debugging print statement

                        # Counting Shoulder Press Reps
                        if left >= 90 and right >= 90 and direction == 0:
                            counter += 0.5
                            st.session_state.counter5 = counter
                            direction = 1
                        if left <= 70 and right <= 70 and direction == 1:
                            counter += 0.5
                            st.session_state.counter5 = counter
                            direction = 0

                        # Display Score
                        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
                        cv2.putText(img, str(int(counter)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 6)

                        # Converting values for meters
                        leftval = np.interp(left, [0, 100], [400, 200])
                        rightval = np.interp(right, [0, 100], [400, 200])

                        # Adjusted Position to Ensure Visibility
                        cv2.putText(img, 'L', (900, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
                        cv2.rectangle(img, (850, 200), (900, 400), (0, 255, 0), 5)
                        cv2.rectangle(img, (850, int(leftval)), (900, 400), (255, 0, 0), -1)

                        cv2.putText(img, 'R', (50, 195), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 5)
                        cv2.rectangle(img, (20, 200), (70, 400), (0, 255, 0), 5)
                        cv2.rectangle(img, (20, int(rightval)), (70, 400), (255, 0, 0), -1)

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(img, "RGB")
                
                cv2.waitKey(1)

        elif st.session_state['type'] == 'Stop5':
            st.write("The video capture has ended")
            st.write("---------")
            st.write("## Analytics")
            st.write("You did ", st.session_state.counter5, " reps")

            calories5 = 0.22 * st.session_state.counter5
            if calories5 < goal_calorie5:
                st.write("You have burned ", calories5, "kcal of calories")
                st.write("You have not achieved your goal. Try again")
            else:
                st.write("You have burned ", calories5, "kcal of calories")
                st.write("You have achieved your goal. Congratulations")

            fig = go.Figure(data=[go.Bar(x=['Shoulder Press'], y=[calories5], name='Calories Burned')])
            fig.add_trace(go.Bar(x=['Shoulder Press'], y=[goal_calorie5], name='Goal Calorie'))

            fig.update_layout(
                title='Calories Burned for Shoulder Press',
                xaxis_title='Exercise',
                yaxis_title='Calories Burned'
            )

            st.plotly_chart(fig)