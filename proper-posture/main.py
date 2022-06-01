import numpy as np
import os
import argparse
import sys
import subprocess
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import warnings
from scipy import spatial
from random import seed
from random import randint

warnings.simplefilter(action='ignore', category=FutureWarning)

cap = cv2.VideoCapture('media\sample_video.mp4')

cosine_sim_avgs = []

template_exercise = ""
template_weight = ""
template_posture = ""

client_body_keypoints_df = pd.DataFrame()
client_left_hand_df = pd.DataFrame()
client_left_elbow_df = pd.DataFrame()
client_left_shoulder_df = pd.DataFrame()
client_center_df = pd.DataFrame()
client_pelvis_df = pd.DataFrame()
client_left_hip_df = pd.DataFrame()
client_left_knee_df = pd.DataFrame()
client_left_foot_df = pd.DataFrame()

template_body_keypoints_df = pd.DataFrame()
template_left_hand_df = pd.DataFrame()
template_left_elbow_df = pd.DataFrame()
template_left_shoulder_df = pd.DataFrame()
template_center_df = pd.DataFrame()
template_pelvis_df = pd.DataFrame()
template_left_hip_df = pd.DataFrame()
template_left_knee_df = pd.DataFrame()
template_left_foot_df = pd.DataFrame()


def get_vid_properties():
    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    cap.release()
    return width, height


def load_keypoints(person_type, exercise_type, weight_type, quality_type, rep_type):
    # Load keypoint data from JSON output
    column_names = ['x', 'y', 'acc']

    # Paths - should be the folder where Open Pose JSON output was stored
    exercise_type_file_name = exercise_type
    exercise_type_file_name = exercise_type_file_name.replace("-", "_")
    folder_name = rep_type + "_" + exercise_type_file_name + "_" + weight_type + "_" + quality_type
    # path_to_json = os.path.join(os.path.abspath(os.getcwd()), 'media/sample_video/')

    if person_type in "client":
        path_to_json = os.path.join("..", "media/sample_video/")
    elif person_type in "template":
        file_dir = exercise_type + "/" + weight_type + "/" + quality_type + "/" + folder_name + "/"
        path_to_json = os.path.join("..", file_dir)

    # path_to_json = output_path

    if person_type in "client":
        global client_body_keypoints_df
        global client_left_hand_df
        global client_left_elbow_df
        global client_left_shoulder_df
        global client_center_df
        global client_pelvis_df
        global client_left_hip_df
        global client_left_knee_df
        global client_left_foot_df

        client_body_keypoints_df = pd.DataFrame()
        client_left_hand_df = pd.DataFrame()
        client_left_elbow_df = pd.DataFrame()
        client_left_shoulder_df = pd.DataFrame()
        client_center_df = pd.DataFrame()
        client_pelvis_df = pd.DataFrame()
        client_left_hip_df = pd.DataFrame()
        client_left_knee_df = pd.DataFrame()
        client_left_foot_df = pd.DataFrame()
    elif person_type in "template":
        global template_body_keypoints_df
        global template_left_hand_df
        global template_left_elbow_df
        global template_left_shoulder_df
        global template_center_df
        global template_pelvis_df
        global template_left_hip_df
        global template_left_knee_df
        global template_left_foot_df

        template_body_keypoints_df = pd.DataFrame()
        template_left_hand_df = pd.DataFrame()
        template_left_elbow_df = pd.DataFrame()
        template_left_shoulder_df = pd.DataFrame()
        template_center_df = pd.DataFrame()
        template_pelvis_df = pd.DataFrame()
        template_left_hip_df = pd.DataFrame()
        template_left_knee_df = pd.DataFrame()
        template_left_foot_df = pd.DataFrame()

    # Import Json files, pos_json = position JSON
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    # print('Found: ', len(json_files), 'json keypoint frame files')
    count = 0

    width, height = get_vid_properties()

    # Loop through all json files in output directory
    # Each file is a frame in the video
    # If multiple people are detected - choose the most centered high confidence points
    for file in json_files:
        temp_df = json.load(open(path_to_json + file))
        temp = []
        for k, v in temp_df['part_candidates'][0].items():

            # Single point detected
            if len(v) < 4:
                temp.append(v)
                # print('Extracted highest confidence points: ',v)

            # Multiple points detected
            elif len(v) > 4:
                near_middle = width
                np_v = np.array(v)

                # Reshape to x,y,confidence
                np_v_reshape = np_v.reshape(int(len(np_v) / 3), 3)
                np_v_temp = []
                # compare x values
                for pt in np_v_reshape:
                    if (np.absolute(pt[0] - width / 2) < near_middle):
                        near_middle = np.absolute(pt[0] - width / 2)
                        np_v_temp = list(pt)

                temp.append(np_v_temp)
                # print('Extracted highest confidence points: ',v[index_highest_confidence-2:index_highest_confidence+1])
            else:
                # No detection - record zeros
                temp.append([0, 0, 0])

        temp_df = pd.DataFrame(temp)
        temp_df = temp_df.fillna(0)

        # IF TYPE IS OF CLIENT ADD KEYPOINTS TO CLIENT
        if person_type in "client":
            try:
                prev_temp_df = temp_df
                client_body_keypoints_df = client_body_keypoints_df.append(temp_df)
                client_left_hand_df = client_left_hand_df.append(temp_df.iloc[7].astype(int))
                client_left_elbow_df = client_left_elbow_df.append(temp_df.iloc[6].astype(int))
                client_left_shoulder_df = client_left_shoulder_df.append(temp_df.iloc[5].astype(int))
                client_center_df = client_center_df.append(temp_df.iloc[1].astype(int))
                client_pelvis_df = client_pelvis_df.append(temp_df.iloc[8].astype(int))
                client_left_hip_df = client_left_hip_df.append(temp_df.iloc[12].astype(int))
                client_left_knee_df = client_left_knee_df.append(temp_df.iloc[13].astype(int))
                client_left_foot_df = client_left_foot_df.append(temp_df.iloc[14].astype(int))
            except:
                print('Missing Point at: ', file)
        # IF TYPE IS OF TEMPLATE ADD KEYPOINTS TO TEMPLATE
        elif person_type in "template":
            try:
                prev_temp_df = temp_df
                template_body_keypoints_df = template_body_keypoints_df.append(temp_df)
                template_left_hand_df = template_left_hand_df.append(temp_df.iloc[7].astype(int))
                template_left_elbow_df = template_left_elbow_df.append(temp_df.iloc[6].astype(int))
                template_left_shoulder_df = template_left_shoulder_df.append(temp_df.iloc[5].astype(int))
                template_center_df = template_center_df.append(temp_df.iloc[1].astype(int))
                template_pelvis_df = template_pelvis_df.append(temp_df.iloc[8].astype(int))
                template_left_hip_df = template_left_hip_df.append(temp_df.iloc[12].astype(int))
                template_left_knee_df = template_left_knee_df.append(temp_df.iloc[13].astype(int))
                template_left_foot_df = template_left_foot_df.append(temp_df.iloc[14].astype(int))
            except:
                print('Missing Point at: ', file)

    if person_type in "client":
        client_body_keypoints_df.columns = column_names
        client_left_hand_df.columns = column_names
        client_left_elbow_df.columns = column_names
        client_left_shoulder_df.columns = column_names
        client_center_df.columns = column_names
        client_pelvis_df.columns = column_names
        client_left_hip_df.columns = column_names
        client_left_knee_df.columns = column_names
        client_left_foot_df.columns = column_names

        client_body_keypoints_df.reset_index()
        client_left_hand_df = client_left_hand_df.reset_index(drop=True)
        client_left_elbow_df = client_left_elbow_df.reset_index(drop=True)
        client_left_shoulder_df = client_left_shoulder_df.reset_index(drop=True)
        client_center_df = client_center_df.reset_index(drop=True)
        client_pelvis_df = client_pelvis_df.reset_index(drop=True)
        client_left_hip_df = client_left_hip_df.reset_index(drop=True)
        client_left_knee_df = client_left_knee_df.reset_index(drop=True)
        client_left_foot_df = client_left_foot_df.reset_index(drop=True)
    elif person_type in "template":
        template_body_keypoints_df.columns = column_names
        template_left_hand_df.columns = column_names
        template_left_elbow_df.columns = column_names
        template_left_shoulder_df.columns = column_names
        template_center_df.columns = column_names
        template_pelvis_df.columns = column_names
        template_left_hip_df.columns = column_names
        template_left_knee_df.columns = column_names
        template_left_foot_df.columns = column_names

        template_body_keypoints_df.reset_index()
        template_left_hand_df = template_left_hand_df.reset_index(drop=True)
        template_left_elbow_df = template_left_elbow_df.reset_index(drop=True)
        template_left_shoulder_df = template_left_shoulder_df.reset_index(drop=True)
        template_center_df = template_center_df.reset_index(drop=True)
        template_pelvis_df = template_pelvis_df.reset_index(drop=True)
        template_left_hip_df = template_left_hip_df.reset_index(drop=True)
        template_left_knee_df = template_left_knee_df.reset_index(drop=True)
        template_left_foot_df = template_left_foot_df.reset_index(drop=True)

    # print("Client Data Frame Done...")

def pose_estimation():
    print("Performing Pose Estimation...")
    video = os.path.basename("sample_video.mp4")

    # Type 0 = Bicep Curl, 1 = Barbell Row

    # COMMENT POSE ESTIMATION WHEN WORKING ON OTHER STUFF

    # Get Path of Open Pose
    output_path = os.path.join(os.path.abspath(os.getcwd()), "media/" + os.path.splitext(video)[0])

    path_to_json = os.path.join(os.path.abspath(os.getcwd()), 'media/sample_video/')

    if os.path.exists(path_to_json):
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        for file in json_files:
            os.remove(path_to_json + file)

    openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
    os.chdir('openpose')

    subprocess.call([openpose_path,
                    '--video', os.path.join("..", "sample_video.mp4"),
                    '--write_json', output_path,
                    '--part_candidates'])

    src_path = os.path.join(os.path.abspath(os.getcwd()), video)
    output_path = os.path.join(os.path.abspath(os.getcwd()), "media/" + video)

    print("Pose Estimation Done...")

def compare_keypoints():
    global cosine_sim_avgs

    global template_exercise
    global template_weight
    global template_posture

    global client_body_keypoints_df
    global client_left_hand_df
    global client_left_elbow_df
    global client_left_shoulder_df
    global client_center_df
    global client_pelvis_df
    global client_left_hip_df
    global client_left_knee_df
    global client_left_foot_df

    global template_body_keypoints_df
    global template_left_hand_df
    global template_left_elbow_df
    global template_left_shoulder_df
    global template_center_df
    global template_pelvis_df
    global template_left_hip_df
    global template_left_knee_df
    global template_left_foot_df

    reps = [1, 2, 3, 4, 5, 6]
    cosine_sim_avgs = []
    for c in reps:
        print(c, " -------------------")

        # print("Loading Template ...")

        load_keypoints("client", "barbell-bicep-curl", "light", "strict", "1")
        load_keypoints("template", template_exercise, template_weight, template_posture, str(c))

        client_keypoints_group = [client_left_hand_df, client_left_elbow_df, client_left_shoulder_df, client_center_df,
                                  client_pelvis_df, client_left_hip_df, client_left_knee_df, client_left_foot_df]
        template_keypoints_group = [template_left_hand_df, template_left_elbow_df, template_left_shoulder_df,
                                    template_center_df, template_pelvis_df, template_left_hip_df, template_left_knee_df,
                                    template_left_foot_df]

        client_len = len(client_left_elbow_df)
        template_len = len(template_left_elbow_df)

        if client_len > template_len:
            diff = client_left_elbow_df.shape[0] - template_left_elbow_df.shape[0]
            for x in range(len(client_keypoints_group)):
                for i in range(diff):
                    rand = randint(0, len(client_keypoints_group[x]) - 1)
                    client_keypoints_group[x] = client_keypoints_group[x].drop([rand, rand])
                    client_keypoints_group[x] = client_keypoints_group[x].reset_index(drop=True)
        elif template_len > client_len:
            diff = len(template_left_elbow_df) - len(client_left_elbow_df)
            for x in range(len(template_keypoints_group)):
                for i in range(diff):
                    rand = randint(0, len(template_keypoints_group[x]) - 1)
                    template_keypoints_group[x] = template_keypoints_group[x].drop([rand, rand])
                    template_keypoints_group[x] = template_keypoints_group[x].reset_index(drop=True)

        cosine_set = []

        for c_g, t_g in zip(client_keypoints_group, template_keypoints_group):
            for index, row in t_g.iterrows():
                set1 = [c_g.x[index], c_g.y[index]]
                set2 = [t_g.x[index], t_g.y[index]]

                cosine_set.append(spatial.distance.cosine(set1, set2))
                # print("Cosine Similarity: ", spatial.distance.cosine(set1, set2))
        avg = sum(cosine_set) / len(cosine_set)
        cosine_sim_avgs.append(avg)

        # print("Cosine Similarity Done...")
        # print("Template Done...")

def main():
    print("Performing Pose Estimation on 'sample_video.mp4'...")

    global cosine_sim_avgs

    global template_exercise
    global template_weight
    global template_posture

    template_exercise = "barbell-bicep-curl"
    template_weight = "light"
    template_posture = "strict"

    pose_estimation()

    print("----------------------------------")
    print("-",template_exercise, " / " ,template_weight, " / " ,template_posture, "-")
    print("----------------------------------")

    compare_keypoints()

    print("Cosine Averages: ")
    for i in cosine_sim_avgs:
        print(i)

    template_exercise = "barbell-bicep-curl"
    template_weight = "light"
    template_posture = "okay"

    print("----------------------------------")
    print("-", template_exercise, " / ", template_weight, " / ", template_posture, "-")
    print("----------------------------------")

    compare_keypoints()

    print("Cosine Averages: ")
    for i in cosine_sim_avgs:
        print(i)

    template_exercise = "barbell-bicep-curl"
    template_weight = "light"
    template_posture = "bad"

    print("----------------------------------")
    print("-", template_exercise, " / ", template_weight, " / ", template_posture, "-")
    print("----------------------------------")

    compare_keypoints()

    print("Cosine Averages: ")
    for i in cosine_sim_avgs:
        print(i)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
