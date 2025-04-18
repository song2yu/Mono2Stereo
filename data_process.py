import cv2
import os

############################################# video path  ############################################################
input_video_path = 'path to your video' 

output_left_dir = 'real_images/left/'
output_right_dir = 'real_images/right/'

# load video
cap = cv2.VideoCapture(input_video_path)

frame_count = 0
init_count = 0 
save_cout = init_count + 1  

target_width = 960 # 1280 
target_height = 540 # 720
frame_interval = 8  # per 8 frames
# FPS 
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Frames per second (FPS): {fps}')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total frames in the video: {total_frames}')
print(f'About {total_frames/frame_interval} frames will be saved.')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    seg_start = int(fps * 300)
    seg_end = int(total_frames - fps * 600)
    if frame_count > seg_start and frame_count < seg_end:
        if frame_count % frame_interval == 0:  # (1080, 1920, 3)

            left_view = frame[:, :width // 2]
            right_view = frame[:, width // 2:]


            left_view_resized = cv2.resize(left_view, (target_width, target_height)) 
            right_view_resized = cv2.resize(right_view, (target_width, target_height))
            left_view_path = os.path.join(output_left_dir, f'{save_cout:09d}.jpg')
            right_view_path = os.path.join(output_right_dir, f'{save_cout:09d}.jpg')

            cv2.imwrite(left_view_path, left_view_resized)
            cv2.imwrite(right_view_path, right_view_resized)
            save_cout += 1
    if frame_count > seg_end:
        print("done")
        print(f'{save_cout - init_count} frames has been saved.')
        break
    frame_count += 1

cap.release()
