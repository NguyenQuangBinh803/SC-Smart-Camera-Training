
print("Init parameter ... ")

thermal_data = 36.5

global_locs = []
global_mask = []

face_detect_status = False
mask_detect_status = False
human_appear_status = False

frame = None
thermal = None
frame_face = {}
frame_face["frame"] = frame
frame_face["thermal"] = thermal

global_face_image = None
global_human_name = None

face_area = 0

# Training mode parameters
start_collecting = False
start_training = False
target_name_entered = False
target_name = None
training_status =  0
collecting_status =  0



print("Done init parameters ... ")



