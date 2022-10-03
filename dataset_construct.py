import json
import os
import numpy as np
np.set_printoptions(suppress=True)

basedir = "/home/enpu/nerfplusplus/out8/"
raw_pose = "posed_images/kai_cameras_normalized.json"
out = "dataset"
f = open(basedir + raw_pose,"r")
content = f.read()
a = json.loads(content)
for key in a:
    for key_in in a[key]:
        if key_in == "K":
            intrinsics = a[key][key_in]
            lines = [
                "{} {} {} {} ".format(intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]),
                "{} {} {} {} ".format(intrinsics[4], intrinsics[5], intrinsics[6], intrinsics[7]),
                "{} {} {} {} ".format(intrinsics[8], intrinsics[9], intrinsics[10], intrinsics[11]),
                "0.0 0.0 0.0 1.0",
            ]
            with open(os.path.join(basedir, out, "intrinsics", "%s.txt" % key.split('.', 1)[0]), "w+") as f:
                f.writelines(lines)
            #print(a[key][key_in])
            #print(key)
        if key_in == "W2C":
            #print(a[key][key_in])
            #print(key)a[key][key_in]
            w2c = np.array(a[key][key_in]).reshape(4,4)
            #print(w2c)
            c2w = np.linalg.inv(w2c)
            lines = [
                "{} {} {} {} ".format(c2w[0, 0], c2w[0, 1], c2w[0, 2], c2w[0, 3]),
                "{} {} {} {} ".format(c2w[1, 0], c2w[1, 1], c2w[1, 2], c2w[1, 3]),
                "{} {} {} {} ".format(c2w[2, 0], c2w[2, 1], c2w[2, 2], c2w[2, 3]),
                "0.0 0.0 0.0 1.0",
            ]
            pose_out = c2w.reshape(1,16)[0].tolist()
            print(c2w)
            with open(os.path.join(basedir, out, "pose", "%s.txt" % key.split('.', 1)[0]), "w+") as f:
                f.writelines(lines)

#print(a)
f.close()