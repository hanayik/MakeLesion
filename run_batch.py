import os
import sys
from glob import glob


basepth = "/Volumes/Taylor1TB/lesion_norm"

# 1) set base paths
lesionDir = os.path.join(basepth, "strokeImgs")
healthyDir = os.path.join(basepth, "healthyImgs")
savepth = os.path.join(basepth, "artificialLesion")



# 2) list sub folders in lesion folder
lesion_subdirs = glob(os.path.join(lesionDir,"M*/"))

# 3) list anat files in healthy folder
healthy_files = glob(os.path.join(healthyDir,"*_anat.nii"))

for i, T1healthy in enumerate(healthy_files):
    for j, sub in enumerate(lesion_subdirs):
        lesT1 = glob(os.path.join(sub, "T1_*.nii"))
        lesLes = glob(os.path.join(sub, "Lesion_*.nii"))
        lesT2 = glob(os.path.join(sub, "T2_*.nii"))
        if not lesT1:
            continue
        if not lesLes:
            continue
        if not lesT2:
            continue

        lesT1 = glob(os.path.join(sub, "T1_*.nii"))[0]
        lesLes = glob(os.path.join(sub, "Lesion_*.nii"))[0]
        lesT2 = glob(os.path.join(sub, "T2_*.nii"))[0]
        print(lesT1)
        print(lesLes)
        print(lesT2)

        cmd = "python makeLesion.py -t1les {} -t2les {} -m {} -t1healthy {} -outpth {} -d".format(
            lesT1,
            lesT2,
            lesLes,
            T1healthy,
            savepth
        )
        print(cmd)
        os.system(cmd)





