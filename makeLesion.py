import nibabel as ni
from nilearn import image
import numpy as np
import sys
import os
from subprocess import call
from scipy import ndimage
import shutil

def getinputs():
    inputs = sys.argv
    inputs.pop(0)  # remove first arg (name of file)
    t1 = ''  # T1w image
    t2 = ''  # T2w image
    m = ''  # lesion (typically drawn on T2w image)
    f = '0.4'  # 0.5 is default, I find it too much for pathological brains
    d = False
    t1healthy = ''
    outputDir = ''
    for i, v in enumerate(inputs):
        if v == '-t1les':
            t1 = inputs[i+1]
        if v == '-t2les':
            t2 = inputs[i+1]
        if v == '-m':  # mask
            m = inputs[i+1]
        if v == '-f':  # bet fractional intensity value
            f = inputs[i+1]
        if v == '-d':
            d = True
        if v == '-t1healthy':
            t1healthy = inputs[i+1]
        if v == '-outpth':
            outputDir = inputs[i+1]
    return t1, t2, m, t1healthy, f, d, outputDir


def fileparts(fnm):
    import os
    e_ = ''
    e2_ = ''
    nm_ = ''
    pth_ = ''
    pth_ = os.path.dirname(fnm)
    nm_, e_ = os.path.splitext(os.path.basename(fnm))
    if ".nii" in nm_:
        nm_, e2_ = os.path.splitext(nm_)
    ext_ = e2_+e_
    return pth_, nm_, ext_


def getfsldir():
    fsldir = os.getenv("FSLDIR")
    return fsldir


def getbet():
    fsldir = getfsldir()
    betpth = os.path.join(fsldir, "bin", "bet")
    return betpth


def getconvertxfm():
    fsldir = getfsldir()
    xfmpth = os.path.join(fsldir, "bin", "convert_xfm")
    return xfmpth


def getrfov():
    fsldir = getfsldir()
    rfovpth = os.path.join(fsldir, "bin", "robustfov")
    return rfovpth


def getfslmaths():
    fsldir = getfsldir()
    fslmathspth = os.path.join(fsldir, "bin", "fslmaths")
    return fslmathspth


def getflirt():
    fsldir = getfsldir()
    flirtpth = os.path.join(fsldir, 'bin', 'flirt')
    return flirtpth


def opennii(fname):
    obj = ni.load(fname)
    imgdata = obj.get_data()
    return obj, imgdata


def mirror(imgmat):
    mirimg = np.flipud(imgmat)
    return mirimg


def makenii(obj, imgdata):
    outobj = ni.Nifti1Image(imgdata, obj.affine, obj.header)
    return outobj


def savenii(obj, name):
    obj.to_filename(name)
    return name


def doflirt(inputFile, referenceFile, dof):
    import os
    # dof = "6"  # rigid body only
    ipth, inm, ie = fileparts(inputFile)
    rpth, rnm, re = fileparts(referenceFile)
    outmat = os.path.join(ipth, "r" + inm + ".mat")
    outimg = os.path.join(ipth, "r" + inm + ie)
    cmd = [
        flirt,
        "-dof",
        dof,
        "-in",
        inputFile,
        "-ref",
        referenceFile,
        "-out",
        outimg,
        "-omat",
        outmat
    ]
    print(cmd)
    if os.path.exists(outimg):
        return outimg, outmat
    call(cmd)
    return outimg, outmat


def applyflirt(inputFile, referenceFile, referenceMat):
    import os
    ipth, inm, ie = fileparts(inputFile)
    rpth, rnm, re = fileparts(referenceFile)
    outimg = os.path.join(ipth, "r" + inm + ie)
    cmd = [
        flirt,
        "-in",
        inputFile,
        "-ref",
        referenceFile,
        "-applyxfm",
        "-init",
        referenceMat,
        "-out",
        outimg
    ]
    print(cmd)
    if os.path.exists(outimg):
        return outimg
    call(cmd)
    return outimg


def threshholdMask(fnm, t):
    I, img = opennii(fnm)
    i = img.flatten()
    i[i >= t] = 1
    i[i < t] = 0
    o = np.reshape(i, img.shape)
    newnii = makenii(I, o)
    pth, nm, e = fileparts(fnm)
    onm = savenii(newnii, os.path.join(pth, "t" + nm + e))
    return onm


def healimg(img, msk):
    mirimg = mirror(img)
    i = img.flatten()
    mi = mirimg.flatten()
    m = msk.flatten()
    # (img(:) .* (1.0-imgLesion(:)))+ (imgFlip(:) .* imgLesion(:))
    himg = (i * (1.0-m)) + (mi * m)
    o = np.reshape(himg, img.shape)
    return o


def deleteTempFiles(files):
    for f in files:
        print("deleting file: {}".format(f))
        os.remove(f)


def getfslreorient():
    fsldir = getfsldir()
    fslrepth = os.path.join(fsldir, "bin", "fslreorient2std")
    return fslrepth


def fslreorient(t1, t2, le, t1h):
    fslreorient2std = getfslreorient()
    fileList = [t1, t2, le, t1h]
    outList = []
    for file in fileList:
        # make out file
        pth, nm, e = fileparts(file)
        outFile = os.path.join(pth, "o" + nm + e)  # "o" for reoriented file
        cmd = [
            fslreorient2std,
            file,
            outFile,
        ]
        outList.append(outFile)
        print(cmd)
        call(cmd)
    return outList[0], outList[1], outList[2], outList[3]


def doBet(fnam, f):
    inObj, inImg = opennii(fnam)
    com = ndimage.measurements.center_of_mass(inImg)
    p, n, e = fileparts(fnam)
    outFile = os.path.join(p, "b" + n + e)  # "o" for reoriented file
    cmd = [
        bet,
        fnam,
        outFile,
        "-f",
        f,
        "-R",
    ]
    print(cmd)
    if os.path.exists(outFile):
        return outFile
    call(cmd)
    return outFile


def insertimg(recipient, donor, donormask):
    robj, rimg = opennii(recipient)
    dobj, dimg = opennii(donor)
    mobj, mimg = opennii(donormask)
    rpth, rnm, re = fileparts(recipient)
    dpth, dnm, de = fileparts(donor)
    dmpth, dmnm, dme = fileparts(donormask)
    # (img(:) .* (1.0-imgLesion(:)))+ (imgFlip(:) .* imgLesion(:))
    i = rimg.flatten()
    mi = dimg.flatten()
    m = mimg.flatten()
    oimg = (i * (1.0-m)) + (mi * m)
    o = np.reshape(oimg, rimg.shape)
    #rimg[mimg > 0] = dimg[mimg > 0]
    outObj = makenii(robj, o)
    outlesObj = makenii(mobj, mimg)
    outname = os.path.join(pth, rnm + "_" + dnm + re)
    outlesname = os.path.join(pth, rnm + "_" + dmnm + re)
    savenii(outObj, outname)
    savenii(outlesObj, outlesname)
    return outname, outlesname


def meanScale(inputImg, referenceImg):
    inObj, inImg = opennii(inputImg)
    refObj, refImg = opennii(referenceImg)
    # imgL1 * (mean(imgH1(:))/mean(imgL1(:)))
    scaledImg = inImg * (np.mean(refImg)/np.mean(inImg))
    outObj = makenii(inObj, scaledImg)
    pth, nm, e = fileparts(inputImg)
    outName = savenii(outObj, os.path.join(pth, "d" + nm + e))
    return outName


def makeLesImg(inputImg, maskImg):
    inObj, inImg = opennii(inputImg)
    mObj, mImg = opennii(maskImg)
    outImg = np.zeros_like(inImg)
    outImg[mImg > 0] = inImg[mImg > 0]
    outObj = makenii(inObj, outImg)
    pth, nm, e = fileparts(inputImg)
    outName = savenii(outObj, os.path.join(pth, "lesionData_" + nm + e))
    return outName


def getOrigin(fname):
    hdr, img = opennii(fname)
    print(hdr.affine)
    from nibabel.affines import apply_affine
    import numpy.linalg as npl
    res = apply_affine(npl.inv(hdr.affine), [0, 0, 0])
    #M = hdr.affine[:3, :3]
    #res = M.dot([0, 0, 0]) + hdr.affine[:3, 3]
    return res


def addToDeleteList(dlist, fname):
    dlist.append(fname)
    return dlist


def cropZ(fname):
    # crop in z dimension to improve bet results
    pth, nm, e = fileparts(fname)
    outname = os.path.join(pth, "f" + nm + e)
    outmname = os.path.join(pth, "f" + nm + ".mat")
    cmd = [
        rfov,
        "-i",
        fname,
        "-r",
        outname,
        "-m",
        outmname
    ]
    print(cmd)
    call(cmd)
    return outname, outmname


def concatxfm(amat, bmat):
    ptha, nma, ea = fileparts(amat)
    pthb, nmb, eb = fileparts(bmat)
    outmat = os.path.join(ptha, nma + "_+_" + nmb + ea)
    cmd = [
        xfm,
        "-omat",
        outmat,
        "-concat",
        amat,
        bmat
    ]
    print(cmd)
    call(cmd)
    return outmat


def moveFile(infile, newfolder):
    pth, nm, e = fileparts(infile)
    shutil.move(infile, os.path.join(newfolder, nm + e))


def changeDataType(infile, newtype):
    fslmaths = getfslmaths()
    cmd = [
        fslmaths,
        infile,
        "-add",
        "0",
        infile,
        "-odt",
        newtype
    ]
    print(cmd)
    call(cmd)
    return



dlist = []
# get bet and flirt commands
bet = getbet()  # get path to bet command
flirt = getflirt()  # get path to flirt command
rfov = getrfov()
xfm = getconvertxfm()

# get inputs to makeLesion.py
T1Name, T2Name, lesName, t1healthyName, f, d, outputDir = getinputs()  # parse inputs, return file names

# reorient input images to std fsl space
T1Name, T2Name, lesName, t1healthyName = fslreorient(T1Name, T2Name, lesName, t1healthyName)
dlist = addToDeleteList(dlist, T1Name)
dlist = addToDeleteList(dlist, T2Name)
dlist = addToDeleteList(dlist, lesName)
dlist = addToDeleteList(dlist, t1healthyName)

# crop neck from lesion images
T1Name, cropZmat = cropZ(T1Name)
dlist = addToDeleteList(dlist, T1Name)
dlist = addToDeleteList(dlist, cropZmat)

# register lesionT2 image to lesionT1 image space
rT2, rmat = doflirt(T2Name, T1Name, "6")  # register T2 to T1
dlist = addToDeleteList(dlist, rT2)
dlist = addToDeleteList(dlist, rmat)

# apply the previous transform to the lesion mask (usually drawn on T2)
rLesion = applyflirt(lesName, rT2, rmat)
dlist = addToDeleteList(dlist, rLesion)
rLesion = threshholdMask(rLesion, 0.5)
dlist = addToDeleteList(dlist, rLesion)


# crop neck from healthy T1
t1healthyName, zmt = cropZ(t1healthyName)
dlist = addToDeleteList(dlist, zmt)
dlist = addToDeleteList(dlist, t1healthyName)

# open lesion mask image and T1 image
LES, lesimg = opennii(rLesion)
T1, T1img = opennii(T1Name)  # open image, return obj and img data

# smooth lesion mask before healing to blend better
LES = image.smooth_img(LES, fwhm=3)  # feather edges
lesimg = LES.get_data()

# make healed T1 image (flip undamaged hemisphere voxels into lesion area)
h = healimg(T1img, lesimg)
healed = makenii(T1, h)
pth, nm, e = fileparts(T1Name)
hT1 = savenii(healed, os.path.join(pth, "h" + nm + e))
dlist = addToDeleteList(dlist, hT1)

# bet healed lesion T1
bhnii = doBet(hT1, f)
dlist = addToDeleteList(dlist, bhnii)

# bet HealthyT1
bhealthyT1 = doBet(t1healthyName, f)
dlist = addToDeleteList(dlist, bhealthyT1)

# register brain of healed image to brain of Healthy T1
rbhnii, rbhniimat = doflirt(bhnii, bhealthyT1, "12")
dlist = addToDeleteList(dlist, rbhnii)
dlist = addToDeleteList(dlist, rbhniimat)

# apply flirt to LesLes
rrLesion = applyflirt(rLesion, bhealthyT1, rbhniimat)
dlist = addToDeleteList(dlist, rrLesion)

rT1Name = applyflirt(T1Name, bhealthyT1, rbhniimat)
dlist = addToDeleteList(dlist, rT1Name)


# feather edges of les
LES, lesimg = opennii(rrLesion)
LES = image.smooth_img(LES, fwhm=8)  # feather edges
pth, nm, e = fileparts(rrLesion)
srrLesion = savenii(LES, os.path.join(pth, "s" + nm + e))
dlist = addToDeleteList(dlist, srrLesion)

# do mean scaling
st1healthyName = meanScale(t1healthyName, rT1Name)
dlist = addToDeleteList(dlist, st1healthyName)

# Put lesion in Healthy T1 that has not been brain extracted
# bWithLes = insertimg(rbhnii, srT1Name, rrLesion)
bWithLes, srrLesion = insertimg(st1healthyName, rT1Name, srrLesion)
tsrrLesion = threshholdMask(srrLesion, 0.5)

# change data type to reduce file size
changeDataType(bWithLes, "short")  # short is 16 bit int
changeDataType(tsrrLesion, "short")  # short is 16 bit int
moveFile(bWithLes, outputDir)
moveFile(tsrrLesion, outputDir)

if d:
    deleteTempFiles(dlist)
