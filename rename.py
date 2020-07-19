import os
import shutil

#reannotate (dont miss any files)
#used this script
#modify the Faster-RCNN script

def renameImage(): 
    count = 1
    imagefolderlist = os.listdir(imagepath) 

    for folder in imagefolderlist:   

        if(folder != "temp"):
            currentfolder = os.path.join(imagepath, folder)
            filelist = os.listdir(currentfolder)

            for file in filelist:
                oldname = os.path.join(currentfolder,file)
                filename = os.path.splitext(file)[0]   
                filetype = os.path.splitext(file)[1]   
                newname=os.path.join(currentfolder, str(count).zfill(6)+filetype) 
                mydict[oldname] = newname
                os.rename(oldname, newname)
                shutil.copy2(newname, imagepath + "\\temp")
                count += 1

def renameAnnotation():
     annotationfolderlist = os.listdir(annotationpath) 

     for folder in annotationfolderlist:   

         if(folder != "temp"):
             currentfolder = os.path.join(annotationpath, folder)
             filelist = os.listdir(currentfolder)

             for file in filelist:
                 oldname = os.path.join(currentfolder,file)
                 dictoldname = oldname.replace("Annotations\\Object", "Images")
                 dictoldname = dictoldname[:-4] + ".jpg"
                 dictnewname = mydict[dictoldname]
                 dictnewname = dictnewname[:-4] + ".xml"
                 newname = dictnewname.replace("Images", "Annotations\\Object")
                 os.rename(oldname, newname)
                 shutil.copy2(newname, annotationpath + "\\temp")

def updatelist():
    traintext = "C:\\Users\\29469\\Downloads\\train.txt"
    newtraintext = "C:\\Users\\29469\\Downloads\\newtrain.txt"
    testtext = "C:\\Users\\29469\\Downloads\\test.txt"
    newtesttext = "C:\\Users\\29469\\Downloads\\newtest.txt"
    valtext = "C:\\Users\\29469\\Downloads\\val.txt"
    newvaltext = "C:\\Users\\29469\\Downloads\\newval.txt"
    trainvaltext = "C:\\Users\\29469\\Downloads\\trainval.txt"
    newtrainvaltext = "C:\\Users\\29469\\Downloads\\newtrainval.txt"
    
    updateOneList(traintext, newtraintext)
    updateOneList(testtext, newtesttext)
    updateOneList(valtext, newvaltext)
    updateOneList(trainvaltext, newtrainvaltext)

def updateOneList(oldtxt, newtxt):
    with open(oldtxt) as fp:
        line = fp.readline()
        
        while line:
            line = line.replace("/", "\\")
            line = line.strip()
            line = "C:\\Users\\29469\\Downloads\\GTEA_GAZE_PLUS\\" + line
            newline = mydict[line]
            lastslash = newline.rfind("\\")
            newline = newline[lastslash+1:]
            with open(newtxt, "a") as myfile:
                myfile.write(newline+"\n")
            line = fp.readline()


imagepath = "C:\\Users\\29469\\Downloads\\GTEA_GAZE_PLUS\\GTEA_GAZE_PLUS\\Images"
annotationpath = "C:\\Users\\29469\\Downloads\\GTEA_GAZE_PLUS\\GTEA_GAZE_PLUS\\Annotations\\Object"

mydict = {}

renameImage()
renameAnnotation()
updatelist()