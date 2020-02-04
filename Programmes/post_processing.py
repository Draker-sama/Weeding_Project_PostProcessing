import cv2
import numpy as np
import pandas
import os
import csv
from sklearn import linear_model
import matplotlib.pyplot as plt


#Chemins vers les dossiers suivant contenant : fichier csv avec les bounding boxes, les images originales
folder_path_csv="/home/hugo/Projet3a/Annotations/Test/Image_Detection"
folder_path_images="/home/hugo/Projet3a/Annotations/Test/Image_Detection/base"

#Chemins vers les dossiers où enregistrer : les images avec droites de régression, images avec masque vert appliqué, images avec pixels verts coloriés
folder_new_image_regression="/home/hugo/Projet3a/Annotations/Test/Image_Detection/new_image_regression"
folder_new_image_masque="/home/hugo/Projet3a/Annotations/Test/Image_Detection/new_image_masque"
folder_new_image_conversion="/home/hugo/Projet3a/Annotations/Test/Image_Detection/new_image_conversion"


#Parcours de toutes les images dans le fichier contenant les images originales
#Architecture probablement à revoir pour l'intégration du post-processing après Yolo qui ne devra traiter que l'image qui vient d'être analysé par YOLO
for file in os.listdir(folder_path_images):
    liste_bb=[] #mise à zéro liste bounding_box 
    os.chdir(folder_path_csv) #Changement de dossier
    with open('Detection_Results.csv',newline='') as csvfile:  #Lecture fichier csv et remplissage liste_bb avec les angles de la bounding_box
        reader=csv.DictReader(csvfile)
        for ligne in reader:
            if ligne['image'] == file:
                if ligne['label'] == '0':
                    liste_bb.append([int(ligne['xmin']),int(ligne['ymin']),int(ligne['xmax']),int(ligne['ymax'])])

    #Traitement de l'image
    os.chdir(folder_path_images) #Changement de dossier

    division=3 #Partage de l'image en 3 parties égales pour faire la régression sur les plants de gauche, du milieu et de droite, peut être mis à 2
    #dans le cas où il n'y pas une rangée centrale (correspond au cas réel du robot)

    #Création des listes contenant les futurs régression linéraires de chaque rangée
    liste_gauche=[]
    liste_droite=[]
    liste_milieu=[]

    #création de image
    image=cv2.imread(file)
    image2=image.copy()
    height,width,channel=image.shape

    #Remplissage des liste gauche, milieu et droite en fonction de la position des centres des bounding box et de la division de l'image (2 ou 3)
    for i in range(0,len(liste_bb)):
        cv2.rectangle(image,(liste_bb[i][0],liste_bb[i][1]),(liste_bb[i][2],liste_bb[i][3]),(0,0,255),5)
        centre=[int((liste_bb[i][2]+liste_bb[i][0])/2),int((liste_bb[i][3]+liste_bb[i][1])/2)]
        if centre[0] < int(width/division) :
            liste_gauche.append(centre)
        elif centre[0] > int((division-1)*width/division):
            liste_droite.append(centre)
        else :
            liste_milieu.append(centre)

    #régression linéaire pour retrouver les droites correspondants à la ligne de lavandin
    #cote gauche
    if len(liste_gauche) != 0:
        x_gauche=[i[0] for i in liste_gauche]
        y_gauche=[i[1] for i in liste_gauche]
        y_gauche=np.array(y_gauche).reshape(-1,1)
        regr=linear_model.LinearRegression()
        regr.fit(y_gauche,x_gauche)
        y_prediction=regr.predict(y_gauche)
        cv2.line(image,(regr.coef_*0+regr.intercept_,0),(regr.coef_*height+regr.intercept_,height),(0,0,255),5)

    #cote droit
    if len(liste_droite) != 0:
        x_droite=[i[0] for i in liste_droite]
        y_droite=[i[1] for i in liste_droite]
        y_droite=np.array(y_droite).reshape(-1,1)
        regr=linear_model.LinearRegression()
        regr.fit(y_droite,x_droite)
        y_prediction=regr.predict(y_droite)
        cv2.line(image,(regr.coef_*0+regr.intercept_,0),(regr.coef_*height+regr.intercept_,height),(0,0,255),5)

    #milieu

    if len(liste_milieu) != 0:
        x_milieu=[i[0] for i in liste_milieu]
        y_milieu=[i[1] for i in liste_milieu]
        y_milieu=np.array(y_milieu).reshape(-1,1)
        regr=linear_model.LinearRegression()
        regr.fit(y_milieu,x_milieu)
        y_prediction=regr.predict(y_milieu)
        cv2.line(image,(regr.coef_*0+regr.intercept_,0),(regr.coef_*height+regr.intercept_,height),(0,0,255),5)


    #Création d'un masque pour appliquer un masque à l'image

    #Création d'une plage de couleur dépendant de la valeur moyenne des couleurs de pixels dans les bounding box avec une plage 
    # s'étendant en fonction de l'écart-type de ces valeurs
    # mean_hue=0
    # mean_sat=0
    # mean_val=0
    # variance_hue=0
    # variance_sat=0
    # variance_val=0
    # for i in range (0,len(liste_bb)):
    #     image_resize=image2[liste_bb[i][1]:liste_bb[i][3],liste_bb[i][0]:liste_bb[i][2]]
    #     image_resize_hsv=cv2.cvtColor(image_resize,cv2.COLOR_BGR2HSV)
    #     hue1, sat1, val1=image_resize_hsv[:,:,0], image_resize_hsv[:,:,1], image_resize_hsv[:,:,2]
    #     hue2, sat2, val2=[np.mean(hue1),np.std(hue1)], [np.mean(sat1),np.std(sat1)], [np.mean(val1),np.std(val1)]
    #     mean_hue+=hue2[0]
    #     mean_sat+=sat2[0]
    #     mean_val+=val2[0]
    #     variance_hue+=hue2[1]
    #     variance_sat+=sat2[1]
    #     variance_val+=val2[1]
        
    # mean_hue=mean_hue/(len(liste_bb))
    # mean_sat=mean_sat/(len(liste_bb))
    # mean_val=mean_val/(len(liste_bb))
    # variance_hue=variance_hue/(len(liste_bb))
    # variance_sat=variance_sat/(len(liste_bb))
    # variance_val=variance_val/(len(liste_bb))

    # lower_green=np.array([mean_hue-variance_hue,mean_sat-variance_sat,mean_val-variance_val])
    # upper_green=np.array([mean_hue+variance_hue,mean_sat+variance_sat,mean_val+variance_val])


    hsv_lavandin=cv2.cvtColor(image2,cv2.COLOR_BGR2HSV) #Changement d'espace de couleur
    lower_green=np.array([25,0,0])
    upper_green=np.array([84,255,255]) #Plage de valeur HSV pour le masque à appliquer
    mask=cv2.inRange(hsv_lavandin,lower_green,upper_green) #Création masque
    green=cv2.bitwise_and(image2, image2, mask=mask)  #Application du masque sur l'image2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erosion = cv2.erode(green, kernel, iterations = 1) #Denoising
    bgr_erosion=cv2.cvtColor(erosion, cv2.COLOR_HSV2BGR) #Passage dans l'espace RGB (BGR)

    #Coloration des pixels verts
    green_pixels=np.where((bgr_erosion!=[0,0,0]).all(axis=2)) #Coordonnées de pixels verts dans l'image filtrée
    green_0=list(green_pixels[0])
    green_1=list(green_pixels[1])

    #Suppression des coordonnées de points contenus dans les rectangle de bounding box
    for i in range (0,len(liste_bb)):
        index=np.where((green_pixels[0]>liste_bb[i][1]) & (green_pixels[0]<liste_bb[i][3]) & (green_pixels[1]>liste_bb[i][0]) & (green_pixels[1]<liste_bb[i][2]))
        green_0=np.delete(green_0,index[0])
        green_1=np.delete(green_1,index[0])
        green_pixels=tuple([np.array(green_0),np.array(green_1)])
    
    image2[green_pixels]=[255,0,0] #Coloration des pixels en bleu


    #Ecriture des différentes images dans les dossiers correspondants
    os.chdir(folder_new_image_regression)
    cv2.imwrite(file,image)
    os.chdir(folder_new_image_conversion)
    cv2.imwrite(file,image2)
    os.chdir(folder_new_image_masque)
    cv2.imwrite(file, erosion)

    #Affichage des différentes images
    cv2.namedWindow("Test"+file,cv2.WINDOW_NORMAL)
    cv2.imshow("Test"+file,image)
    cv2.namedWindow("HSV"+file,cv2.WINDOW_NORMAL)
    cv2.imshow("HSV"+file,erosion)
    cv2.namedWindow("Affichage"+file,cv2.WINDOW_NORMAL)
    cv2.imshow("Affichage"+file,image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()