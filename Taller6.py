# ----------------------------------------------------------------------------------------
# PROGRAMA: <<Automatic Stitching>>
# ----------------------------------------------------------------------------------------
# Descripción: <<Este es un programa que relaciona puntos de interés entre imagenes y reconstruye una sola>>
# ----------------------------------------------------------------------------------------
# Autores:
''' 
# Miguel David Benavides Galindo            md_benavidesg@javeriana.edu.co
# Christian Fernando Rodriguez Rodriguez    rodriguezchristianf@javeriana.edu.co
'''
# Version: 1.0
# [19.10.2021]
# ----------------------------------------------------------------------------------------
# IMPORTAR MODULES
import cv2 
import os 
import glob 
import numpy as np
# ----------------------------------------------------------------------------------------
# PATH EN EL QUE SE ENCUENTRAN LAS IMAGENES
# ----------------------------------------------------------------------------------------
img_dir = "C:/Users/User/Documents/Trabajo/Maestria/2021_I/Imagenes y Video/Taller6/Imagen"
# ----------------------------------------------------------------------------------------

# LISTADO DE IMAGENES
# ----------------------------------------------------------------------------------------
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path)        # Listado de Imagenes disponibles
# ----------------------------------------------------------------------------------------
# ORDENAR IMAGENES POR FECHA
# ----------------------------------------------------------------------------------------
fail_count = 0
success_count = 0

for img in files:
 	with open(img, "rb") as file:
		tags = exifread.process_file(file, details=False, stop_tag="DateTimeOriginal")
		try:
 			date_path = str(tags["DateTimeOriginal"])[:10].replace(":", ".")
 			success_count += 1
		except:
 			print(str(img) + " does not have tags.")
 			fail_count += 1
 			date_path = "0000"
		if not os.path.exists(img_dir + date_path):
 			os.mkdir(img_dir + date_path)
 	shutil.move(img, img_dir + date_path + "\\" + img[-12:])
# ----------------------------------------------------------------------------------------
# CARGUE MASIVO DE IMAGENES
# ----------------------------------------------------------------------------------------
data = []                       # imagen original
data_gray = []                  # imagen en grises
data_draw = []                  # copia original

for ima in files: 
    img = cv2.imread(ima)  
    img = cv2.resize(img, (360, 360))               # cambio del tamaño de la imagen
    data.append(img) 
    
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # transformación a grises
    data_gray.append(image_gray)
    
    image_draw = np.copy(img)                   # copia
    data_draw.append(image_draw)
# ----------------------------------------------------------------------------------------
# DESCRIPCIÓN DEL CONTENIDO
# ----------------------------------------------------------------------------------------
##### Se muesta información de la base y se solicita la imagen de referencia para luego validar el valor
print("El número de imagenes en la ruta ingresada son:",len(data))
#%%
# ----------------------------------------------------------------------------------------
# SELECION ENTRE LOS METODOS SIFT Y ORB 
# ----------------------------------------------------------------------------------------
print("SIFT method press: 1")
print("ORB method press: 2")
print(" ")
method = int(input())    
# ----------------------------------------------------------------------------------------
# EJECUCIÓN DEL MÉTODO
# ----------------------------------------------------------------------------------------
Transformacion = []
for i in range(len(data)-1):
    imagen1 = data_gray[i]
    imagen2 = data_gray[i+1]
    
    if method == 1:                                         # Metodo SIFT
        sift = cv2.SIFT_create(nfeatures=200)   # shift invariant feature transform
        keypoints_1, descriptors_1 = sift.detectAndCompute(imagen1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(imagen2, None)
    else:
        orb = cv2.ORB_create(nfeatures=200)     # oriented FAST and Rotated BRIEF
        keypoints_1, descriptors_1 = orb.detectAndCompute(imagen1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(imagen2, None)
    
    image_draw_1 = cv2.drawKeypoints(imagen1, keypoints_1, None)
    image_draw_2 = cv2.drawKeypoints(imagen2, keypoints_2, None)
    # ----------------------------------------------------------------------------------------
    # Descripcion: << Se emplean dos métodos diferentes para la generación de puntos de interés >>
    # ----------------------------------------------------------------------------------------
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)                # Interest points matching
    image_matching = cv2.drawMatchesKnn(imagen1, keypoints_1, imagen2, keypoints_2, matches, None)
    # ----------------------------------------------------------------------------------------
    # PUNTOS DE INTERÉS
    # ----------------------------------------------------------------------------------------
    points_1 = []
    points_2 = []
    for idx, match in enumerate(matches):
        idx2 = match[0].trainIdx                     # Retrieve matched points
        points_1.append(keypoints_1[idx].pt)
        points_2.append(keypoints_2[idx2].pt)
    # ----------------------------------------------------------------------------------------
    # HOMOGRAFIA
    # ----------------------------------------------------------------------------------------
    H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)            # Compute homography and warp image_1
    Transformacion.append(H)
# ----------------------------------------------------------------------------------------

Trans_ima = []
ref = 1
for i in range(len(data)):
    TFF = np.identity(3)
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIONES A IZQUIERDA DEL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------   
    if i < ref:
        j = i
        while j<ref:
            TF = Transformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIONES A DERECHA DEL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------  
    elif i > ref:
        j = ref
        while j<i:
            TF = Transformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1   
        TFF = np.linalg.inv(TFF)
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIÓN HACÍA EL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------  
    image_warped = cv2.warpPerspective(data[i], TFF, (data[i].shape[1], data[i].shape[0]))
    Trans_ima.append(image_warped)
# ----------------------------------------------------------------------------------------
# VISUALIZACIÓN DE LAS PERSPECTIVAS
# ----------------------------------------------------------------------------------------
cv2.imshow("warpPerspective1", Trans_ima[0])
cv2.imshow("warpPerspective2", Trans_ima[1])
# ----------------------------------------------------------------------------------------
# IMAGEN RESULTADO PROMEDIO
avg_img = np.mean(Trans_ima, axis=0)
avg_img = avg_img.astype(np.uint8)
# ----------------------------------------------------------------------------------------
# VISUALIZACIÓN DEL RESULTADO STITCHING 
# ----------------------------------------------------------------------------------------
cv2.imshow("warpPerspective Result", avg_img)
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# EXPORTAR RESULTADO STITCHING EN IMAGEN FORMATO JPG
# ----------------------------------------------------------------------------------------
cv2.imwrite("warpPerspectiveResult.PNG", avg_img)   # Save stitching
#%%
# ----------------------------------------------------------------------------------------
# SELECCION DE LA IMAGEN REFERENCIA POR EL USUARIO
# ----------------------------------------------------------------------------------------
print("¿Cual de las",len(data),"imagenes, se usará como referencia?")
ref = int(input())-1
# ----------------------------------------------------------------------------------------
# VALIDACIÓN DE LA REFERENCIA
# ----------------------------------------------------------------------------------------
if ref in range(len(data)):
    None    
else:
    print(ref, "No es valido, por favor introducir un número entre 1 y", len(data))

#%%
# ----------------------------------------------------------------------------------------
# VISUALIZACIÓN ENTRE PARES DE IMAGENES Y PUNTOS MANUALES
# ----------------------------------------------------------------------------------------
# Descripcion: << Se concatenan las imagenes para poder ingresar los puntos del usuario, intercalados uno en la imagen >>
# ----------------------------------------------------------------------------------------
Tranformacion = []
for i in range(len(data)-1):
    imagen1 = data[i]
    imagen2 = data[i+1]
    image = cv2.hconcat([imagen1, imagen2])             # Merge entre pares de imagenes
  # ----------------------------------------------------------------------------------------
  # INGRESO DE PUNTOS DE REFERENCIA SEGUN USUARIO
  # ----------------------------------------------------------------------------------------  
    points = []
    def click(event, x, y, flags, param):               # Función para procesar el input(click) del usuario
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
    
    image_draw = np.copy(image)
  # ----------------------------------------------------------------------------------------
  # INPUT INTERACTIVO PARA EL USUARIO
  # ----------------------------------------------------------------------------------------   
    points1 = []
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)                # Input del usuario
    
    point_counter = 0
    c=0
  # ----------------------------------------------------------------------------------------
  # ITERACIÓN PARA VALIDAR PARES DE PUNTOS DE REFERENCIA
  # ---------------------------------------------------------------------------------------- 
    while True:
        if c%2 == 0:
            color = [0, 0, 255]                         # Color azul
        else:
            color = [255, 0, 0]                         # Color rojo
  # ----------------------------------------------------------------------------------------
  # VISUALIZADOR DE LAS IMAGENES PARA EL USUARIO
  # ----------------------------------------------------------------------------------------              
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
  # ----------------------------------------------------------------------------------------
  # CRITERIO DE FINALIZACIÓN
  # ----------------------------------------------------------------------------------------         
        if key == ord("x"):                             # Regla de finalización para el ingreso de puntos de referencia
            points1 = points.copy()
            points = []
            break
        if len(points) > point_counter:
            point_counter = len(points)                 # Número de puntos de referencia ingresados
            cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, color, -1)
            c = c +1   
  # ----------------------------------------------------------------------------------------
  # PUNTOS DE REFERENCIA PARA LA HOMOGRAFIA
  # ----------------------------------------------------------------------------------------   
    points_1 = []
    points_2 = []
    for i in range(len(points1)):
        if i%2 == 0:
            points_1.append(points1[i])
        else:
            val = (points1[i][0]-image_draw.shape[0],points1[i][1])
            points_2.append(val)
  # ----------------------------------------------------------------------------------------
  # VALIDACIÓN DEL MÍNIMO NÚMERO DE PUNTOS DE REFERENCIA
  # ----------------------------------------------------------------------------------------   
    N = min(len(points_1), len(points_2))
    assert N >= 4, 'Se requieren minimo 4 punto por imagen'         # Validación del mínimo de puntos necesarios
    
    pts1 = np.array(points_1[:N])
    pts2 = np.array(points_2[:N])
  # ----------------------------------------------------------------------------------------
  # HOMOGRAFIAS ENTRE IMAGENES
  # ----------------------------------------------------------------------------------------   
    H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)        # Homografia entre puntos de las imagenes
    
    Tranformacion.append(H)
cv2.waitKey(0)
#### Se separan los puntos en dos para calculas su tranformacion    
#%% 
# ----------------------------------------------------------------------------------------
# HOMOGRAFIA DE IMAGENES A PARTIR DE PUNTOS DE REFERENCIA
# ----------------------------------------------------------------------------------------
# Descripcion: << Se calculan las tranformaciones a la imagen de referencia, ya sea por debajo o por encima >>
# ----------------------------------------------------------------------------------------
Trans_ima = []
for i in range(len(data)):
    TFF = np.identity(3)
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIONES A IZQUIERDA DEL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------   
    if i < ref:
        j = i
        while j<ref:
            TF = Tranformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIONES A DERECHA DEL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------  
    elif i > ref:
        j = ref
        while j<i:
            TF = Tranformacion[j]
            TFF = np.dot(TFF, TF)
            j += 1   
        TFF = np.linalg.inv(TFF)
  # ----------------------------------------------------------------------------------------
  # TRANSFORMACIÓN HACÍA EL PUNTO DE REFERENCIA
  # ----------------------------------------------------------------------------------------  
    image_warped = cv2.warpPerspective(data[i], TFF, (data[i].shape[1], data[i].shape[0]))
    Trans_ima.append(image_warped)
    
#%%
# ----------------------------------------------------------------------------------------
# STITCHING 
# ----------------------------------------------------------------------------------------
# Descripcion: << Combinación de las imagenes resultantes de una imagen resultante promedio >>
# ----------------------------------------------------------------------------------------
# List of images, all must be the same size and data type.
avg_img = np.mean(Trans_ima, axis=0)
avg_img = avg_img.astype(np.uint8)
# ----------------------------------------------------------------------------------------
# VISUALIZACIÓN DEL RESULTADO STITCHING 
# ----------------------------------------------------------------------------------------
cv2.imshow("Image warped", avg_img)        # Imagen resultado (STITCHING)
cv2.waitKey(0)
# ----------------------------------------------------------------------------------------
# EXPORTAR RESULTADO STITCHING EN IMAGEN FORMATO JPG
# ----------------------------------------------------------------------------------------
cv2.imwrite('image_warped.jpg', image_warped)   # Save stitching
#%%
|