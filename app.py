import cv2
import uuid
import time


ruta = "Data/fotos_mias/"
n = 100

for i in range(n):
	cap = cv2.VideoCapture(0)
	leido, frame = cap.read()
	if leido == True:
		#nombre_foto = str(uuid.uuid4()) + ".png"
		nombre_foto = str(i+500) + ".png"
		cv2.imwrite(ruta+nombre_foto, frame)
		print("Foto tomada correctamente con el nombre {}".format(nombre_foto))
	else:
		print("Error al acceder a la cámara")

	cap.release()
	time.sleep(30)
