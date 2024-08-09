import requests
import base64
import json
from rembg import remove
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageEnhance, ImageFilter
import datetime
import time
import sys
import os
import cv2
import numpy as np





def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


timestamp = str(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')).replace(" ","-")
print(timestamp)

newpath = "results/" + timestamp + "/";


if(str(sys.argv[1])=="-bgrem"):
    os.mkdir(newpath)
    os.makedirs(newpath + "tmp/outputs") 
    os.makedirs(newpath + "tmp/nobg") 
    os.makedirs(newpath + "results/bgremoval")

if(str(sys.argv[1])=="-nobgrem"):
    os.mkdir(newpath)
    os.makedirs(newpath + "results/nobgremoval") 




n = len(sys.argv)
if(n==0 or (str(sys.argv[1])!="-nobgrem" and str(sys.argv[1])!="-bgrem")):
    print("Errore, nessun parametro o parametro sbagliato")
    print("Istruzioni:")
    print("-nobgrem -> Output img2img level5")
    print("-bgrem -> Output img2img level1 + copypaste background modello level0\n")
    exit()


# Define the URL and the payload to send.
url = "http://0.0.0.0:7861"

#moltiplicatore risoluzione
imgmultiplier = 2

#numero immagini
nimmagini = 9


starttime = time.time()

payloadglobal = {}

for i in range(nimmagini):
    print("###############################################")
    print("Lavorando all'immagine " + str(i))


    #Scelgo il livello 0 senza sfondo con gli omini primitivi
    if(str(sys.argv[1])=="-bgrem"): 
        init_images = [
            encode_file_to_base64("images/Mall_synt_reduced/Level0/map" + str(i) +".png")
        ]
    #Scelgo il livello 5 con lo sfondo con gli omini modellati e vestiti
    elif(str(sys.argv[1])=="-nobgrem"):
        init_images = [
            encode_file_to_base64("images/Mall_synt_reduced/Level5/map" + str(i) +".png")
        ]


    payload = {
      #"prompt": "people walking, top angle, cctv still frame",
      'prompt':'A CCTV still frame from a top-down angle showing an urban street with numerous people walking in different directions. The pedestrians vary in age, clothing, and accessories. The sidewalk is crowded, with street vendors and parked bicycles visible. Shadows of the people and buildings are cast on the pavement.',
      'negative_prompt': 'Cartoonish characters, unrealistic proportions, exaggerated features, bright and unrealistic colors, fantasy elements, non-human characters, artificial lighting, surreal backgrounds, comic-style outlines, animated effects, overly clean or glossy surfaces, lack of shadows, floating objects, distorted perspectives',
      'width': 640 * imgmultiplier,
      'height': 480 * imgmultiplier,
      'sampler_index': 'Euler',
      #'steps': 80,
      'steps':40,
      'cfg_scale': 7,
      'denoising_strength': 0.35,
      'init_images': init_images
    }

    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
    #response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    #Rimuovo lo sfondo, incollo il png sopra lo sfondo e salvo
    if(str(sys.argv[1])=="-bgrem"): 
        # Decode and save the image.

        with open(newpath + "tmp/outputs/output" + str(i) +".png", 'wb') as f:
            f.write(base64.b64decode(r['images'][0]))


        with open(newpath + "tmp/outputs/response" + str(i) +".json", "w") as file:
            del r['images']
            del r['parameters']
            payloadglobal = str(r['info'])
            file.write(str(r['info']))

        #RIMOZIONE SFONDO
 
        input = Image.open(newpath + "tmp/outputs/output" + str(i) +".png")

        inputnormale = np.array(input)
        inputnormale = inputnormale[:, :, ::-1].copy()

        input_ench = input
        input_ench = ImageEnhance.Contrast(input_ench).enhance(6)
        input_ench = ImageEnhance.Color(input_ench).enhance(0)


        input_ench.save(newpath + "tmp/nobg/outcontrasty" + str(i) +".png")


        low_gray = np.array([0, 0, 50])
        high_gray = np.array([255, 255, 255])


        img = np.array(input_ench)
        img = img[:, :, ::-1].copy()

        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
        inputnormale = cv2.resize(inputnormale, (640, 480), interpolation=cv2.INTER_CUBIC)

        mask = cv2.inRange(img, low_gray, high_gray)

        mask = cv2.blur(mask,(3,3))
        #mask = cv2.GaussianBlur(mask,(5,5),1000)


        mask = 255-mask
        res = cv2.bitwise_and( inputnormale, inputnormale, mask=mask)

        res = cv2.cvtColor(res, cv2.COLOR_BGR2BGRA)
        res[:,:,3] = mask 

        res = Image.fromarray(res)

        res = res.resize((640, 480), Image.LANCZOS)
        res = np.array(res)

        cv2.imwrite(newpath + "tmp/nobg/outputnobg" + str(i) +".png",res)


        img1 = Image.open("background.png").convert('RGBA')
        img2 = Image.open(newpath + "tmp/nobg/outputnobg" + str(i) +".png").convert('RGBA')
        #img2 = remove(img2,alpha_matting=True,
        #alpha_matting_foreground_threshold=240,
        #alpha_matting_background_threshold=10,
        #alpha_matting_erode_structure_size=10,
        #alpha_matting_base_size=1000)

        #img1.paste(img2, (0,0), img2) 
        Image.alpha_composite(img1, img2).save(newpath + "results/bgremoval/outputbgadded" + str(i) +".png")
        print("Salvata in " + newpath + "results/bgremoval/outputbgadded" + str(i) +".png")



    #Salvo l'immagine
    elif(str(sys.argv[1])=="-nobgrem"):
        with open(newpath + "results/nobgremoval/output" + str(i) +".png", 'wb') as f:
            f.write(base64.b64decode(r['images'][0]))

        with open(newpath + "results/nobgremoval/response" + str(i) +".json", "w") as file:
            del r['images']
            del r['parameters']
            payloadglobal = str(r['info'])
            file.write(str(r['info']))


        print("Salvata in " + newpath + "results/nobgremoval/output" + str(i) +".png")


endtime = time.time()
elapsedtime = round((endtime - starttime),2)

print("Tempo passato: " + str(elapsedtime) + " secondi")

with open(newpath + "endreport.txt", "w") as file:
    file.write(str(payloadglobal) + "\n" + "######################" + "\n" + 
        "\nt sec: " + str(elapsedtime) +
        "\nt min: " + str(elapsedtime/60) +
        "\nt hours: " + str((elapsedtime/60)/60)
        )
