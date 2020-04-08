import numpy
import scipy
from scipy import ndimage
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random

####
# UNIVERSITY OF VAASA
# Author: Kim Lehtinen
# Project: Evolutionary Computing, Sobel + ACO image edge detection
# 
# Note: 
# This software is partially based on the processes described in article written
# by Tian, J. , Y. Weiyu & X. Shengli (2008) "An ant colony optimization algorithm for image edge detection"
# which can be found here https://ieeexplore.ieee.org/document/4630880?arnumber=4630880
# or for univaasa students/teachers https://ieeexplore-ieee-org.proxy.uwasa.fi/document/4630880.
# In the ACO part of this software, same processes are used as described in Tian et al (2008) paper: 
# initialization process, construction process, update process and decision process.
# In addition, the idea of neighborhood also origins from Tian et al (2008) paper.
#
# This software is implemented by Kim Lehtinen, 
# but the concepts and ideas mentioned earlier belongs to Tian et al (2008).
####

####
# VAASAN YLIOPISTO
# Tekijä: Kim Lehtinen
# Harjoitustyö: Evolutionary Computing, Sobel + ACO reunantunnistus
#
# Huom: 
# Tämä ohjelmisto perustuu osittain artikkelissa
# Tian, J. , Y. Weiyu & X. Shengli (2008) "An ant colony optimization algorithm for image edge detection"
# kuvattuihin prosesseihin joka löytyy täältä https://ieeexplore.ieee.org/document/4630880?arnumber=4630880
# tai täältä jos opiskelet/työskentelet Vaasan yliopistossa https://ieeexplore-ieee-org.proxy.uwasa.fi/document/4630880.
# Tämä ohjelmisto käyttää ACO osassa samat mainitut prosessit kuten artikkelissa: 
# alustamisprosessi, rakennusprosessi, päivitysprosessi ja päätösprosessi.
# Sen lisäksi, idea naapuripikseleistä on käytetty Tian et al (2008) artikkelista.
#
# Tämän ohjelmiston on tehnyt Kim Lehtinen, 
# mutta aiemmin mainityt käsitteet ja ideat kuuluvat Tian (2008) ym.
####

########## SOBEL ##########
## Python Sobel koodiesimerkki lainettu stackoverflow (cgohlke Aug 25 2011) 
### https://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
### Bikesgray.jpg image Wikipediasta 
### https://en.wikipedia.org/wiki/Sobel_operator#/media/File:Bikesgray.jpg
kuva = imageio.imread('Bikesgray.jpg') # luetaan kuva muistiin
kuva = kuva.astype('int32')
dx = ndimage.sobel(kuva, 0)  # vaakasuora suodatin
dy = ndimage.sobel(kuva, 1)  # pystysuora suodatin
mag = numpy.hypot(dx, dy)  # magnituudi
mag *= 255.0 / numpy.max(mag)  # normalisoidaan
## sobel lainaus loppuu

## sobel kynnys
kuva = mag
kuva_koko = kuva.shape
rivit = kuva_koko[0]
sarakkeet = kuva_koko[1]
sobel_kynnys_tulos = np.argwhere(kuva > 50)
print(sobel_kynnys_tulos)
implot = plt.imshow(np.zeros(kuva_koko))
i = 0
print(len(kuva))
print(len(sobel_kynnys_tulos))
tmp = []
for i in range(0,len(sobel_kynnys_tulos)):
    x = sobel_kynnys_tulos[i][1]
    y = sobel_kynnys_tulos[i][0]
    if (i+1)%15==0:
        plt.scatter(x,y,c='r', s=0.1)
        tmp.append(sobel_kynnys_tulos[i])

# käytetään vain muutama piste sobelista
sobel_kynnys_tulos = tmp
print(len(tmp))
plt.show()

## end plot sobel edge points


########## ACO ##########
# VAIHE 1: Alustamisprosessi [Tian et al (2008) alkuperäinen idea]
# alustamisprosessi saadaan sobel_kynnys_tulos matriisista
# eli muurahaiset ovat jo jaettu kuvan päälle
feromoni_mat = np.ones(kuva_koko) # feromoni matriisi
muurahaisia_tot = len(sobel_kynnys_tulos) # muurahaisia yhteensä
rakennus_tot = 40 # kuinka monta kertaa suoritetaan rakennusprosessi

# luodaan muisti, ja tallennetaan alku positio muurahaiselle
muisti = {}
muurahainen_nykyinen_positio = {}
for m in range(muurahaisia_tot):
    muisti[m] = {}
    muurahainen_nykyinen_positio[m] = sobel_kynnys_tulos[m] # muista alku positio Xo, Yo
    for r in range(rakennus_tot):
        muisti[m][r] = []

# VAIHE 2: Rakennusprosessi [Tian et al (2008) alkuperäinen idea]
for rakennus_idx in range(rakennus_tot):
    #print("RAKENNUS KIERROS:", str(rakennus_idx))
    # siirretään yksi muurahainen kerrallaan
    for muurahainen_idx in range(muurahaisia_tot):
        nykyinen_positio = muurahainen_nykyinen_positio[muurahainen_idx] # haetaan nykyinen positio
        x = nykyinen_positio[0] # x-koordinaatti
        y = nykyinen_positio[1] # y-koordinaatti

        # naapuristo, 8-liitettävyysmalli, [Tian et al (2008) alkuperäinen idea]
        naapuristo = []
        if x > 0 and y > 0:
            naapuristo.append([x - 1, y - 1])
        if x > 0:
            naapuristo.append([x - 1, y])
        if x > 0 and (y+1) <= (sarakkeet-1):
            naapuristo.append([x - 1, y + 1])
        if y > 0:
            naapuristo.append([x, y - 1])
        if (y+1) <= (sarakkeet-1):
            naapuristo.append([x, y + 1])
        if (x+1) <= (rivit-1) and y > 0:
            naapuristo.append([x + 1, y - 1])
        if (x+1) <= (rivit-1) and y > 0:
            naapuristo.append([x + 1, y])
        if (x+1) <= (rivit-1) and (y+1) <= (sarakkeet-1):
            naapuristo.append([x + 1, y + 1])

        # valitaan uusi positio naapuristosta
        uusi_positio = [0, 0]
        paras_naapuri_arvo = 1
        # tutkitaan jokainen naapuri
        for naapuri_idx in range(len(naapuristo)):
            naapuri_positio = naapuristo[naapuri_idx] # naapurin koordinatit
            naapuri_x = naapuri_positio[0] # naapuri x-koordinaatti
            naapuri_y = naapuri_positio[1] # naapuri y-koordinaatti
            naapurin_nykyinen_feromoni_arvo = feromoni_mat[naapuri_x, naapuri_y] # naapurin feromoni arvo

            # katso onko tämä naapuri muistissa
            muistissa = False
            for r_idx in range(rakennus_tot):
                muisti_paikka = muisti[muurahainen_idx][r_idx]
                if len(muisti_paikka):
                    muisti_x = muisti[muurahainen_idx][r_idx][0]
                    muisti_y = muisti[muurahainen_idx][r_idx][1]
                    if naapuri_x == muisti_x and naapuri_y == muisti_y:
                        muistissa = True

            # annetaan vain uusi positio jos valittu ei ole muistissa
            # tälle muurahaiselle
            if not muistissa:
                # anna uusi positio muurahaiselle jos arvo on sama tai parempi
                if naapurin_nykyinen_feromoni_arvo >= paras_naapuri_arvo:
                    numerot = list(range(10))
                    random_numero = random.choice(numerot)
                    # lisätään satunnaisuus niin että paras ei aina valita
                    if random_numero > 5:
                        paras_naapuri_arvo = naapurin_nykyinen_feromoni_arvo
                        uusi_positio = naapuri_positio

        # lisätään feromoni + 0.1 koska täällä on käyty
        feromoni_nyt = feromoni_mat[uusi_positio[0], uusi_positio[1]]
        feromoni_nyt += 0.1

        # lisätään 0.2 feromonia jos rgb arvo > 100 ja < 150 (vähän ruokaa)
        if kuva[uusi_positio[0], uusi_positio[1]] > 100 and kuva[uusi_positio[0], uusi_positio[1]] < 150:
            feromoni_nyt += 0.2

        # lisätään 0.5 feromonia jos rgb arvo > 150 (paljon ruokaa)
        if kuva[uusi_positio[0], uusi_positio[1]] > 150:
            feromoni_nyt += 0.5
        
        # VAIHE 3: Päivitysprosessi [Tian et al (2008) alkuperäinen idea]
        feromoni_mat[uusi_positio[0], uusi_positio[1]] = feromoni_nyt

        # tallennetaan uusi positio muistiin [Tian et al (2008) alkuperäinen idea]
        muisti[muurahainen_idx][rakennus_idx] = uusi_positio

        # tallennetaan nykyinen positio, niin tiedetään missä muurahainen on tällä hetkellä
        muurahainen_nykyinen_positio[muurahainen_idx] = uusi_positio
# rakennusprosessi loppuu

# VAIHE 4: Päätösprosessi [Tian et al (2008) alkuperäinen idea]
piirros = plt.imshow(np.zeros(kuva_koko))
for r in range(rivit):
    for s in range(sarakkeet):
        feromoni_arvo = feromoni_mat[r][s]
        # kynnystesti
        if feromoni_arvo >= 1.5:
            plt.scatter(s,r,c='r', s=0.1)
plt.show()