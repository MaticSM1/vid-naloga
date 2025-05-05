import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

def izracunaj_centre(slika, izbira='naključno', dimenzija_centra=3, T=50, k=3):
    h, w, _ = slika.shape
    pixels = []

    if dimenzija_centra == 5:  # [R, G, B, h(0-1), w(0-1)]
        for i in range(h):
            for j in range(w):
                pixel = slika[i, j].tolist() + [i / h, j / w]
                pixels.append(pixel)
    else:                      # [R, G, B]
         for i in range(h):
            for j in range(w):
                pixel = slika[i, j].tolist() 
                pixels.append(pixel)

    if izbira == 'naključno':
        centri = []
        while len(centri) < k:
            kandidat = random.choice(pixels)
            if all(np.sum(np.abs(np.array(c) - np.array(kandidat))) > T for c in centri):
                centri.append(kandidat)
        return np.array(centri, dtype=np.float32)

    elif izbira == 'ročna':

        plt.imshow(slika)
        plt.title("Klikni")
        plt.axis('off')

        centri = []

        def onclick(event):
            if len(centri) < k:
                x, y = int(event.xdata), int(event.ydata)
                if dimenzija_centra == 5:
                    centri.append(slika[y, x].tolist() + [y / h, x / w])
                else:
                    centri.append(slika[y, x].tolist())
                if len(centri) == k:
                    plt.close()

        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        plt.gcf().canvas.mpl_disconnect(cid)

        return np.array(centri, dtype=np.float32)

