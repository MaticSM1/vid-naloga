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

ef manhattan(a, b):
    return np.sum(np.abs(a - b)) #D=∣x1​−x2​∣+∣y1​−y2​∣

def kmeans(slika, k=3, iteracije=10, dimenzija_centra=3, T=50):
    h, w, _ = slika.shape
    slika_data = []

    for i in range(h):
        for j in range(w):
            if dimenzija_centra == 5:
                slika_data.append(np.append(slika[i, j], [i / h, j / w]))
            else:
                slika_data.append(slika[i, j])

    slika_data = np.array(slika_data, dtype=np.float32)

    centri = izracunaj_centre(slika, 'naključno', dimenzija_centra, T, k)
    oznake = np.zeros(len(slika_data), dtype=int)

    for interacija in range(iteracije):
        for i in range(len(slika_data)):
            min_razdalja = float('inf')
            naj_center = 0
            for j in range(k):
                razdalja = np.sum(np.abs(slika_data[i] - centri[j]))
                if razdalja < min_razdalja:
                    min_razdalja = razdalja
                    naj_center = j
            oznake[i] = naj_center

        for j in range(k):
            pripadajoči = [slika_data[i] for i in range(len(slika_data)) if oznake[i] == j]
            if pripadajoči:
                centri[j] = np.mean(pripadajoči, axis=0)

    nova_slika = np.zeros((h * w, 3), dtype=np.uint8)
    for i in range(len(slika_data)):
        nova_slika[i] = centri[oznake[i]][:3]

    return nova_slika.reshape((h, w, 3))

