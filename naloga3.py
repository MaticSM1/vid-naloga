NV-13-prikaz-originalneimport cv2 as cv
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

def manhattan(a, b):
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



def meanshift(slika, velikost_okna, dimenzija=3, max_ponovitve=5, min_cd=0.1, vzorec_stevilo=500):
    h_, w_, _ = slika.shape
    podatki = []

    for i in range(h_):
        for j in range(w_):
            if dimenzija == 5:
                podatki.append(np.append(slika[i, j], [i / h_, j / w_]))
            else:
                podatki.append(slika[i, j])

    podatki = np.array(podatki, dtype=np.float32)

    # Naključno izbrani centri
    vzorec_idx = np.random.choice(len(podatki), size=min(vzorec_stevilo, len(podatki)), replace=False)
    vzorec = podatki[vzorec_idx]
    konvergirane = []

    for xi in vzorec:
        for _ in range(max_ponovitve):
            # kvadrat razdalj za vse točke
            razdalje = np.sum((podatki - xi) ** 2, axis=1)
            # gaussovo jedro
            utezi = np.exp(-razdalje / (2 * velikost_okna ** 2))
            # premik točke
            nova_x = np.sum(podatki * utezi[:, np.newaxis], axis=0) / np.sum(utezi)
            # konvergenca
            if np.sum((nova_x - xi) ** 2) < 1e-6:
                break
            xi = nova_x

        # v stare centre
        dodaj = True
        for c in konvergirane:
            if np.sum((c - xi) ** 2) < min_cd ** 2:
                dodaj = False
                break
        if dodaj:
            konvergirane.append(xi)

    # najbližji center
    rezultat = np.zeros((h_ * w_, 3), dtype=np.uint8)
    for i in range(len(podatki)):
        px = podatki[i]
        razdalje = [np.sum((px - c) ** 2) for c in konvergirane]
        najblizji = konvergirane[np.argmin(razdalje)]
        rezultat[i] = najblizji[:3]

    return rezultat.reshape((h_, w_, 3))


if __name__ == "__main__":
    slika = cv.imread("slika.jpg")
    slika = cv.cvtColor(slika, cv.COLOR_BGR2RGB)
    slika = cv.resize(slika, (200, 200))  # da gre hitreje

    segment_kmeans = kmeans(slika, k=3, iteracije=1, dimenzija_centra=3, T=30)
    segment_meanshift = meanshift(slika, velikost_okna=0.2, dimenzija=3, max_ponovitve=5, min_cd=0.1, vzorec_stevilo=500)

    # Prikaz
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("org")
    plt.imshow(slika)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("kmeans")
    plt.imshow(segment_kmeans)
    plt.axis('off')

  