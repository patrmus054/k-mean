'''
a) Wczytać obraz 'landscape.jpg' (np. za pomocą biblioteki PIL) i potraktować każdy piksel jako pojedynczy punkt o 3 wymiarach.

b) Dokonać klasteryzacji metodą k-średnich i wybrać do ostatecznego podziału jako liczbę k najmniejszą z liczb,
 dla której wartość inercji jest mniejsza niż  1.4 \cdot 10^{8} .
 UWAGA! Grupowania dokonywać dla argumentu random_state=1, tzn.
 funkcja KMeans ma przyjąć jako wartość argumentu random_state liczbę 1, zaś początkowe miejsca centroidów mają zostać wybrane metodą 'k-means++'.
  Wszystkie inne rozwiązania będą odrzucane.

(Podpowiedź z uwagi na potencjalnie długi czas obliczeń: rozwiązanie ma nie więcej niż 20 kolorów, dokładną liczbę klastrów znaleźć samemu).

c) Każdą współrzędną wszystkich uzyskanych centroidów zaokrąglić do najbliższej liczby całkowitej.

c) Dla grupowania na tak ustalone k klastrów dokonać przypisania każdemu pikselowi jego zaokrąglonej wartości środka ciężkości.

d) Nowo utworzony obraz zapisać w pliku .png i dołączyć go do rozwiązania.

e) Moduł ma zostać stworzony tak, że główna funkcja sterująca przyjmie na starcie TYLKO zaimportowany wcześniej obraz 'landscape.jpg' (przekonwertowany wcześniej w module __main__ do tablicy numpy) oraz ma zwracać JEDYNIE trójwymiarową tablicę numpy z nowymi wartościami pikseli.

Kod ma zostać opatrzony komentarzami. Za rozwiązania oddane po terminie będą odejmowane punkty.
'''

#Klasycznie zaczynamy od dodania dependency
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

#główna funkcja
def main(np_im):

    pixels = []  #pusta tablica w której będą przetrzymywane pixele
    for i in np_im:
        for j in i:
            pixels.append(j)  #dzieki podwojnej pętli dostaniemy sie do kazdego pixela z obrazka

    #czas na wybranie odpowiedniej ilości kolorów tak by spełniały warunki brzegowe zadania

    k = 8
    limit = 1.4 * (10 ** 8)
    #mierzymy interia i jeśli, któreś przekroczy limit przerywamy działanie pętli
    while k < 21:
        inertias = []
        arguments = []
        clustering = KMeans(k, 'k-means++', 100, 1)
        clustering.fit(pixels)
        inertias.append(clustering.inertia_)
        arguments.append(k)
        if inertias[-1] < limit:
            break
        k += 1

    #Gdy znamy juz odpowienia licze k przystepujmy do trenowania modelu
    km = KMeans(k, 1, 'k-means++')
    x_fit = km.fit(pixels)
    labels = x_fit.predict(pixels)
    centers = x_fit.cluster_centers_
    # spełniamy podpunkt c) zadania
    centers = np.around(centers).astype(np.uint8)
    res = centers[labels]
    #operacje na pixelach by uformowac obraz
    img_format = np.reshape(res, (480, 640, 3))
    img_format = np.around(img_format).astype(np.uint8)

    return img_format

if __name__ == "__main__":
    #Korzystam z polecanej biblioteki PIL do wczytania obrazu
  img = Image.open('landscape.jpg')
  temp = np.array(img)
  temp2 = main(temp)
  # formowanie nowego zdjecia
  output = Image.fromarray(temp2)
  output.save("test.png")