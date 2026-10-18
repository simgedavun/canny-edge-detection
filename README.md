Canny kenar algılama algoritması nesnelerin kenarlarını bulabilmek için kullanılan bir çok aşamalı standart algoritmadır. Pikseller arası ani parlaklık ya da renk değişikliklerini yakalayarak görselin yapısal hatlarını ortaya koymaktadır.


Gürültü Bastırma (5x5 Gauss Filtresi) amacı rastgele olan parlaklık dalgalanmalarını temizlemek ve görüntüyü hafifçe pürüzsüzleştirmektir.

Sobel Filtre ile gradyan hesaplama amacı pikseller arasındaki parlaklığın ne kadar şiddetle nerede ve hangi yöne doğru değiştiğini sayısal olarak ölçmektedir.

Non-maximum suppression amacı Sobel adımında kalın ve bulanık bantlar halinde çıkan kenarları incelterek tam olarak 1 piksel genişliğinde keskin, tekil hatlara dönüştürmektir.

Hysteresis Thresholding amacı kenar sürekliliğini korurken sahte kenarları ve gürültü artıklarını temizlemektir. Bu aşama tek bir eşik değeri yerine biri yüksek, diğeri düşük iki ayrı eşik değeri kullanarak kenarları akıllı bir bağlantı mantığıyla filtrelemektedir.