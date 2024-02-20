# Laporan Proyek _Machine Learning_ - Moch. Avin

## Domain Proyek

Biji labu sering dikonsumsi sebagai camilan di seluruh dunia karena kandungan protein, lemak, karbohidrat, dan mineral. Biji labu dibagi menjadi banyak jenis, dan dua jenis yang paling penting dan berkualitas adalah _Ürgüp Sivrisi_ dan _Çerçevelik_, yang ditanam di Turki [1].

Salah satu masalah terbesar di sektor benih bukanlah menghilangkan benda asing di dalam benih, melainkan membedakan varietas yang berbeda dari spesies yang sama dari benih. Selama ini tidak ada metode atau mesin otomatis yang dapat membedakannya. Dengan _machine learning_, akan memungkinkan untuk mengenali varietas dan membedakan varietas yang berbeda yang tercampur dalam benih. Dengan cara ini, pengenalan varietas biji labu akan dilakukan dengan cepat dan akurat dengan _machine learning_ [1].

Masalah yang akan diangkat adalah mengklasifikasikan dua jenis biji labu, _Ürgüp Sivrisi_ dan _Çerçevelik_, berdasarkan ciri-ciri morfologi yang diekstrak dari gambar. Masalah ini penting untuk diselesaikan karena dapat membantu meningkatkan hasil panen dan mengontrol kualitas bibit unggul [1].

_Machine learning_ dapat membedakan varietas bibit biji labu dengan akurat dan efisien. Ini membantu untuk mengontrol kemurnian varietas bibit. Dengan analisis fitur morfologis, model _machine learning_ dapat mengklasifikasikan spesies bibit yang unggul sehingga menjaga kemurnian varietas bibit [2].

> Masalah ini harus diselasaikan agar dapat meningkatkan efisiensi dan efektivitas dalam rangka meningkatkan kemurnian varietas, meningkatkan hasil panen, dan menjaga kualitas labu. Masalah ini dapat diselesaikan dengan cara membangun model _Machine Learning_ berkaitan kasus klasifikasi, dengan targetnya adalah kategori benihnya dan prediktornya seperti luas, soliditas, rasio aspek dsb [2].

> Dengan membangun model _Machine Learning_ untuk klasifikasi varietas benih labu berdasarkan atribut fisik seperti luas, soliditas, rasio aspek, dan lainnya, mampu mencapai tujuan-tujuan penting dalam pertanian. Pertama, model ini membantu meningkatkan kemurnian varietas dengan kemampuannya yang secara akurat dapat membedakan kedua varietas labu, mengurangi risiko pencemaran silang yang dapat merugikan hasil panen. Kedua, dengan menggunakan model ini, petani dapat mengidentifikasi varietas yang paling cocok dengan kondisi lingkungan, yang pada akhirnya dapat meningkatkan hasil panen secara keseluruhan. Petani dapat memilih benih yang memiliki potensi tertinggi untuk menghasilkan labu berkualitas tinggi [1].

## Business Understanding

### 1. Problem Statements

- Terdapat kebutuhan untuk mengklasifikasi biji labu antara varietas _Ürgüp Sivrisi_ dan _Çerçevelik_, yang memiliki kemiripan fisik yang tinggi namun perbedaan genetik yang signifikan.
- Tidak adanya sistem yang efisien dan akurat dalam mengklasifikasi biji labu antara _Ürgüp Sivrisi_ dan _Çerçevelik_ secara otomatis dapat menghambat proses pengelolaan dan pengembangan varietas tanaman labu yang optimal.
- Petani kesulitan dalam mengidentifikasi varietas, sehingga sulit untuk memilah biji labu yang cocok dengan kondisi lingkungan.

### 2. Goals

- Mengembangkan model klasifikasi yang mampu membedakan antara biji labu varietas _Ürgüp Sivrisi_ dan _Çerçevelik_ dengan tingkat akurasi yang tinggi, sehingga dapat mendukung para peneliti dalam memilih dan mengelola benih dengan tepat.
- Menciptakan solusi yang efektif dalam mengidentifikasi varietas biji labu secara cepat dan akurat, sehingga dapat meningkatkan efisiensi dan produktivitas dalam pengelolaan sumber daya tanaman serta membantu para peneliti dalam mengembangkan varietas yang lebih unggul dan adaptif.
- Petani dapat dengan mudah memilah dan mengidentifikasi varietas biji labu sehingga petani dapat memilih benih yang memiliki potensi tertinggi untuk menghasilkan labu berkualitas tinggi.

  ### Solution statements

  Terdapat 3 solution statement. Pertama, akan dicoba membangun model **K-Nearest Neighbor (KNN)** dengan mengatur parameter yang ada. Kedua, akan dicoba membangun model **Random Forest Classifier** dengan mengatur berbagai parameter yang ada. Ketiga, akan dibangun model **AdaBoosting** dengan mengatur berbagai parameter yang ada. Ketiga solusi ini akan diukur dengan metrik yang sama yaitu _f1 score_. _F1 score_ digunakan karena memberikan gambaran yang seimbang antara _presisi_ dan _recall_ dalam satu nilai, memudahkan evaluasi kinerja model klasifikasi secara komprehensif. Nilai _f1 score_ dapat dikatakan berhasil ketika angkanya di atas 0.85 atau 85%.

## Data Understanding

Dataset yang digunakan awalnya terdiri dari 2500 baris dan 13 kolom dengan judul [Pumpkin Seeds Dataset](https://www.kaggle.com/datasets/muratkokludataset/pumpkin-seeds-dataset/).

Pada projek ini terdapat beberapa tahapan untuk memahami data :

#### 1. Deskripsi Variabel - Analisis Univariate

![Univariate](https://raw.githubusercontent.com/mochavin/ML-terapan-dicoding/main/univariate.png)
Gambar 1. Distribusi setiap variabel

Pada gambar 1, diperoleh informasi bahwa:

- `Area`: Menunjukkan luas dari biji labu, Rata-rata `area` adalah sekitar 80658.22 dengan standar deviasi 13664.51. `Area` berkisar dari 47939 hingga 136574. Distribusi tampaknya cenderung normal, dengan sebagian besar data terpusat di sekitar nilai rata-rata.
- `Perimeter`: Menunjukkan keliling dari biji labu, Rata-rata `perimeter` adalah sekitar 1130.28 dengan standar deviasi 109.26. `Perimeter` berkisar dari 868.49 hingga 1559.45. Distribusi tampaknya cenderung normal, dengan sebagian besar data terpusat di sekitar nilai rata-rata.
- `Major_Axis_Length`: Menunjukkan panjang dari sumbu utama biji labu, Rata-rata panjang sumbu utama adalah sekitar 456.60 dengan standar deviasi 56.24. Distribusi tampaknya cenderung normal.
- `Minor_Axis_Length`: Menunjukkan panjang sumbu minor, Rata-rata panjang sumbu minor adalah sekitar 225.79 dengan standar deviasi 23.30. Distribusi tampaknya cenderung normal.
- `Convex_Area`: Menunjukkan luas konveks biji labu, Rata-rata luas konvex adalah sekitar 81508.08 dengan standar deviasi 13764.09. Distribusi tampaknya cenderung normal.
- `Equiv_Diameter`: Menunjukkan diameter biji labu, Rata-rata diameter setara adalah sekitar 319.33 dengan standar deviasi 26.89. Distribusi tampaknya cenderung normal.
- `Eccentricity`: Menunjukkan eksentrisitas, Rata-rata eksentrisitas adalah sekitar 0.86 dengan standar deviasi 0.05. Distribusi tampaknya miring ke kanan.
- `Solidity`: Menunjukkan soliditas dari biji labu, Rata-rata soliditas adalah sekitar 0.99 dengan standar deviasi mendekati 0 karena nilainya rata-ratanya kecil. Soliditas memiliki distribusi yang sangat mendekati nilai maksimumnya.
- `Extent`: Menunjukkan Ekstensi pada biji labu, Rata-rata `extent` adalah sekitar 0.69 dengan standar deviasi 0.06. Distribusi tampaknya agak normal.
- `Roundness`: Menunjukkan tingkat kebulatan biji labu, Rata-rata kebulatan adalah sekitar 0.79 dengan standar deviasi 0.06. Distribusi tampaknya agak normal.
- `Aspect_Ration`: Menunjukkan rasio aspek dari biji labu, Rata-rata rasio aspek adalah sekitar 2.04 dengan standar deviasi 0.32. Distribusi tampaknya agak normal.
- `Compactness`: Menunjukkan tingkat kepadatan dari biji labu, Rata-rata kepadatannya adalah sekitar 0.70 dengan standar deviasi 0.05. Distribusi tampaknya agak normal.
- `Class`: Kelas _Çerçevelik_ atau Kelas _Ürgüp Sivrisi_ menunjukkan varietas dari biji labu yang akan diklasifikasi, kuantitasnya seimbang dengan varietas _Çerçevelik_ sedikit lebih banyak.

#### 2. Mengecek Ringkasan Data

Tahapan ini dilakukan agar dapat mendapatkan gambaran data secara umum. Di sini terlihat baris data, fitur yang ada, standar deviasi, dsb.

#### 3. Exploratory Data Analysis - Multivariate Analysis

Dengan mengamati Correlation Matrix dan Pairplot di `.ipynb`, diperoleh _insight_ sebagai berikut:

**Pairplot**
![Plot](https://raw.githubusercontent.com/mochavin/ML-terapan-dicoding/main/plot.png)
Gambar 2. Pairplot

**Correlation Matrix**

![Correlation Matrix](https://raw.githubusercontent.com/mochavin/ML-terapan-dicoding/main/corr.png)
Gambar 3. _Correlation Matrix_

Menurut gambar 3, diperoleh _insight_ sebagai berikut,

1. **Area, Convex_Area, Equiv_Diameter, Solidity berkolerasi lemah terhadap varietas Çerçevelik secara positif**, Ini menunjukkan bahwa semakin besar area, area conveks, diameter, dan soliditas, semakin mungkin objek tersebut termasuk dalam varietas _Çerçevelik_. Namun, korelasi ini tidak terlalu kuat, yang berarti ada faktor-faktor lain yang mempengaruhi varietas selain dari ukuran dan soliditas.

2. **Perimeter, Major_Axis_Length berkolerasi sedang terhadap varietas Çerçevelik secara positif**, Korelasi yang sedang antara perimeter dan panjang sumbu utama dengan varietas menunjukkan bahwa objek dengan perimeter dan panjang sumbu utama yang lebih besar cenderung termasuk dalam varietas _Çerçevelik_. Namun, seperti sebelumnya, hal ini tidak menjamin varietas _Çerçevelik_ secara pasti.

3. **Eccentricity, Aspect_Ratio berkolerasi kuat terhadap varietas Çerçevelik secara positif**, Korelasi yang kuat antara eksentrisitas dan rasio aspek dengan varietas menunjukkan bahwa objek yang lebih eksentrik atau memiliki rasio aspek yang lebih besar cenderung termasuk dalam varietas _Çerçevelik_.

4. **Extent berkolerasi lemah terhadap varietas _Ürgüp Sivrisi_ secara negatif**, Korelasi negatif antara extent dan varietas menunjukkan bahwa objek dengan extent yang lebih rendah cenderung termasuk dalam varietas _Ürgüp Sivrisi_. Ini mungkin menunjukkan bahwa objek yang lebih kompak atau lebih terkonsentrasi cenderung memiliki varietas _Ürgüp Sivrisi_.

5. **Minor_Axis_Length berkolerasi sedang terhadap varietas Ürgüp Sivrisi secara negatif**, Korelasi negatif antara panjang sumbu minor dan varietas menunjukkan bahwa objek dengan panjang sumbu minor yang lebih kecil cenderung termasuk dalam varietas _Ürgüp Sivrisi_.

6. **Roundness, Compactness berkolerasi kuat terhadap varietas _Ürgüp Sivrisi_ secara negatif**, Korelasi yang kuat antara bulat dan kekompakan dengan varietas menunjukkan bahwa objek yang lebih bulat atau lebih kompak cenderung termasuk dalam varietas _Ürgüp Sivrisi_.

Dengan pemahaman ini, faktor-faktor apa yang mungkin mempengaruhi varietas _Çerçevelik_ atau varietas _Ürgüp Sivrisi_ dalam dataset, serta mempertimbangkan pentingnya setiap fitur dalam membuat prediksi.

## Data Preparation

### 1. Mengecek _Missing Value_

Tahapan ini dilakukan agar setiap data yang akan diolah sudah dalam kondisi lengkap. Pada kasus ini tidak terdapat nilai yang kosong (_missing value_) sehingga dapat ke tahap selanjutnya.

### 2. Mengecek Keseimbangan Data

Tahapan ini mengecek keseimbangan data, dan memastikan bahwa tidak terjadi ketidakseimbangan antara kedua kelas biji labu agar model yang dihasilkan tidak mengalami kecenderungan prediksi kelas tertentu. pada kasus ini tidak terjadi ketidakseimbangan data sehingga tidak perlu diproses lebih lanjut.

### 3. Mengecek dan Menangani _Outliers_

Tahapan ini dilakukan agar meningkatkan performa model karena nilai yang sangat berbeda dari kumpulan data (_Outliers_) cenderung berpengaruh dalam pelatihan model. akan digunakan metode **IQR** sehingga nilai yang lebih kecil dari `Q1 - 1.5*IQR` dan lebih besar dari `Q3+1.5*IQR` merupakan nilai _Outlier_. Setelah membuang nilai _outlier_ data diubah dari 2500 baris dan 13 kolom menjadi 2285 baris dan 13 kolom.

### 4. Pembersihan Data: Menghapus Kolom yang Redundan

_Data Cleaning_ dengan menghapus kolom-kolom yang diidentifikasi sebagai redundant dalam DataFrame. Kolom yang dihapus adalah 'Convex_Area' dan 'Equiv_Diameter' karena keduanya dianggap sangat mirip dengan kolom 'Area'. Ini membantu menyederhanakan dataset dan mengurangi kelebihan fitur yang mungkin tidak diperlukan dalam analisis berikutnya.

### 5. _Train-Test-Split_

_Train-Test-Split_ yaitu membagi dataset menjadi data train dan data test diperlukan untuk menguji seberapa baik model dapat menggeneralisasi data baru. Fungsi train_test_split() dari library sklearn digunakan untuk membagi dataset. Pada proses ini, data dibagi dataset dengan rasio 80 20, yang berarti 80% (1828 baris data) akan digunakan untuk melatih model dan 20% (457 baris data) akan digunakan untuk menguji model.

### 6. Standarisasi

Standarisasi adalah proses penting dalam persiapan data untuk algoritma _machine learning_. Tujuannya adalah untuk menghasilkan data dengan skala relatif serupa atau mendekati distribusi normal, sehingga algoritma _machine learning_ dapat memiliki performa yang lebih baik dan konvergen lebih cepat.

Dalam proses standarisasi, tidak akan menggunakan teknik seperti _one-hot-encoding_ yang biasanya digunakan untuk fitur kategorikal. Sebaliknya, akan menggunakan teknik `StandarScaler` yang tersedia dalam library _Scikit-learn_. Ini akan membantu dalam mengubah fitur-fitur numerik sehingga memiliki mean 0 dan varians 1.

Melalui standarisasi, data menjadi lebih mudah diolah oleh algoritma _machine learning_, memungkinkan mereka untuk bekerja lebih efisien dan menghasilkan hasil yang lebih baik.

## Modeling

### 1. Model Development dengan _K-Nearest Neighbors_ (_KNN_)

_KNN_ memanfaatkan konsep "kesamaan fitur" untuk memprediksi nilai setiap data baru berdasarkan kemiripannya dengan titik data dalam set pelatihan. Dengan menggunakan algoritma ini, dapat ditentukan prediksi berdasarkan k-neighbors terdekat dari data baru. _KNN_ adalah pendekatan sederhana namun efektif dalam berbagai kasus, menawarkan kemudahan interpretasi dan penerapan yang luas dalam pemodelan data terstruktur.

Parameter yang digunakan pada proses pemodelan adalah menentukan parameter k (jumlah data terdekat) `n_neighbors=5`. Memilih k=5 dalam algoritma _K-Nearest Neighbors_ (_KNN_) seringkali dipertimbangkan karena nilai ini secara umum menawarkan keseimbangan yang memadai antara bias dan varians dalam model. Dengan mempertimbangkan lima tetangga terdekat, k=5 cenderung mengurangi dampak dari noise dalam dataset serta mencegah model terlalu responsif terhadap titik data individu yang mungkin merupakan outlier. Pilihan k=5 juga sering ditemukan memberikan performa yang stabil dalam banyak kasus tanpa memerlukan komputasi yang berlebihan.

_K-Nearest Neighbors_ (_KNN_) memiliki kelebihan dalam kemudahan implementasi dan interpretasi, serta fleksibilitasnya dalam menangani data non-linier dan multikelas. Pendekatan ini tidak memerlukan asumsi tertentu tentang distribusi data, sehingga cocok untuk berbagai macam masalah klasifikasi dan regresi. Selain itu, _KNN_ efektif dalam menangani data yang berubah-ubah, dan dapat beradaptasi dengan perubahan pola dalam dataset.

Meskipun demikian, _KNN_ memiliki beberapa kelemahan yang perlu dipertimbangkan. Algoritma ini rentan terhadap ketidakseimbangan kelas dalam dataset, yang dapat menyebabkan prediksi yang bias terhadap kelas mayoritas. Selain itu, _KNN_ memerlukan penyimpanan seluruh dataset dalam memori untuk melakukan prediksi, sehingga dapat menghabiskan banyak sumber daya pada dataset besar. _KNN_ juga sensitif terhadap fitur yang tidak relevan dan memiliki kinerja yang buruk dalam dimensi fitur tinggi karena _curse of dimensionality_.

### 2. Model Development dengan _Random Forest_

_Random Forest_ adalah algoritma pembelajaran mesin yang kuat dan serbaguna yang memanfaatkan teknik _ensemble_ dengan cara membangun sejumlah besar pohon keputusan selama pelatihan. Setiap pohon keputusan dalam _Random Forest_ diberi bobot dan keputusan akhir diambil berdasarkan mayoritas suara dari semua pohon. Pendekatan _ensemble_ ini membantu mengurangi _overfitting_ dan meningkatkan ketahanan model terhadap _noise_ dalam data, membuatnya cocok untuk berbagai macam masalah klasifikasi dan regresi.

Parameter yang digunakan pada proses pemodelan :

`max_depth=5`, `max_features=20`, dan `random_state=56` digunakan dalam _Random Forest_ untuk mengendalikan kompleksitas model. `max_depth=5` mengontrol kedalaman maksimum dari setiap pohon keputusan dalam _ensemble_, membantu mencegah _overfitting_. `max_features=20` menentukan jumlah maksimum fitur yang dipertimbangkan saat mencari split terbaik, memberikan variasi yang cukup tanpa membuat model terlalu rumit. `random_state=56` digunakan untuk menginisialisasi generator nomor acak, memastikan hasil yang sama di setiap pelatihan dan memfasilitasi pembandingan model yang konsisten. Keseluruhan, pengaturan ini membantu memperoleh keseimbangan antara kinerja model dan kompleksitas yang terkendali.

_Random Forest_ memiliki beberapa kelebihan, termasuk kemampuannya untuk mengatasi _overfitting_ melalui penggunaan beberapa pohon keputusan yang dihasilkan secara acak. Selain itu, model ini mampu menangani dataset besar dengan fitur yang banyak, serta memiliki kemampuan untuk menangani data yang hilang tanpa memerlukan preprocessing yang ekstensif. Namun, kelemahan _Random Forest_ termasuk kompleksitas komputasi yang tinggi, terutama pada dataset besar dan kompleks. Selain itu, interpretasi model _Random Forest_ seringkali sulit karena kompleksitasnya yang tinggi, dan kecenderungan untuk menjadi _blackbox_ yang sulit dimengerti oleh manusia.

## 3. Model Development dengan _Boosting Algorithm_

_Boosting_ adalah teknik di _machine learning_ di mana model lemah digunakan secara berulang dan ditingkatkan untuk membuat model yang lebih kuat. Model lemah ini fokus pada kesalahan sebelumnya dan diberi bobot lebih besar untuk memperbaikinya. Dengan cara ini, _Boosting_ meningkatkan performa model secara bertahap, menjadikannya efektif untuk menangani masalah klasifikasi atau regresi yang kompleks. Di sini digunakan _AdaBoost_.

Pemilihan parameter `learning_rate = 0.0001` dan `random_state = 42` dalam `AdaBoostClassifier` didasarkan pada pertimbangan tertentu. `learning_rate = 0.0001` dipilih untuk mengurangi dampak dari setiap pohon keputusan yang ditambahkan dalam proses _boosting_, memungkinkan algoritma untuk secara perlahan menyesuaikan diri dengan data dan mencegah _overfitting_. `random_state = 42` digunakan untuk menginisialisasi generator nomor acak, memastikan hasil yang konsisten pada setiap pelatihan. Penggunaan kedua parameter ini membantu mengoptimalkan performa dan stabilitas model AdaBoost.

Kelebihan _Adaboost_ meliputi kemampuannya untuk mengidentifikasi dan menyesuaikan kesalahan prediksi sebelumnya, fokus pada sampel yang sulit diklasifikasikan untuk meningkatkan performa model secara keseluruhan, serta kemampuannya untuk bekerja dengan baik pada dataset yang kompleks. Namun, _Adaboost_ rentan terhadap gangguan oleh data yang _noise_ dan _outlier_, serta sensitif terhadap data yang tidak seimbang dalam kelasnya, yang dapat menghasilkan model yang bias terhadap kelas mayoritas.

### Model Yang Dipilih

Akan dipilih model **Random Forest** karena secara performa lebih baik dibandingkan kedua model lainnya pada kasus ini. Performa yang digunakan adalah _F1 Score_. Cenderung performa model **Random Forest** memberikan solusi terbaik bagi kasus ini karena baik train dan testing memiliki performa lebih baik.

## Evaluation

_F1 score_ adalah metrik evaluasi klasifikasi yang mengukur presisi (precision) dan recall secara bersamaan. _F1 score_ merupakan rata-rata harmonis dari presisi dan recall, yang memberikan bobot yang sama kepada kedua metrik tersebut. _F1 score_ berguna saat ingin mencapai keseimbangan antara presisi dan recall.

Formula _F1 Score_:
$$F1 = 2 \times \frac{precision \times recall}{precision + recall}$$

Presisi (precision) adalah jumlah true positive dibagi dengan jumlah true positive ditambah false positive. Presisi mengukur seberapa banyak dari kasus positif yang terprediksi benar oleh model.

$$Precision = \frac{TP}{TP + FP}$$

Recall (recall), juga dikenal sebagai sensitivity atau true positive rate, adalah jumlah true positive dibagi dengan jumlah true positive ditambah false negative. Recall mengukur seberapa banyak dari kasus positif yang diprediksi dengan benar oleh model.

$$Recall = \frac{TP}{TP + FN}$$

Dalam konteks klasifikasi, True Positive (TP) adalah jumlah kasus positif yang diprediksi dengan benar, False Positive (FP) adalah jumlah kasus negatif yang salah diprediksi sebagai positif, dan False Negative (FN) adalah jumlah kasus positif yang salah diprediksi sebagai negatif.

_F1 score_ memberikan ukuran yang baik tentang kinerja model klasifikasi ketika kelas target tidak seimbang atau ketika biaya kesalahan false positives dan false negatives tidak simetris. Misalnya, dalam kasus di mana lebih peduli dengan false negatives daripada false positives atau sebaliknya.

Jadi, _F1 score_ adalah metrik evaluasi klasifikasi yang baik untuk kasus di mana ingin memperhatikan keseimbangan antara presisi dan recall.

Berikut hasil peroleh _F1 score_ setiap model :

Tabel 1. Perbandingan _f1 score_ dari tiap model
| Model | Train | Test |
|-----------|---------|---------|
| KNN | 0.892709| 0.864734|
| RFC | 0.901629| 0.875 |
| Boosting | 0.850935| 0.867925|

Berikut gambar bar _chart_ perbandingan nilai _F1 score_ :

![Chart Metrics](https://raw.githubusercontent.com/mochavin/ML-terapan-dicoding/main/chart_metriks.png)

Gambar 4. Metriks perbandingan _f1 score_ dari tiap model

Berikut sample hasil prediksi model dalam bentuk tabel :

Tabel 2. Sample hasil prediksi tiap model
| y_true | prediksi_KNN | prediksi_rfc | prediksi_Boosting |
|--------|--------------|--------------|-------------------|
| 804 | 0.0 | 0.0 | 0.0 |
| 743 | 0.0 | 0.0 | 1.0 |
| 315 | 0.0 | 0.0 | 0.0 |
| 1994 | 1.0 | 1.0 | 1.0 |
| 248 | 0.0 | 1.0 | 1.0 |
| 1297 | 0.0 | 0.0 | 0.0 |
| 2310 | 1.0 | 0.0 | 1.0 |
| 2478 | 1.0 | 1.0 | 1.0 |
| 697 | 0.0 | 0.0 | 0.0 |
| 1584 | 1.0 | 1.0 | 1.0 |

Berdasarkan tabel dan chart nilai _F1 score_ cenderung model **Random Forest** lebih baik dari kedua algoritma lainnya. Melalui evaluasi yang telah dilakukan, didapat bahwa model yang terbaik adalah model _Random Forest_ dengan _f1 score_ pada test sebesar 87,5% yang mana artinya proyek ini telah berhasil melampaui skor minimum yang telah ditentukan sehingga diharapkan mampu menyelesaikan _problem statement_ dan mencapai _goals_ yang telah dipaparkan.

## _Conclusion_

Dalam proyek ini, penulis telah menyelesaikan serangkaian tahapan yang penting. Penulis memulai dengan memahami domain proyek dan kasus bisnis yang ingin diselesaikan. Setelah itu, penulis menyelami data yang tersedia, menyiapkan dataset, melakukan proses pemodelan, dan mengevaluasi hasilnya. Penulis berhasil menyelesaikan tantangan bisnis yang dihadapi dan mencapai tujuan dengan membangun 3 model yang berbeda. Dari ketiga model tersebut, penulis memilih model `Random Forest` sebagai solusi terbaik untuk masalah yang dihadapi.

### REFERENCES

[1] Koklu, M., Sarigil, S. & Ozbek, O. "The use of machine learning methods in classification of pumpkin seeds (Cucurbita pepo L.)". Genet Resour Crop Evol 68, 2713–2726 (2021).

[2] Seymen M, Yavuz D, Dursun A, Kurtar ES, Turkmen O (2019)
"Identification of drought-tolerant pumpkin (_Cucurbita
pepo L._) genotypes associated with certain fruit characteristics, seed yield, and quality". Agric Water Manag 221:150–159
