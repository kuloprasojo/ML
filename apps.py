import streamlit as st
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn. tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from math import e
import pandas as pd

description, Dataset, Preprocessing, Implementasi = st.tabs(["Description", "Dataset", "Preprocessing", "implementasi"])

with description:
    st.write("""# Data Set Description """)
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link")
    st.write("""1. Prediksi Kanker Paru :

    Kanker paru-paru adalah kanker yang terbentuk di paru-paru. Kanker ini merupakan salah satu kanker yang umum terjadi di Indonesia. Secara global, kanker paru-paru merupakan penyebab pertama kematian akibat kanker pada pria dan penyebab kedua kematian akibat kanker pada wanita.

    Meski sering terjadi pada perokok, kanker paru-paru juga bisa terjadi pada orang yang bukan perokok. Pada orang bukan perokok, kanker paru-paru terjadi akibat sering terpapar asap rokok dari orang lain (perokok pasif) atau paparan zat kimia di lingkungan kerja.
    """)

    st.write("""## Tentang kumpulan data ini :

    Kumpulan data ini berisi informasi tentang pasien kanker paru-paru, termasuk usia, jenis kelamin, paparan polusi udara, 
    penggunaan alkohol, alergi debu, bahaya pekerjaan, risiko genetik, penyakit paru-paru kronis, diet seimbang, obesitas, merokok, perokok pasif, nyeri dada, batuk darah, kelelahan, penurunan berat badan, sesak napas, mengi, kesulitan menelan, kuku jari tabuh dan mendengkur
    """)

    st.write("""## Cara menggunakan kumpulan data :

    Kanker paru-paru adalah penyebab utama kematian akibat kanker di seluruh dunia, terhitung 1,59 juta kematian pada tahun 2018. 
    Sebagian besar kasus kanker paru disebabkan oleh merokok, tetapi paparan polusi udara juga merupakan faktor risiko. 
    Sebuah studi baru menemukan bahwa polusi udara dapat dikaitkan dengan peningkatan risiko kanker paru-paru, bahkan pada bukan perokok.
    Studi yang diterbitkan dalam jurnal Nature Medicine ini mengamati data dari lebih dari 462.000 orang di China yang diikuti selama rata-rata enam tahun. 
    Peserta dibagi menjadi dua kelompok yaitu mereka yang tinggal di daerah dengan tingkat polusi udara tinggi dan mereka yang tinggal di daerah dengan tingkat polusi udara rendah.
    Para peneliti menemukan bahwa orang-orang dalam kelompok polusi tinggi lebih mungkin mengembangkan kanker paru-paru daripada kelompok polusi rendah. 
    Mereka juga menemukan bahwa risiko lebih tinggi pada bukan perokok daripada perokok, dan risiko meningkat seiring bertambahnya usia.
    Sementara studi ini tidak membuktikan bahwa polusi udara menyebabkan kanker paru-paru, hal itu menunjukkan bahwa mungkin ada hubungan antara keduanya. 
    Penelitian lebih lanjut diperlukan untuk mengkonfirmasi temuan ini dan untuk menentukan apa pengaruh jenis dan tingkat polusi udara yang berbeda terhadap risiko kanker paru-paru
    """)

with Dataset:
    st.write("""# Dataset""")

    # Membaca dataset
    dataset = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/lung_cancer_patient_datasets.csv')

    # Menampilkan dataset di Streamlit
    st.write("Dataset asli:")
    st.dataframe(dataset)

with Preprocessing:
    st.write("""# Preprocessing""")

    dataset = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/lung_cancer_patient_datasets.csv')
    dataset = dataset.drop(columns=['index', 'Patient Id'])
    X = dataset.drop(columns=["Level"])
    y = dataset["Level"]

    # Menampilkan dataset di Streamlit
    st.write("Dataset:")
    st.write(dataset)

    # Menampilkan X di Streamlit
    st.write("Variabel X:")
    st.write(X)

    # Menampilkan y di Streamlit
    st.write("Variabel y:")
    st.write(y)


with Implementasi:
    st.write("""# Implementasi""")
    dataset = pd.read_csv('https://raw.githubusercontent.com/aliefakbrp/dataset/main/lung_cancer_patient_datasets.csv')
    dataset = dataset.drop(columns=['index', 'Patient Id'])
    X = dataset.drop(columns=["Level"])
    y = dataset["Level"]
    X=X.values.tolist()
    X = np.array(X)
    y = np.array(y)

    class adaboost:
        def __init__(self, num_of_model = 20,lr=0.1 ):
            self.num_of_model = num_of_model
            self.lr = lr
            self.clfs = []
            self.clfs_weight = []
            self.clfs_weight_final = []
            self.rj_final = []
            self.alpha_final = []
            self.update_weight_final = []
            self.y_pred_final_fit = []
            self.y_pred_every_clf=[]
            self.final_result = []

        def CalcWeightedError(self,y_true, y_pred):
            # kalo nilainya tidak sama +1
            return np.sum(y_true != y_pred) / len(y_pred)

        def CalcError(self,y_true, y_pred):
            # menghasilkan True apabila y_true (target aktual) tidak sesuai dengan nilai aktual
            res = np.array([y_true != y_pred])
            res = res.astype(int)
            return res

        def UpdateWeights(self,weights, pred_weight, error):
            new_weights = weights * (e ** (pred_weight * error))
            new_weights = new_weights/np.sum(new_weights)
            return new_weights

        def CalcPredWeight(self,lr, weighted_error):
            return self.lr * np.log((1 - weighted_error + 1)/(weighted_error + 1))
        
        def fit(self,X,y):
            X_train = X
            m, n = X_train.shape
            y_train = y
            weights = np.full(len(X_train), 1/len(X_train))
            for i in range(self.num_of_model):
                # print(i)
                # inisialisasi classifier
                clf = KNeighborsClassifier(n_neighbors=3)
                # fitting data
                clf.fit(X_train, y_train)
                # append clf to clfs
                self.clfs.append(clf)
                # predict X_train
                y_pred = clf.predict(X_train)
                # append y_pred to y_pred_final_fit
                self.y_pred_final_fit.append(y_pred)
                # print("ini y pred fit",y_pred)

                # menghitung persamaan 2 (rj)
                weighted_error = self.CalcWeightedError(y_train, y_pred)
                # menghitung persamaan 3 (alpha)
                pred_weight = self.CalcPredWeight(self.lr, weighted_error)
                # menambahkan persamaan 3 to clfs_weight
                self.clfs_weight.append(pred_weight)

                # print("ini self.clfs_weight",self.clfs_weight)
                # perhitungan error
                error = self.CalcError(y_train, y_pred)
                # bobot baru
                new_weights = self.UpdateWeights(weights, pred_weight, error)
                # print("ini new_weigth",new_weights)

                # index baru
                new_indices = np.random.choice(m, m, p=new_weights.flatten())
                # print("new_indices",new_indices)
                X_train = X_train[new_indices]
                # print(X_train)
                y_train = y_train[new_indices]

        def predict(self,x,y):
            n_estimator = len(self.clfs)
            m=len(x)
            # membuat matrix m x n_estimator
            constructor=[[str(i)+str(j) for i in range(n_estimator)] for j in range(m)]
            # mengisi konstruksi dengan hasil prediktor
            for i in range(n_estimator):
                clf = self.clfs[i]
            for u in range(m):
                constructor[u][i]=clf.predict([x[u]])
            self.y_pred_every_clf=constructor
            # mengisi prediksi akhir
            # perulangan sebanyak data
            for i in range(len(self.y_pred_every_clf)):
                y_pred_temp=[]
                weight_temp=[]
            # perulangan clf sebanyak n_estimator(banyaknya model)
            for u in range(len(self.y_pred_every_clf[0])):
                if self.y_pred_every_clf[i][u] not in y_pred_temp:
                    y_pred_temp.append(self.y_pred_every_clf[i][u])
                    weight_temp.append(self.clfs_weight[u])
                else:
                    for o in range(len(y_pred_temp)):
                        if self.y_pred_every_clf[i][u]==y_pred_temp[o]:
                            weight_temp[o]+=self.clfs_weight[u]
            self.clfs_weight_final.append(weight_temp)
            indexnya=weight_temp.index(max(weight_temp))
            # st.write("ini index",indexnya)
            # st.write("ini y_pred_temp",y_pred_temp)
            self.final_result.append(y_pred_temp[19])
            return self.final_result

    # X=X.values.tolist()
    X = np.array(X)

    y = np.array(y)

    st.write("## Inputan")
    Age = st.number_input('Masukkan Age (Usia) : ')
    Gender = st.number_input('Masukkan Gender (Jenis kelamin) : ')
    Air_Pollution = st.number_input('Masukkan Air Pollution (Air Pollution) : ')
    Alcohol_Use = st.number_input('Masukkan Alcohol Use (Alcohol Use) : ')
    Dust_Allergy = st.number_input('Masukkan Dust Allergy (Dust Allergy) : ')
    OccuPational_Hazards = st.number_input('Masukkan OccuPational Hazards (OccuPational Hazards) : ')
    Genetic_Risk = st.number_input('Masukkan Genetic Risk (Genetic Risk) : ')
    chronic_Lung_Disease = st.number_input('Masukkan chronic Lung Disease (chronic Lung Disease) : ')
    Balanced_Diet = st.number_input('Masukkan Balanced Diet (Balanced Diet) : ')
    Obesity = st.number_input('Masukkan Obesity (Obesity) : ')
    Smoking = st.number_input('Masukkan Smoking (Smoking) : ')
    Passive_Smoker = st.number_input('Masukkan Passive Smoker (Passive Smoker) : ')
    Chest_Pain = st.number_input('Masukkan Chest Pain (Chest Pain) : ')
    Coughing_of_Blood = st.number_input('Masukkan Coughing of Blood (Coughing of Blood) : ')
    Fatigue = st.number_input('Masukkan Fatigue (Fatigue) : ')
    Weight_Loss = st.number_input('Masukkan Weight Loss (Weight Loss) : ')
    Shortness_of_Breath = st.number_input('Masukkan Shortness of Breath (Shortness of Breath) : ')
    Wheezing = st.number_input('Masukkan Wheezing (Wheezing) : ')
    Swallowing_Difficulty = st.number_input('Masukkan Swallowing Difficulty (Swallowing Difficulty) : ')
    Clubbing_of_Finger_Nails = st.number_input('Masukkan Clubbing of Finger Nails (Clubbing of Finger Nails) : ')
    Frequent_Cold = st.number_input('Masukkan Frequent Cold (Frequent Cold) : ')
    Dry_Cough = st.number_input('Masukkan Dry Cough (Dry Cough) : ')
    Snoring = st.number_input('Masukkan Snoring (Snoring) : ')

    inputs = np.array([[
        Age,
        Gender,
        Air_Pollution,
        Alcohol_Use,
        Dust_Allergy,
        OccuPational_Hazards,
        Genetic_Risk,
        chronic_Lung_Disease,
        Balanced_Diet,
        Obesity,Smoking,
        Passive_Smoker,
        Chest_Pain,
        Coughing_of_Blood,
        Fatigue,
        Weight_Loss,
        Shortness_of_Breath,
        Wheezing,
        Swallowing_Difficulty,
        Clubbing_of_Finger_Nails,
        Frequent_Cold,
        Dry_Cough,
        Snoring,
        ]])
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    clf = adaboost()
    clf.fit(X_train, y_train)
    clf.predict(inputs, y_test)
    st.write("ini hasilnya",clf.final_result[0])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5)
    clf = adaboost()
    clf.fit(X_train, y_train)

    clf.predict(X_test,y_test)
    # clf.final_result


    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5)
    clf = adaboost()
    clf.fit(X_train, y_train)

    clf.predict(inputs, y_test)
    # st.write(clf.final_result)
