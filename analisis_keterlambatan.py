import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Pelanggan": [
        "Pelanggan A", "Pelanggan B", "Pelanggan C",
        "Pelanggan D", "Pelanggan E", "Pelanggan F",
        "Pelanggan G", "Pelanggan H", "Pelanggan I",
        "Pelanggan J", "Pelanggan K", "Pelanggan L",
        "Pelanggan M", "Pelanggan N", "Pelanggan O"
    ],

    "Metode_Pembayaran": [
        "Transfer", "Transfer", "Tunai", "Kredit", "Transfer",
        "Kredit", "Tunai", "Transfer", "Kredit", "Transfer",
        "Tunai", "Kredit", "Transfer", "Kredit", "Transfer"
    ],

    "Nilai_Invoice": [
        2500000, 7500000, 1500000, 9000000, 3500000,
        8000000, 1200000, 6000000, 9500000, 2800000,
        1800000, 8500000, 4500000, 10000000, 3200000
    ],

    "Jumlah_Transaksi_Sebelumnya": [
        10, 15, 5, 18, 12,
        20, 4, 14, 22, 8,
        6, 19, 13, 25, 9
    ],

    "Rata_Rata_Pembayaran": [
        20, 15, 10, 45, 18,
        50, 12, 25, 55, 22,
        14, 48, 20, 60, 17
    ],

    "Terlambat": [
        0, 0, 0, 1, 0,
        1, 0, 0, 1, 0,
        0, 1, 0, 1, 0
    ]
}

df = pd.DataFrame(data)

print("======================================================")
print("   ANALISIS KETERLAMBATAN PEMBAYARAN PELANGGAN")
print("======================================================")

print("\nDATA PELANGGAN:")
print(df.to_string(index=False))

encoder = LabelEncoder()

df["Metode_Pembayaran"] = encoder.fit_transform(
    df["Metode_Pembayaran"]
)

X = df[
    [
        "Metode_Pembayaran",
        "Nilai_Invoice",
        "Jumlah_Transaksi_Sebelumnya",
        "Rata_Rata_Pembayaran"
    ]
]

y = df["Terlambat"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

akurasi = accuracy_score(y_test, y_pred)

print("\n======================================================")
print("                  HASIL ANALISIS")
print("======================================================")

print("Data Aktual    :", list(y_test))
print("Hasil Prediksi :", list(y_pred))

print("\nAkurasi :", round(akurasi * 100, 2), "%")

print("\nHasil Evaluasi:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

pelanggan_baru = pd.DataFrame({
    "Metode_Pembayaran": ["Kredit"],
    "Nilai_Invoice": [9000000],
    "Jumlah_Transaksi_Sebelumnya": [20],
    "Rata_Rata_Pembayaran": [50]
})

pelanggan_baru["Metode_Pembayaran"] = encoder.transform(
    pelanggan_baru["Metode_Pembayaran"]
)

hasil = model.predict(pelanggan_baru)

print("======================================================")
print("              PREDIKSI PELANGGAN BARU")
print("======================================================")

if hasil[0] == 1:
    print("Hasil : TERLAMBAT MEMBAYAR")
    print("Pelanggan perlu mendapatkan perhatian lebih.")
else:
    print("Hasil : TIDAK TERLAMBAT MEMBAYAR")
    print("Pelanggan diprediksi membayar tepat waktu.")

print("======================================================")