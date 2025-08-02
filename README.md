# CHAOS - Quantum Computing Simulator

CHAOS adalah simulator komputasi kuantum yang diimplementasikan dalam Python. Proyek ini bertujuan untuk mensimulasikan perilaku komputer kuantum di atas komputer klasik.

## Konsep

Di mitologi Yunani, Chaos adalah ketiadaan primordial tempat segala sesuatu lahir. Sama seperti komputer kuantum yang beroperasi di ranah probabilitas dan superposisi sebelum "lahir" menjadi satu jawaban pasti.

## Fitur

Proyek ini akan membangun library Python yang dapat:

1. Mendefinisikan Qubit, unit dasar komputasi kuantum yang bisa berada dalam keadaan 0, 1, atau keduanya sekaligus (superposisi).
2. Menerapkan Gerbang Kuantum (Quantum Gates), yaitu operasi-operasi (seperti rotasi atau flip) yang memanipulasi keadaan Qubit.
3. Mensimulasikan Keterkaitan Kuantum (Entanglement), fenomena "ajaib" di mana dua Qubit terhubung secara misterius.
4. Menjalankan Sirkuit Kuantum (serangkaian gerbang) dan "mengukur" hasilnya untuk mendapatkan jawaban probabilistik.

## Instalasi

Proyek ini menggunakan Python dan NumPy. Untuk menginstal dependensi:

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Di Windows:
venv\Scripts\activate
# Di Unix/MacOS:
source venv/bin/activate

# Instal dependensi
pip install numpy
```

## Penggunaan

### Qubit

```python
from qubit import Qubit

# Membuat qubit dalam keadaan |0⟩
q0 = Qubit(0)

# Membuat qubit dalam keadaan |1⟩
q1 = Qubit(1)

# Membuat qubit dalam superposisi
# 1/√2 |0⟩ + 1/√2 |1⟩ (50% kemungkinan 0, 50% kemungkinan 1)
import numpy as np
q_super = Qubit([1/np.sqrt(2), 1/np.sqrt(2)])

# Mengukur qubit
result = q_super.measure()
print(f"Hasil pengukuran: {result}")  # 0 atau 1 dengan probabilitas 50%

# Melihat probabilitas
prob_zero, prob_one = q_super.get_probabilities()
print(f"Probabilitas |0⟩: {prob_zero}, Probabilitas |1⟩: {prob_one}")
```

## Pengembangan

Proyek ini dikembangkan dalam beberapa fase:

### Fase 1: Partikel Kuantum (The Qubit)
- [x] Milestone 1.1: Buat class Qubit di Python. Secara internal, representasikan keadaannya sebagai vektor 2D dengan bilangan kompleks (pakai NumPy).
- [x] Milestone 1.2: Implementasikan fungsi untuk menginisialisasi Qubit ke keadaan |0⟩ atau |1⟩.
- [x] Milestone 1.3: Implementasikan fungsi measure() yang akan "meruntuhkan" superposisi Qubit menjadi 0 atau 1 berdasarkan probabilitas amplitudonya.

### Fase 2: Hukum Fisika (The Quantum Gates)
- [ ] Milestone 2.1: Representasikan setiap gerbang kuantum (Pauli-X, Hadamard, CNOT) sebagai matriks 2x2 atau 4x4.
- [ ] Milestone 2.2: Buat fungsi apply_gate(qubit, gate) yang melakukan perkalian matriks untuk mengubah keadaan Qubit.

### Fase 3: Alam Semesta Mini (The Quantum Circuit)
- [ ] Milestone 3.1: Buat class QuantumCircuit yang bisa mengelola banyak Qubit (sebuah quantum register).
- [ ] Milestone 3.2: Implementasikan metode untuk menambah serangkaian gerbang ke dalam sirkuit.
- [ ] Milestone 3.3: Buat fungsi run() yang akan mengeksekusi semua gerbang secara berurutan di dalam sirkuit dan mengembalikan hasil pengukuran akhir.
