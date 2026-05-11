# CAV Merging — GBD Tabanlı Otonom Araç Birleşme Kontrolü

Bağlantılı otonom araçların (CAV) çok şeritli otoyol birleşme problemini **Generalized Benders Decomposition (GBD)** algoritması ile çözen Python implementasyonu. SUMO trafik simülasyonu entegrasyonu içerir.

## Kurulum

```bash
pip install pulp cvxpy numpy matplotlib
```

## Run

```bash
python case_study_1.py
python case_study_1.py --real
```

## Algoritma

GBD çözücüsü MINLP birleşme problemini iki alt probleme ayırır:

| Bileşen | Açıklama | Kütüphane |
|---|---|---|
| **RMP** | Binary sıralama ve şerit değişikliği kararları (MILP) | PuLP / CBC |
| **PS** | Sürekli yörünge optimizasyonu (Konveks program) | CVXPY / ECOS |

**Döngü:**
1. RMP çöz → binary kararlar + alt sınır (LB)
2. PS çöz → yörünge maliyeti + üst sınır (UB)
3. PS infeasible → Feasibility Cut ekle
4. PS feasible → Optimality Cut ekle
5. `|UB - LB| ≤ ε` olana kadar tekrarla

> Bu proje [Claude](https://claude.ai) (Anthropic) yapay zeka asistanı yardımıyla geliştirilmiştir.