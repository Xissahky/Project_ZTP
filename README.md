# Credit Repayment Prediction (Prognoza spłaty kredytu)

## Cel
Predykcja czy klient spłaci pożyczkę na czas (`paid_on_time`: 1/0) na podstawie cech finansowych i demograficznych.

## Dane
Pliki w `data/raw/`:
- `loans.csv` – informacje o pożyczkach (m.in. kwota, okres, itp.)
- `customers.csv` – demografia i profil klienta

Łączenie po `customer_id`.

## Uruchomienie
1) Instalacja:
```bash
pip install -r requirements.txt
