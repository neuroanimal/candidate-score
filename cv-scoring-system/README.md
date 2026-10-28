# cvscore: ocena CV, listów motywacyjnych i dopasowania do ofert

Narzędzie ocenia CV (PDF, także eksport z Adobe InDesign, DOCX, TXT), opcjonalnie list motywacyjny,
i porównuje je z ofertą pracy. Zwraca około 60 metryk w czterech grupach, wynik zbiorczy, pewność
oceny i listę poprawek uszeregowaną według tego, ile każda z nich podniosłaby wynik.

**Nie używa modeli językowych (LLM).** Do ekstrakcji służą PyMuPDF i Tesseract OCR (`pol+eng`).
Dalej działają słowniki umiejętności PL/EN, reguły, TF-IDF, BM25 oraz, opcjonalnie, lokalny model
embeddingów zdań (`paraphrase-multilingual-MiniLM-L12-v2`, sieć do podobieństwa zdań, nie LLM).
Wagi można douczyć regresją logistyczną na prawdziwych wynikach rekrutacji (`cvscore calibrate`).

Profile zawodowe są wykrywane automatycznie z oferty, a przy ofercie mieszanej dwa profile są łączone:

| Profil | Dla kogo |
|---|---|
| `tech` | automatyzacja, architektura oprogramowania, data science |
| `creative_arts` | malarstwo, grafika warsztatowa i komputerowa, ilustracja |
| `social_media` | media społecznościowe, content, marketing |
| `events` | organizacja wydarzeń, animacja kultury, granty |
| `education` | warsztaty dla dzieci, dorosłych i seniorów, edukacja artystyczna |
| `therapy` | arteterapia, terapia zajęciowa, pedagogika specjalna, praca z osobami z chorobami neurologicznymi i zaburzeniami psychicznymi (ŚDS, WTZ, DPS) |

## Instalacja (WSL / Linux)

```bash
cd cv-scoring-system
python3 -m venv .venv
.venv/bin/pip install -e .                    # zależności z przypiętymi wersjami
sudo apt install tesseract-ocr tesseract-ocr-pol   # OCR dla PDF bez warstwy tekstowej
# opcjonalnie: embeddingi semantyczne (CPU, ok. 1 GB z torch)
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install sentence-transformers
CVSCORE_EMBEDDINGS=download .venv/bin/python -c "from cvscore import similarity; print(similarity.embeddings_available())"
```

Model embeddingów pobiera się raz, z publicznego repozytorium Hugging Face. Potem narzędzie działa
wyłącznie offline i nie wysyła nigdzie żadnych dokumentów. Bez embeddingów wszystko działa, tylko
podobieństwo semantyczne liczy się z TF-IDF i BM25.

## Użycie

```bash
# jedno CV, opcjonalnie z ofertą i listem
.venv/bin/cvscore score data/cv.pdf --job data/oferta.pdf --letter data/list.pdf --out reports/fundacja

# podgląd: co narzędzie odczytało z PDF (kolejność kolumn, OCR, sekcje, daty)
.venv/bin/cvscore extract data/cv.pdf

# podgląd: jak zrozumiano ofertę (wymagania, mile widziane, obowiązki, warunki konieczne)
.venv/bin/cvscore job data/oferta.pdf

# kilka wersji CV wobec jednej oferty: która lepsza i dlaczego
.venv/bin/cvscore compare cv_terapia.pdf cv_grafika.pdf cv_ogolne.pdf --job oferta.pdf --out reports/porownanie

# wiele aplikacji naraz, plus macierz "każde CV x każda oferta"
.venv/bin/cvscore batch data/aplikacje --out reports/batch

# lista metryk i profili
.venv/bin/cvscore metrics
.venv/bin/cvscore profiles
```

Raport powstaje w trzech formatach: `.html` (czytelny, z tabelami), `.md` i `.json`. Zawiera:

- wyniki zbiorcze i przesiew rekrutera;
- listę poprawek uszeregowaną według zysku;
- szkice przeredagowania słabych punktów. Fakty, których narzędzie nie zna, zostają w `[nawiasach]`,
  a odmianę wyrazów trzeba poprawić ręcznie;
- wszystkie metryki z dowodami;
- wymagania oferty z poziomem dowodu i ocenę trafności każdej roli;
- podgląd tego, jak odczytano CV.

Wynik odczytu PDF (zwłaszcza OCR) jest zapamiętywany w `~/.cache/cvscore` według treści pliku,
więc ponowne uruchomienie jest szybkie. Wyłącza się to zmienną `CVSCORE_CACHE=0`.

Obsługiwane zapisy dat: `03.2019 – 06.2021`, `01/09/2018 – 28/02/2021` (Europass), `III 2019 – nadal`,
`marzec 2019 – obecnie`, `15 marca 2021 – 30 czerwca 2022`, `2016-2018`, `od 2018`, `Jan 2022 – Present`.
Etykiety pól Europass („Zajmowane stanowisko”, „Daty”...) są pomijane. Oferty z BIP z numerowanymi
punktami i wymaganiami rozdzielonymi średnikami są dzielone na pojedyncze wymagania.

### Układ folderu z aplikacjami

Najprościej: jeden podfolder na aplikację. Typ pliku (CV, list, oferta) rozpoznaje się po nazwie
albo po treści.

```
data/aplikacje/
  2026-03_fundacja_sds/
    CV_Anna.pdf
    list_motywacyjny.pdf
    oferta.pdf            # ogłoszenie zapisane jako PDF (Ctrl+P -> Zapisz jako PDF) albo .txt
    wynik.txt             # opcjonalnie: "rozmowa", "odrzucona", "brak odpowiedzi", "zatrudniona"
  2026-04_dom_kultury/
    ...
```

Zamiast podfolderów można dać `manifest.yaml` z jawną listą (`id`, `cv`, `letter`, `job`, `outcome`).
Można też wrzucić wszystkie pliki do jednego płaskiego folderu: wtedy każde CV jest oceniane
z każdą ofertą.

Zestawienie `reports/batch/index.html` pokazuje:

- wszystkie aplikacje z wynikami i trzema najważniejszymi poprawkami;
- która wersja CV najlepiej pasuje do której oferty;
- które metryki odróżniały aplikacje zakończone rozmową od pozostałych (AUC, korelacja, NDCG).
  Liczy się to dopiero przy co najmniej 4 rekrutacjach z wynikiem, w tym pozytywnych i negatywnych.

Kiedy uzbiera się co najmniej 8 rekrutacji z wynikiem, `cvscore calibrate data/aplikacje --out model.json`
pokaże, które metryki faktycznie przewidują zaproszenie na tym rynku pracy (regresja logistyczna,
walidacja leave-one-out). Przy tak małej próbie to wskazówka, nie dowód.

## Eksport CV z Adobe InDesign

Najczęstsze problemy i to, jak narzędzie je obsługuje:

| Problem | Objaw | Co robi narzędzie | Co zrobić w InDesign |
|---|---|---|---|
| Tekst zamieniony na krzywe (*Create Outlines*) albo wklejony jako obraz | brak warstwy tekstowej | OCR (Tesseract), pewność w raporcie | nie zamieniaj tekstu na krzywe |
| Układ wielokolumnowy, pasek boczny | ATS czyta kolumny na przemian | rekonstrukcja kolejności (XY-cut) | dla formularzy online trzymaj też wersję jednokolumnową |
| Rozstrzelone nagłówki (tracking) | „D O Ś W I A D C Z E N I E” | sklejanie liter, rozpoznanie nagłówka bez spacji | - |
| Link schowany pod ikoną | adres niewidoczny w tekście i na wydruku | odczyt adnotacji linków | wpisz adres portfolio także jako tekst |
| Fonty bez mapy ToUnicode | „krzaki” zamiast tekstu | wykrycie i OCR strony | *Create Tagged PDF*, osadzanie fontów |
| Strony z pracami (portfolio) | OCR obrazu = szum | strona graficzna pomijana w ocenie tekstu i długości, zaliczana do portfolio | - |

## Co jest mierzone

Pełna lista z opisem: `cvscore metrics`. Wagi bazowe są w `src/cvscore/metrics/base.py`, mnożniki
profili w `src/cvscore/data/profiles.yaml`.

- **Dokument (14):** warstwa tekstowa, parsowalność ATS, złożoność układu, kompletność sekcji, kontakt,
  długość, czytelność (zdania, FOG), skanowalność, spójność (formaty dat, chronologia, mieszanie form
  -łam/-łem), frazesy, poprawność zapisu (brak polskich znaków, wersaliki), klauzula RODO,
  minimalizacja danych osobowych (PESEL, data urodzenia, stan cywilny, zdjęcie), portfolio.
- **Treść (18):** udział osiągnięć, kwantyfikacja, czasowniki akcji, język obowiązków, konkretność,
  samodzielność (ownership), skala, struktura STAR, dowody umiejętności (evidence-to-claim),
  głębokość umiejętności, ścieżka kariery, stabilność, luki, aktualność, wykształcenie, kursy,
  wystawy/nagrody/publikacje, profil.
- **Dopasowanie do oferty (21):** warunki konieczne (knock-out), pokrycie wymagań obowiązkowych,
  dodatkowych i zadań w skali 0-3 (brak / wzmianka / w doświadczeniu / z wynikiem), trafność
  doświadczenia (najlepsza rola: problem, kontekst, skala, samodzielność, przemnożone przez świeżość),
  zgodność umiejętności, narzędzi, technik, grup odbiorców, typu placówki, kwalifikacji formalnych,
  słowa kluczowe, podobieństwo semantyczne, BM25, wymagany staż, poziom stanowiska, dziedzina,
  świeżość umiejętności, języki, dopasowanie CV do oferty, lokalizacja.
- **List motywacyjny (9):** personalizacja, odniesienie do wymagań, spójność z CV, struktura,
  długość, konkrety, motywacja, ton i forma grzecznościowa, czytelność.

Na górze raportu z ofertą jest **przesiew rekrutera**, czyli decyzja *shortlista / dopytaj / odrzuć*.
Kroki idą w tej kolejności:

1. Warunki konieczne: twardy brak oznacza odrzucenie.
2. Wymagania obowiązkowe: średnia poniżej 2/3 albo choć jedno bez dowodu oznacza „dopytaj”,
   średnia poniżej 1,5 oznacza „odrzuć”.
3. Trafność najlepszej roli: poniżej 1,6/3 oznacza „odrzuć”, od 1,6 do 2,39 „dopytaj”.
4. Poziom stanowiska: dwa szczeble różnicy oznaczają „dopytaj”.

Decyzji towarzyszy jedno zdanie uzasadnienia.

Zasady, które warto znać przy czytaniu wyników:

- **„–” to nie zero.** Brak informacji w dokumencie nie obniża wyniku metryki, tylko pewność oceny.
- **Kompetencje miękkie i dokumenty** (empatia, niekaralność, Rejestr Sprawców, badania sanepid)
  nie są oceniane z CV. Raport oznacza je jako „do weryfikacji na rozmowie” albo „do dostarczenia”.
- **Niespełniony twardy warunek konieczny** ogranicza wynik ogólny do 40 i trafia na początek listy
  poprawek.
- **Wynik jest porównawczy.** Nie jest prawdopodobieństwem zaproszenia ani zatrudnienia.
- **Bez uprzedzeń.** Wiek, płeć, zdjęcie i stan cywilny nie wpływają na dopasowanie. Raport tylko
  informuje, że takie dane są w CV.

## Rozszerzanie

Słowniki są w YAML i nie wymagają zmian w kodzie:

- `src/cvscore/data/taxonomy.yaml`: umiejętności z odmianami PL/EN, w podziale na dziedziny;
- `src/cvscore/data/lexicons.yaml`: nagłówki sekcji, czasowniki akcji, frazesy, warunki konieczne,
  poziomy języków, zwroty w listach;
- `src/cvscore/data/profiles.yaml`: wagi i progi profili.

Własne wersje plików można trzymać poza repozytorium: wskaż katalog zmienną `CVSCORE_DATA`.

## Rozwój

```bash
.venv/bin/python -m pytest -q         # testy na syntetycznych PDF (generowanych w katalogu tymczasowym)
.venv/bin/pylint src/cvscore tools tests
.venv/bin/python tools/make_samples.py   # przykładowe PDF w samples/pdf i samples/batch
```

Wszystkie osoby i organizacje w `samples/` są fikcyjne. Prawdziwe CV trzymaj w `data/`, a raporty
w `reports/`. Oba katalogi są w `.gitignore`, bo zawierają dane osobowe.
