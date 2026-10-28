"""Generate synthetic sample documents that reproduce real-world PDF problems.

The CVs imitate an Adobe InDesign export: two columns with a shaded sidebar,
letter-spaced headings, bullets placed as separate glyphs, a photo, icon links,
and a variant with all text converted to outlines (image only -> needs OCR).
All persons and organisations are fictitious.

    python tools/make_samples.py            # writes samples/pdf and samples/batch
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
A4 = pymupdf.paper_rect("a4")


class Writer:
    """Tiny layout helper on top of PyMuPDF with Polish-capable fonts."""

    def __init__(self, creator: str = "Adobe InDesign 19.2 (Windows)"):
        self.doc = pymupdf.open()
        self.doc.set_metadata({"creator": creator, "producer": "Adobe PDF Library 17.0", "title": "CV"})
        self.page = None
        self.new_page()

    def new_page(self):
        """Add an A4 page and register fonts."""
        self.page = self.doc.new_page(width=A4.width, height=A4.height)
        self.page.insert_font(fontname="dj", fontfile=FONT)
        self.page.insert_font(fontname="djb", fontfile=FONT_B)
        return self.page

    def text(self, x, y, s, size=9.5, bold=False, color=(0.1, 0.1, 0.1)):
        """Insert a single line."""
        self.page.insert_text((x, y), s, fontname="djb" if bold else "dj", fontsize=size, color=color)

    def wrap(self, x, y, s, width, size=9.5, bold=False, lead=1.35, bullet=False):
        """Insert wrapped text; returns the next y. Bullets are separate glyphs like in InDesign."""
        font = pymupdf.Font(fontfile=FONT_B if bold else FONT)
        words, line, lines = s.split(), "", []
        indent = 9 if bullet else 0
        for w in words:
            cand = (line + " " + w).strip()
            if font.text_length(cand, fontsize=size) > width - indent:
                lines.append(line)
                line = w
            else:
                line = cand
        if line:
            lines.append(line)
        for i, ln in enumerate(lines):
            if bullet and i == 0:
                self.text(x, y, "•", size, False)
            self.text(x + indent, y, ln, size, bold)
            y += size * lead
        return y

    def spaced_heading(self, x, y, s, size=11, color=(0.55, 0.2, 0.3)):
        """Letter-spaced heading drawn glyph by glyph (tracking), as designers do."""
        font = pymupdf.Font(fontfile=FONT_B)
        for ch in s:
            self.text(x, y, ch, size, True, color)
            x += font.text_length(ch, fontsize=size) + 2.6
        return y + size * 1.8

    def link(self, rect, uri):
        """Clickable area (e.g. behind an icon)."""
        self.page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(*rect), "uri": uri})

    def photo(self, rect):
        """A placeholder portrait image."""
        img = Image.new("RGB", (300, 380), (205, 190, 180))
        d = ImageDraw.Draw(img)
        d.ellipse((90, 60, 210, 190), fill=(150, 120, 110))
        d.rectangle((60, 210, 240, 380), fill=(120, 95, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.page.insert_image(pymupdf.Rect(*rect), stream=buf.getvalue())

    def save(self, path: Path):
        """Write the PDF."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.subset_fonts()
        self.doc.save(str(path), garbage=3, deflate=True)
        self.doc.close()


# ----------------------------------------------------------------- content

PERSON = {"name": "Anna Przykładowa", "tagline": "Malarka · graficzka · edukatorka plastyczna",
          "email": "anna.przykladowa@example.com", "phone": "600 100 200", "city": "00-001 Warszawa",
          "www": "www.anna-przykladowa.example.pl", "ig": "instagram.com/anna.przykladowa.art"}

JOBS_THERAPY = [
    ("03.2021 – obecnie", "Instruktorka terapii zajęciowej – pracownia plastyczna",
     "Środowiskowy Dom Samopomocy „Przykład”, Warszawa",
     ["Prowadziłam zajęcia plastyczne i arteterapeutyczne dla 2 grup po 10 uczestników z zaburzeniami psychicznymi i chorobami neurologicznymi",
      "Opracowałam autorski program arteterapii z elementami grafiki warsztatowej (linoryt, monotypia)",
      "Zorganizowałam 3 wystawy prac uczestników, w tym wernisaż dla 120 gości",
      "Współtworzyłam indywidualne plany postępowania wspierająco-aktywizującego dla 20 osób"]),
    ("09.2018 – 02.2021", "Specjalistka ds. social media i grafiki", "Dom Kultury „Przykładowy”, Warszawa",
     ["Prowadziłam profile na Facebooku i Instagramie – liczba obserwujących wzrosła z 1,2 tys. do 4,5 tys.",
      "Projektowałam plakaty, ulotki i grafiki do social media w Photoshopie, Illustratorze i InDesignie",
      "Koordynowałam 6 edycji festiwalu rodzinnego, ok. 2000 uczestników rocznie"]),
    ("10.2016 – 06.2018", "Edukatorka", "Galeria Sztuki „Przykład”, Warszawa",
     ["Prowadziłam warsztaty plastyczne dla dzieci w wieku 5–12 lat (ok. 40 zajęć w roku)",
      "Odpowiedzialna za przygotowanie materiałów dydaktycznych"]),
]
EDU = [("2011 – 2016", "Akademia Sztuk Pięknych w Warszawie", "Wydział Grafiki, magister sztuki"),
       ("2019 – 2020", "Studia podyplomowe: Arteterapia", "Uniwersytet Przykładowy")]
COURSES = ["2022 Kurs pierwszej pomocy", "2020 Trening umiejętności społecznych – 40 h",
           "2019 Przygotowanie pedagogiczne – kurs kwalifikacyjny"]
EXHIB = ["2023 Wystawa indywidualna „Ślady”, Galeria Przykładowa, Kraków",
         "2021 Wystawa zbiorowa grafiki, Muzeum Przykładowe", "2019 Nagroda w konkursie Grafika Roku"]
SKILLS = ["Adobe Photoshop, Illustrator, InDesign", "Canva, Procreate", "Malarstwo olejne i akrylowe, akwarela",
          "Linoryt, sitodruk, monotypia", "Meta Business Suite, planowanie treści"]
RODO = ("Wyrażam zgodę na przetwarzanie moich danych osobowych dla potrzeb niezbędnych do realizacji procesu "
        "rekrutacji zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady (UE) 2016/679 z dnia 27 kwietnia "
        "2016 r. (RODO).")


def cv_two_column(path: Path, jobs=None, tagline=None, photo=True, links=True):
    """Two-column InDesign-like CV."""
    jobs = jobs or JOBS_THERAPY
    w = Writer()
    pg = w.page
    side = 190
    pg.draw_rect(pymupdf.Rect(0, 0, side, A4.height), color=None, fill=(0.95, 0.92, 0.9))
    if photo:
        w.photo((35, 36, 155, 188))
    y = 215
    y = w.spaced_heading(20, y, "KONTAKT")
    for line in (PERSON["phone"], PERSON["email"], PERSON["city"], PERSON["www"], PERSON["ig"]):
        y = w.wrap(20, y, line, side - 35, 8.5)
    if links:
        w.link((20, y - 30, 170, y - 18), "https://www.anna-przykladowa.example.pl")
        w.link((150, 20, 175, 35), "https://www.behance.net/anna-przykladowa")  # icon-only link
    y += 12
    y = w.spaced_heading(20, y, "UMIEJĘTNOŚCI")
    for s in SKILLS:
        y = w.wrap(20, y, s, side - 35, 8.5, bullet=True)
    y += 12
    y = w.spaced_heading(20, y, "JĘZYKI")
    y = w.wrap(20, y, "Angielski – B2", side - 35, 8.5)
    y = w.wrap(20, y, "Prawo jazdy kat. B", side - 35, 8.5)
    # main column
    x, width = side + 22, A4.width - side - 45
    w.text(x, 58, PERSON["name"], 24, True, (0.25, 0.15, 0.2))
    w.text(x, 80, tagline or PERSON["tagline"], 11, False, (0.4, 0.4, 0.4))
    y = 112
    y = w.spaced_heading(x, y, "PROFIL")
    y = w.wrap(x, y, "Malarka i graficzka po ASP. Od 8 lat prowadzę warsztaty plastyczne dla dzieci, dorosłych i seniorów, "
               "także dla osób z niepełnosprawnością intelektualną i chorobami neurologicznymi.", width) + 10
    y = w.spaced_heading(x, y, "DOŚWIADCZENIE ZAWODOWE")
    for dates, title, org, bullets in jobs:
        if y > A4.height - 120:
            w.new_page()
            y = 60
        w.text(x, y, title, 10, True)
        y += 13
        # dates right-aligned on the organisation line, in grey - a typical designed layout
        w.text(x, y, org, 9, False, (0.35, 0.35, 0.35))
        w.text(A4.width - 25 - pymupdf.Font(fontfile=FONT).text_length(dates, fontsize=8.5), y, dates, 8.5, False,
               (0.45, 0.45, 0.45))
        y += 14
        for b in bullets:
            y = w.wrap(x, y, b, width, 9, bullet=True)
        y += 8
    if y > A4.height - 230:
        w.new_page()
        y = 60
    y = w.spaced_heading(x, y, "WYKSZTAŁCENIE")
    for dates, school, detail in EDU:
        y = w.wrap(x, y, f"{dates}  {school}", width, 9.5, bold=True)
        y = w.wrap(x, y, detail, width, 9) + 4
    y = w.spaced_heading(x, y + 6, "KURSY I SZKOLENIA")
    for c in COURSES:
        y = w.wrap(x, y, c, width, 9)
    y = w.spaced_heading(x, y + 8, "WYSTAWY")
    for e in EXHIB:
        y = w.wrap(x, y, e, width, 9)
    w.wrap(x, A4.height - 52, RODO, width, 6.5, lead=1.25)
    w.save(path)


def outlines_variant(src: Path, dst: Path, dpi: int = 300):
    """Simulate 'Create Outlines': every page becomes an image without a text layer."""
    s = pymupdf.open(str(src))
    d = pymupdf.open()
    d.set_metadata({"creator": "Adobe InDesign 19.2 (Windows)", "producer": "Adobe PDF Library 17.0"})
    for page in s:
        pix = page.get_pixmap(dpi=dpi)
        new = d.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("jpeg", jpg_quality=85))
    d.save(str(dst))
    s.close()
    d.close()


def _painting(seed: int) -> bytes:
    """A colourful abstract 'artwork' for portfolio pages."""
    import random  # pylint: disable=import-outside-toplevel

    rnd = random.Random(seed)
    img = Image.new("RGB", (900, 700), tuple(rnd.randint(150, 250) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(60):
        x, y = rnd.randint(0, 900), rnd.randint(0, 700)
        r = rnd.randint(20, 160)
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rnd.randint(0, 255) for _ in range(3)))
    for _ in range(25):
        d.line([(rnd.randint(0, 900), rnd.randint(0, 700)) for _ in range(2)], fill=(20, 20, 20), width=rnd.randint(2, 9))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def with_portfolio(src: Path, dst: Path, pages: int = 2):
    """Append picture pages with short captions, like an artist's CV with a portfolio appendix."""
    doc = pymupdf.open(str(src))
    for i in range(pages):
        page = doc.new_page(width=A4.width, height=A4.height)
        page.insert_font(fontname="dj", fontfile=FONT)
        page.insert_image(pymupdf.Rect(40, 60, A4.width - 40, 470), stream=_painting(i))
        page.insert_image(pymupdf.Rect(40, 490, A4.width - 40, A4.height - 90), stream=_painting(i + 10))
        page.insert_text((40, A4.height - 60), f"Bez tytułu {i + 1}, olej na płótnie, 2023", fontname="dj", fontsize=8)
    doc.subset_fonts()
    doc.save(str(dst), garbage=3, deflate=True)
    doc.close()


def simple_doc(path: Path, lines: list[tuple[str, float, bool]], creator="Microsoft Word"):
    """Single-column document from (text, size, bold) lines; '' = blank line."""
    w = Writer(creator)
    y = 60
    for text, size, bold in lines:
        if not text:
            y += size
            continue
        if y > A4.height - 60:
            w.new_page()
            y = 60
        bullet = text.startswith("• ")
        y = w.wrap(50, y, text[2:] if bullet else text, A4.width - 100, size, bold, bullet=bullet)
    w.save(path)


def lines_from(text: str, heading_words=()) -> list[tuple[str, float, bool]]:
    """Plain text to (text, size, bold) with headings bold."""
    out = []
    for i, ln in enumerate(text.strip().split("\n")):
        s = ln.strip()
        if not s:
            out.append(("", 6, False))
        elif i == 0:
            out.append((s, 15, True))
        elif s.rstrip(":").lower() in heading_words or s.endswith(":"):
            out.append((s, 11, True))
        else:
            out.append((s, 10, False))
    return out


OFFERS = {
    "oferta_sds_instruktor.pdf": (ROOT / "samples/text/oferta_sds.txt").read_text(encoding="utf-8"),
    "oferta_grafik_social_media.pdf": """Grafik / specjalista ds. social media
Centrum Kultury Przykładowe, Warszawa (praca hybrydowa)

Zakres obowiązków:
• projektowanie materiałów promocyjnych: plakaty, ulotki, banery, grafiki do social media
• prowadzenie profili na Facebooku, Instagramie i TikToku
• przygotowywanie harmonogramu publikacji i raportów zasięgów
• przygotowanie plików do druku

Wymagania:
• min. 2 lata doświadczenia na podobnym stanowisku
• bardzo dobra znajomość Adobe Photoshop, Illustrator, InDesign
• doświadczenie w prowadzeniu mediów społecznościowych instytucji
• portfolio z przykładowymi pracami
• znajomość języka angielskiego na poziomie B1

Mile widziane:
• montaż krótkich form wideo (Reels, TikTok)
• doświadczenie w kampaniach płatnych Meta Ads
• znajomość Canvy

Oferujemy:
• umowę o pracę, pracę w zespole kreatywnym
""",
    "oferta_animator_dzieci.pdf": """Instruktorka zajęć plastycznych dla dzieci
Dom Kultury Przykładowy, Kraków

Twoje zadania:
• prowadzenie zajęć plastycznych dla dzieci w wieku 4-12 lat
• przygotowywanie scenariuszy zajęć i materiałów
• organizacja wystaw prac i pikników rodzinnych

Wymagania:
• wykształcenie artystyczne lub pedagogiczne
• przygotowanie pedagogiczne
• doświadczenie w pracy z dziećmi
• kreatywność i cierpliwość

Mile widziane:
• kurs wychowawcy wypoczynku
• prawo jazdy kat. B

Wymagane dokumenty: CV, zaświadczenie z Krajowego Rejestru Karnego, informacja z Rejestru Sprawców Przestępstw na Tle Seksualnym.
""",
    "oferta_koordynator_wydarzen.pdf": """Koordynator / koordynatorka wydarzeń kulturalnych
Fundacja Festiwalowa Przykład, Gdańsk

Obowiązki:
• planowanie i koordynacja festiwali, wernisaży i konferencji
• zarządzanie budżetem wydarzeń i rozliczanie projektów
• współpraca ze sponsorami i partnerami
• pisanie wniosków grantowych (NCK, MKiDN)
• koordynacja wolontariuszy

Wymagania:
• min. 3 lata doświadczenia w organizacji wydarzeń
• doświadczenie w rozliczaniu projektów grantowych
• bardzo dobra organizacja pracy
• prawo jazdy kat. B
• język angielski C1

Oferujemy:
• umowę B2B lub umowę o pracę
""",
    "oferta_automation_developer.pdf": """Senior Automation Developer
Example Tech Sp. z o.o., Kraków (remote)

Responsibilities:
• design and build CI/CD pipelines in Jenkins
• automate processes in Python and Bash
• develop REST API integrations

Requirements:
• min. 5 years of experience in software development
• very good Python, SQL, Git, Docker, Kubernetes
• experience with Jenkins and CI/CD
• English B2

Nice to have:
• Power Automate, Azure
• machine learning experience
""",
}

WEB_OFFER = [  # a job-board page "printed to PDF": menu, cookies, the offer, then similar offers
    ("praca.example.pl   Oferty pracy   Pracodawcy   Zaloguj", 8, False),
    ("Ta strona używa plików cookies. Akceptuję", 7, False),
    ("", 10, False),
    ("Edukatorka muzealna – warsztaty dla seniorów", 20, True),
    ("Muzeum Sztuki Przykładowej, Łódź", 11, False),
    ("umowa o pracę · pełny etat · specjalista", 9, False),
    ("", 8, False),
    ("Twój zakres obowiązków", 13, True),
    ("• prowadzenie warsztatów plastycznych dla seniorów i osób z demencją", 10, False),
    ("• przygotowywanie scenariuszy zajęć i materiałów dydaktycznych", 10, False),
    ("Nasze wymagania", 13, True),
    ("• doświadczenie w prowadzeniu warsztatów dla dorosłych lub seniorów", 10, False),
    ("• wykształcenie wyższe artystyczne lub z zakresu historii sztuki", 10, False),
    ("• umiejętność pracy z grupą", 10, False),
    ("Mile widziane", 13, True),
    ("• doświadczenie w pracy z osobami z chorobą Alzheimera", 10, False),
    ("To oferujemy", 13, True),
    ("• stabilne zatrudnienie, kartę sportową", 10, False),
    ("Podobne oferty", 13, True),
    ("Grafik z portfolio – prawo jazdy kat. B wymagane", 10, False),
    ("Kierowca kat. C", 10, False),
    ("Klauzula informacyjna", 11, True),
    ("Administratorem danych osobowych jest Muzeum Sztuki Przykładowej. Dane będą przetwarzane w celu rekrutacji.", 8, False),
]

LETTER_SDS = (ROOT / "samples/text/list_sds.txt").read_text(encoding="utf-8")
LETTER_GRAFIK = """Anna Przykładowa
Warszawa, 12 października 2026

Szanowni Państwo,

w odpowiedzi na ogłoszenie Centrum Kultury Przykładowego zgłaszam swoją kandydaturę na stanowisko grafika i specjalisty ds. social media.

Przez ponad dwa lata prowadziłam profile Domu Kultury na Facebooku i Instagramie – liczba obserwujących wzrosła z 1,2 tys. do 4,5 tys. Projektowałam plakaty, ulotki i grafiki do social media w Photoshopie, Illustratorze i InDesignie oraz przygotowywałam pliki do druku.

Cenię Państwa program wydarzeń dla rodzin i chciałabym współtworzyć jego komunikację wizualną. Moje portfolio jest dostępne na stronie www.anna-przykladowa.example.pl.

Chętnie porozmawiam o tym, jak mogę wesprzeć Państwa zespół.

Z wyrazami szacunku
Anna Przykładowa
"""

CV_TECH = """Jan Przykładowy
Automation Developer · Software Architect · Applied Data Scientist
Kraków | +48 600 200 300 | jan.przykladowy@example.com | github.com/jan-przykladowy | linkedin.com/in/jan-przykladowy

Profil
Automation developer z 12-letnim doświadczeniem w telekomunikacji (5G Cloud Core). Projektuję pipeline'y CI/CD i narzędzia w Pythonie, które skróciły generowanie pakietów konfiguracyjnych z 3 dni do 4 godzin.

Doświadczenie zawodowe
01.2019 – obecnie
Senior Automation Developer / Software Architect, Example Telecom, Kraków
• Zaprojektowałem i zbudowałem pipeline Jenkins (Groovy) generujący pakiety dla 5 produktów i 27 wersji
• Zautomatyzowałem walidację schematów JSON Schema i modeli YANG – 0 błędów ładowania w 40 kolejnych wydaniach
• Prowadziłem zespół 4 osób i mentorowałem 2 juniorów
• Wdrożyłem testy GUI w Robot Framework z Playwright – czas suite skrócony z 388 s do 173 s

06.2014 – 12.2018
Software Engineer, Example Software House, Kraków
• Rozwijałem usługi REST API w Pythonie i Javie dla 3 klientów
• Zbudowałem moduł analizy danych (pandas, scikit-learn) prognozujący obciążenie, błąd MAPE 8%

Wykształcenie
2009 – 2014 Politechnika Przykładowa, Informatyka, magister inżynier

Umiejętności
Python, SQL, Bash, Groovy, Jenkins, Docker, Kubernetes, Git, REST API, JSON Schema, YANG, pandas, scikit-learn, Azure, Power Automate

Języki
Angielski – C1

Wyrażam zgodę na przetwarzanie moich danych osobowych dla potrzeb niezbędnych do realizacji procesu rekrutacji zgodnie z Rozporządzeniem Parlamentu Europejskiego i Rady (UE) 2016/679 (RODO).
"""

JOBS_GRAFIK = [JOBS_THERAPY[1], JOBS_THERAPY[0], JOBS_THERAPY[2]]


def main():
    """Write all samples."""
    pdf = ROOT / "samples" / "pdf"
    pdf.mkdir(parents=True, exist_ok=True)
    cv_two_column(pdf / "cv_anna_2kolumny_indesign.pdf")
    cv_two_column(pdf / "cv_anna_grafik.pdf", jobs=JOBS_GRAFIK, tagline="Graficzka i specjalistka ds. social media",
                  photo=False)
    outlines_variant(pdf / "cv_anna_2kolumny_indesign.pdf", pdf / "cv_anna_krzywe_bez_tekstu.pdf")
    with_portfolio(pdf / "cv_anna_2kolumny_indesign.pdf", pdf / "cv_anna_z_portfolio.pdf")
    for name, text in OFFERS.items():
        simple_doc(pdf / name, lines_from(text), creator="Chrome PDF")
    simple_doc(pdf / "oferta_strona_www_muzeum.pdf", WEB_OFFER, creator="Chrome PDF")
    simple_doc(pdf / "list_anna_sds.pdf", lines_from(LETTER_SDS))
    simple_doc(pdf / "list_anna_grafik.pdf", lines_from(LETTER_GRAFIK))
    simple_doc(pdf / "cv_jan_tech.pdf", lines_from(CV_TECH, ("profil", "doświadczenie zawodowe", "wykształcenie",
                                                              "umiejętności", "języki")))
    batch = ROOT / "samples" / "batch"
    if batch.exists():
        shutil.rmtree(batch)
    plan = [
        ("2026-09_sds", "cv_anna_2kolumny_indesign.pdf", "oferta_sds_instruktor.pdf", "list_anna_sds.pdf", "zaproszenie na rozmowę"),
        ("2026-09_centrum_kultury", "cv_anna_grafik.pdf", "oferta_grafik_social_media.pdf", "list_anna_grafik.pdf", "rozmowa"),
        ("2026-10_dom_kultury", "cv_anna_krzywe_bez_tekstu.pdf", "oferta_animator_dzieci.pdf", None, "brak odpowiedzi"),
        ("2026-10_festiwal", "cv_anna_grafik.pdf", "oferta_koordynator_wydarzen.pdf", None, "odrzucona"),
        ("2026-10_tech", "cv_jan_tech.pdf", "oferta_automation_developer.pdf", None, None),
    ]
    for app, cv, job, letter, outcome in plan:
        d = batch / app
        d.mkdir(parents=True)
        shutil.copy(pdf / cv, d / ("CV_" + cv))
        shutil.copy(pdf / job, d / job)
        if letter:
            shutil.copy(pdf / letter, d / letter)
        if outcome:
            (d / "wynik.txt").write_text(outcome + "\n", encoding="utf-8")
    print(f"Zapisano przykłady w {pdf} i {batch}")


if __name__ == "__main__":
    main()
