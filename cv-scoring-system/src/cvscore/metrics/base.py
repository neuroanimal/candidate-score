"""Metric result type, the metric registry and score helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

# Metric groups and how they are presented
GROUPS = {
    "document": "Dokument i czytelność dla ATS",
    "content": "Treść i dowody",
    "fit": "Dopasowanie do oferty",
    "letter": "List motywacyjny",
}


@dataclass
class MetricResult:
    """Outcome of one metric.

    ``score`` is 0..100, or ``None`` when the metric could not be measured
    (*unknown* is deliberately different from 0: missing information in a CV is
    not proof that the candidate lacks the skill).
    """

    id: str
    label: str
    group: str
    score: float | None
    value: str = ""
    evidence: list[str] = field(default_factory=list)
    explanation: str = ""
    suggestion: str = ""
    weight: float = 1.0
    critical: bool = False  # e.g. a failed knock-out

    @property
    def status(self) -> str:
        """``good``, ``ok``, ``weak``, ``bad`` or ``unknown``."""
        if self.score is None:
            return "unknown"
        if self.score >= 80:
            return "good"
        if self.score >= 60:
            return "ok"
        if self.score >= 35:
            return "weak"
        return "bad"

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {"id": self.id, "label": self.label, "group": self.group,
                "score": None if self.score is None else round(self.score, 1),
                "status": self.status, "value": self.value, "weight": round(self.weight, 3),
                "evidence": self.evidence, "explanation": self.explanation,
                "suggestion": self.suggestion, "critical": self.critical}


# id -> (label, group, base weight, short description). Profile multipliers from
# profiles.yaml are applied on top of the base weight.
REGISTRY: dict[str, tuple[str, str, float, str]] = {
    # ---- document
    "text_extractability": ("Warstwa tekstowa PDF", "document", 1.5, "Czy tekst da się odczytać bez OCR i bez błędów kodowania"),
    "ats_parseability": ("Parsowalność przez ATS", "document", 1.5, "Sekcje, daty, kontakt i kolejność czytania rozpoznane automatycznie"),
    "layout_complexity": ("Złożoność układu", "document", 0.6, "Kolumny, grafiki, tekst w obrazach"),
    "section_coverage": ("Kompletność sekcji", "document", 1.2, "Obecność sekcji oczekiwanych dla profilu zawodowego"),
    "contact_completeness": ("Dane kontaktowe", "document", 1.0, "E-mail, telefon, miasto, LinkedIn/portfolio"),
    "length_fit": ("Długość dokumentu", "document", 0.7, "Liczba stron i słów względem stażu i profilu"),
    "readability": ("Czytelność tekstu", "document", 0.8, "Długość zdań i punktów, trudne słowa"),
    "scannability": ("Skanowalność (6-30 s)", "document", 1.0, "Punktory, nagłówki, stanowisko widoczne na górze"),
    "consistency": ("Spójność formy", "document", 0.8, "Formaty dat, kolejność chronologiczna, forma gramatyczna i rodzaj"),
    "noise_ratio": ("Szum i frazesy", "document", 0.8, "Puste frazesy, powtórzenia, nadmiar nieistotnych informacji"),
    "language_mechanics": ("Poprawność zapisu", "document", 0.6, "Polskie znaki, WERSALIKI, powtórzone słowa, mieszanie języków"),
    "rodo_clause": ("Klauzula RODO", "document", 1.0, "Aktualna klauzula zgody na przetwarzanie danych (rekrutacje w Polsce)"),
    "personal_data": ("Minimalizacja danych osobowych", "document", 0.6, "PESEL, data urodzenia, stan cywilny, zdjęcie"),
    "portfolio_presence": ("Portfolio / linki do prac", "document", 1.0, "Link do portfolio, Behance, Instagram, strony, GitHub"),
    # ---- content
    "achievement_ratio": ("Udział osiągnięć", "content", 1.4, "Punkty opisujące rezultat, a nie tylko obowiązek"),
    "quantification_ratio": ("Kwantyfikacja", "content", 1.2, "Punkty z liczbą: uczestnicy, edycje, %, budżet, zasięg"),
    "action_verbs": ("Czasowniki akcji", "content", 0.8, "Punkty zaczynające się od konkretnego działania"),
    "duty_language": ("Język obowiązków", "content", 0.6, "Ile punktów to 'odpowiedzialna za...' zamiast działania i efektu"),
    "specificity": ("Konkretność", "content", 1.0, "Nazwy własne, narzędzia, liczby w opisach"),
    "ownership": ("Samodzielność / ownership", "content", 1.0, "Prowadziłam, koordynowałam vs pomagałam, uczestniczyłam"),
    "scope_signals": ("Skala działań", "content", 1.0, "Wielkość grup, liczba uczestników, budżety, zespoły"),
    "star_structure": ("Struktura działanie-kontekst-wynik", "content", 0.8, "Punkty z działaniem, kontekstem i rezultatem (STAR)"),
    "skill_evidence": ("Dowody umiejętności", "content", 1.3, "Umiejętności z listy potwierdzone w doświadczeniu (evidence-to-claim)"),
    "skill_depth": ("Głębokość umiejętności", "content", 0.8, "Wymieniona / użyta w pracy / użyta w wielu rolach"),
    "career_trajectory": ("Ścieżka kariery", "content", 0.6, "Rosnący zakres odpowiedzialności"),
    "tenure_stability": ("Stabilność zatrudnienia", "content", 0.6, "Mediana stażu, bardzo krótkie epizody"),
    "gap_transparency": ("Luki w zatrudnieniu", "content", 0.5, "Przerwy powyżej 6 miesięcy i ich wyjaśnienie"),
    "recency": ("Aktualność doświadczenia", "content", 0.8, "Czy ostatnia aktywność zawodowa jest świeża"),
    "education_strength": ("Wykształcenie", "content", 0.7, "Stopnie, kierunki, uczelnie artystyczne/pedagogiczne"),
    "courses_certificates": ("Kursy i certyfikaty", "content", 0.7, "Liczba i aktualność szkoleń"),
    "exhibitions_awards": ("Wystawy, nagrody, publikacje", "content", 0.8, "Dorobek: wystawy, konkursy, plenery, publikacje"),
    "summary_quality": ("Profil / podsumowanie", "content", 0.6, "Krótkie podsumowanie z zawodem i specjalizacją, bez frazesów"),
    # ---- fit
    "knockouts": ("Warunki konieczne (knock-out)", "fit", 2.0, "Prawo jazdy, kwalifikacje, wykształcenie, portfolio..."),
    "must_have_coverage": ("Pokrycie wymagań obowiązkowych", "fit", 3.0, "Każde wymaganie ocenione 0-3: brak/wzmianka/zakres/wynik"),
    "nice_to_have_coverage": ("Pokrycie wymagań dodatkowych", "fit", 1.0, "Mile widziane"),
    "duty_coverage": ("Pokrycie zadań ze stanowiska", "fit", 1.5, "Czy kandydatka robiła już to, co opisuje 'zakres obowiązków'"),
    "experience_relevance": ("Trafność doświadczenia", "fit", 2.5, "Najlepsza rola: problem, kontekst, skala, samodzielność x świeżość"),
    "skill_match": ("Zgodność umiejętności", "fit", 1.5, "Umiejętności z oferty znalezione w CV"),
    "tool_match": ("Zgodność narzędzi", "fit", 1.0, "Programy i platformy z oferty"),
    "technique_match": ("Zgodność technik", "fit", 1.0, "Techniki artystyczne/graficzne z oferty"),
    "population_match": ("Grupy odbiorców", "fit", 1.0, "Dzieci, dorośli, seniorzy, osoby z niepełnosprawnościami, choroby neurologiczne, zaburzenia psychiczne"),
    "setting_match": ("Typ placówki", "fit", 0.6, "ŚDS, WTZ, DPS, fundacja, szpital - zgodność środowiska pracy"),
    "qualification_match": ("Kwalifikacje formalne", "fit", 1.2, "Wymagane wykształcenie/kursy potwierdzone w sekcji edukacji"),
    "keyword_coverage": ("Słowa kluczowe oferty", "fit", 1.0, "Najważniejsze terminy ogłoszenia obecne w CV (ATS)"),
    "semantic_similarity": ("Podobieństwo semantyczne", "fit", 1.0, "TF-IDF słów i n-gramów znakowych, embeddingi jeśli dostępne"),
    "bm25_relevance": ("Trafność BM25 wymagań", "fit", 0.8, "Najlepsze dopasowanie fragmentu CV do każdego wymagania"),
    "years_requirement": ("Wymagany staż", "fit", 1.2, "Lata trafnego doświadczenia vs wymagane"),
    "seniority_fit": ("Poziom stanowiska", "fit", 1.0, "Junior / mid / senior / lead - zgodność z ofertą"),
    "domain_alignment": ("Zgodność dziedziny", "fit", 1.0, "Rozkład dziedzin w CV vs w ofercie"),
    "skill_recency": ("Świeżość wymaganych umiejętności", "fit", 0.8, "Wymagane umiejętności użyte w ostatnich 3 latach"),
    "language_requirements": ("Języki obce", "fit", 1.0, "Poziom wymagany vs deklarowany"),
    "tailoring": ("Dopasowanie CV do oferty", "fit", 0.8, "Tytuł/profil pod tę rolę, bez upychania słów kluczowych"),
    "location_match": ("Lokalizacja", "fit", 0.3, "Miasto z oferty vs miasto w CV"),
    # ---- letter
    "letter_personalization": ("Personalizacja", "letter", 1.5, "Nazwa organizacji, stanowisko, adresat"),
    "letter_jd_alignment": ("Odniesienie do wymagań", "letter", 1.5, "Ile wymagań obowiązkowych list adresuje"),
    "letter_cv_consistency": ("Spójność z CV", "letter", 1.0, "Deklaracje z listu potwierdzone w CV, list nie jest kopią CV"),
    "letter_structure": ("Struktura listu", "letter", 1.0, "Powitanie, wstęp ze stanowiskiem, rozwinięcie, zakończenie, CTA, podpis"),
    "letter_length": ("Długość listu", "letter", 0.6, "Ok. 180-450 słów, 1 strona"),
    "letter_specificity": ("Konkrety w liście", "letter", 1.2, "Przykłady, liczby, nazwy projektów zamiast ogólników"),
    "letter_motivation": ("Motywacja", "letter", 1.0, "Dlaczego ta organizacja i ta rola"),
    "letter_tone": ("Ton i forma grzecznościowa", "letter", 0.6, "Pan/Pani/Państwo wielką literą, formalne zakończenie"),
    "letter_readability": ("Czytelność listu", "letter", 0.5, "Długość zdań i akapitów"),
}


def make(metric_id: str, score: float | None, **kwargs) -> MetricResult:
    """Create a MetricResult with label and group from the registry."""
    label, group, _, desc = REGISTRY[metric_id]
    kwargs.setdefault("explanation", desc)
    if score is not None:
        score = max(0.0, min(100.0, float(score)))
    return MetricResult(metric_id, label, group, score, **kwargs)


def ramp(value: float, low: float, high: float) -> float:
    """Linear 0..100 between ``low`` (0) and ``high`` (100); works for decreasing ramps too."""
    if high == low:
        return 100.0 if value >= high else 0.0
    frac = (value - low) / (high - low)
    return max(0.0, min(100.0, 100.0 * frac))


def band(value: float, good_low: float, good_high: float, zero_low: float, zero_high: float) -> float:
    """100 inside [good_low, good_high], falling linearly to 0 at zero_low / zero_high."""
    if good_low <= value <= good_high:
        return 100.0
    if value < good_low:
        return ramp(value, zero_low, good_low)
    return ramp(value, zero_high, good_high)


def pct(x: float) -> str:
    """Format a 0..1 ratio as a percentage."""
    return f"{x * 100:.0f}%"


def short(text: str, limit: int = 110) -> str:
    """Shorten a text for evidence lists."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
