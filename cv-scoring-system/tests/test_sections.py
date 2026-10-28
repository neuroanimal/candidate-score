"""Sections, dates and the timeline."""

import datetime as dt

from cvscore.extract import from_text
from cvscore.sections import extract_contact, find_dates, gaps, match_heading, merged_months, split_sections, timeline

T = dt.date(2026, 10, 28)


def test_find_dates_variants():
    """Find dates variants."""
    cases = {
        "03.2019 – 06.2021": (dt.date(2019, 3, 1), dt.date(2021, 6, 1), False),
        "marzec 2019 - obecnie": (dt.date(2019, 3, 1), T, True),
        "III 2019 – nadal": (dt.date(2019, 3, 1), T, True),
        "2016-2018": (dt.date(2016, 1, 1), dt.date(2018, 12, 1), False),
        "od 2018": (dt.date(2018, 1, 1), T, True),
        "since 2020": (dt.date(2020, 1, 1), T, True),
        "09/2018 - 02/2021": (dt.date(2018, 9, 1), dt.date(2021, 2, 1), False),
    }
    for text, (start, end, ongoing) in cases.items():
        got = find_dates(text, T)
        assert got is not None, text
        assert (got[0], got[1], got[2]) == (start, end, ongoing), text


def test_single_year_only_at_edges():
    """Single year only at edges."""
    assert find_dates("2023 Wystawa indywidualna, Kraków", T)[3] is True
    assert find_dates("Koordynowałam 6 edycji festiwalu rodzinnego, ok. 2000 uczestników rocznie", T) is None


def test_headings():
    """Headings."""
    assert match_heading("DOŚWIADCZENIE ZAWODOWE") == "experience"
    assert match_heading("Doświadczenie:") == "experience"
    assert match_heading("DOŚWIADCZENIEZAWODOWE") == "experience"
    assert match_heading("KURSYISZKOLENIA") == "courses"
    assert match_heading("Wymagania:", "job_sections") == "requirements"
    assert match_heading("Mile widziane", "job_sections") == "nice_to_have"
    assert match_heading("Prowadziłam warsztaty plastyczne dla dzieci.") is None


LAYOUTS = {
    "inline": """Doświadczenie zawodowe
Instruktorka 03.2021 – obecnie
SDS Przykład
• Prowadziłam zajęcia dla 2 grup
Edukatorka 10.2016 – 06.2018
Galeria Przykład
• Prowadziłam warsztaty
""",
    "title_before": """Doświadczenie zawodowe
Instruktorka
SDS Przykład 03.2021 – obecnie
• Prowadziłam zajęcia dla 2 grup
Edukatorka
Galeria Przykład 10.2016 – 06.2018
• Prowadziłam warsztaty
""",
    "date_before": """Doświadczenie zawodowe
03.2021 – obecnie
Instruktorka, SDS Przykład
• Prowadziłam zajęcia dla 2 grup
10.2016 – 06.2018
Edukatorka, Galeria Przykład
• Prowadziłam warsztaty
""",
}


def test_timeline_layouts():
    """Timeline layouts."""
    for name, text in LAYOUTS.items():
        entries = timeline(split_sections(from_text(text)), T)
        jobs = [e for e in entries if e.section == "experience"]
        assert len(jobs) == 2, name
        assert "Instruktorka" in jobs[0].title, (name, jobs[0].title)
        assert "Edukatorka" in jobs[1].title, (name, jobs[1].title)
        assert any("2 grup" in ln for ln in jobs[0].lines), name


def test_merged_months_and_gaps():
    """Merged months and gaps."""
    text = "Doświadczenie\n01.2015 - 12.2016 A\n01.2016 - 06.2017 B\n01.2019 - 12.2019 C\n"
    jobs = timeline(split_sections(from_text(text)), T)
    assert merged_months(jobs) == 30 + 12
    found = gaps(jobs)
    assert len(found) == 1 and found[0][2] == 18


def test_contact():
    """Contact."""
    doc = from_text("Anna\ntel. +48 600 100 200 | anna@example.com\n00-001 Warszawa\nbehance.net/anna\n"
                    "linkedin.com/in/anna")
    c = extract_contact(doc)
    assert c.emails == ["anna@example.com"] and c.phones and c.city == "Warszawa"
    assert c.linkedin and any("behance" in u for u in c.portfolio)


def test_dates_with_day():
    """Europass and Polish long dates (day + month + year)."""
    assert find_dates("01/09/2018 – 28/02/2021 Warszawa", T)[:2] == (dt.date(2018, 9, 1), dt.date(2021, 2, 1))
    assert find_dates("15 marca 2021 – 30 czerwca 2022", T)[:2] == (dt.date(2021, 3, 1), dt.date(2022, 6, 1))
    assert find_dates("od 01.10.2019", T)[2] is True


def test_europass_labels(root):
    """Europass field labels are removed, entries are found."""
    from cvscore.extract import extract  # pylint: disable=import-outside-toplevel
    from cvscore.sections import strip_label  # pylint: disable=import-outside-toplevel

    assert strip_label("Zajmowane stanowisko: Instruktorka") == "Instruktorka"
    assert strip_label("Stanowisko kierownicze objęłam w 2020") == "Stanowisko kierownicze objęłam w 2020"
    jobs = [e for e in timeline(split_sections(extract(root / "samples/text/cv_europass.txt")), T)
            if e.section == "experience"]
    assert [j.title for j in jobs] == ["Specjalistka ds. social media", "Edukatorka"]
