"""Job offer parsing."""

from cvscore.analysis import parse_job
from cvscore.extract import from_text

OFFER = """Grafik / specjalista ds. social media
Centrum Kultury Przykładowe, Warszawa

Zakres obowiązków:
• projektowanie plakatów i grafik do social media
• prowadzenie profili na Facebooku i Instagramie

Wymagania:
• min. 2 lata doświadczenia na podobnym stanowisku
• bardzo dobra znajomość Adobe Photoshop, Illustrator, InDesign
• portfolio
• znajomość języka angielskiego na poziomie B1
• prawo jazdy kat. B mile widziane

Mile widziane:
• montaż wideo

Wymagane dokumenty: CV, zaświadczenie o niekaralności.
"""


def test_parse_job_structure():
    """Parse job structure."""
    job = parse_job(from_text(OFFER))
    assert job.title.startswith("Grafik")
    assert "Centrum Kultury" in job.organisation
    assert len(job.duties) == 2 and len(job.nice) == 2  # "prawo jazdy ... mile widziane" downgraded
    assert job.years_required == 2
    assert job.languages == {"angielski": "B1"}
    assert {"portfolio_required", "driving_licence", "no_criminal_record"} <= set(job.knockouts)
    assert job.knockouts["driving_licence"]["soft"] is True
    assert job.knockouts["no_criminal_record"]["document_only"] is True
    assert "Adobe Photoshop" in job.skills(("must",))
    assert max(job.domains, key=job.domains.get) in ("graphic_design", "social_media")


def test_unstructured_offer_still_gives_requirements():
    """Unstructured offer still gives requirements."""
    job = parse_job(from_text("Szukamy osoby do prowadzenia warsztatów plastycznych dla seniorów. "
                              "Wymagamy doświadczenia w pracy z grupą."))
    assert job.requirements and "Praca z seniorami" in job.skills()


def test_public_sector_offer(root):
    """Numbered lists, inline headings and ';'-separated requirements (BIP style)."""
    from cvscore.extract import extract  # pylint: disable=import-outside-toplevel

    job = parse_job(extract(root / "samples/text/oferta_bip.txt"))
    assert job.title == "terapeuta zajęciowy"
    assert job.organisation.startswith("Środowiskowego Domu Samopomocy")
    assert job.seniority is None and job.years_required == 2
    assert len(job.must) == 3 and len(job.nice) == 3 and len(job.duties) == 1
    assert job.knockouts["no_criminal_record"]["document_only"]
