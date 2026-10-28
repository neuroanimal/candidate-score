"""Normalisation and stemming."""

from cvscore import textnorm as tn


def test_fold_polish():
    """Fold polish."""
    assert tn.fold("Łódź, Świętokrzyska, ŻÓŁĆ") == "lodz, swietokrzyska, zolc"


def test_stem_inflection_and_gender_meet():
    """Stem inflection and gender meet."""
    pairs = [("warsztaty", "warsztatów"), ("prowadziłam", "prowadziłem"), ("zorganizowałam", "organizacja"),
             ("zaprojektowałam", "projektowanie"), ("opracowałam", "opracowywanie"), ("graficzka", "grafik"),
             ("instruktorka", "instruktora"), ("specjalistka", "specjalista"), ("pedagożka", "pedagoga"),
             ("uczestnicy", "uczestników"), ("terapeutka", "terapeuta")]
    for a, b in pairs:
        assert tn.stem(tn.fold(a)) == tn.stem(tn.fold(b)), (a, b)


def test_stem_keeps_nouns_with_prefix_letters():
    """Stem keeps nouns with prefix letters."""
    assert tn.stem("zespol") == "zespol"
    assert tn.stem(tn.fold("zajęcia")).startswith("zaj")


def test_collapse_spaced_letters():
    """Collapse spaced letters."""
    assert tn.collapse_spaced_letters("D O Ś W I A D C Z E N I E") == "DOŚWIADCZENIE"
    assert tn.collapse_spaced_letters("K U R S Y  I  S Z K O L E N I A") == "KURSY I SZKOLENIA"
    assert tn.collapse_spaced_letters("Kurs A B C") == "Kurs A B C"


def test_clean_text_repairs_pdf_artefacts():
    """Clean text repairs pdf artefacts."""
    raw = "Prowadziłam warsz-\ntaty\u00ad dla ﬁrm \ue001\n\n\n● Punkt"
    out = tn.clean_text(raw)
    assert "warsztaty" in out and "firm" in out and "\ue001" not in out
    assert "• Punkt" in out and "\n\n\n" not in out


def test_garbage_ratio():
    """Garbage ratio."""
    assert tn.garbage_ratio("Doświadczenie zawodowe w pracy z dziećmi") < 0.1
    assert tn.garbage_ratio("Ä‡Å‚Ã³ Ã³Å‚Ä‡ Å›Ä‡Ã³ Ã³Ä‡Å‚Ã³") > 0.3
    assert tn.garbage_ratio("") == 1.0


def test_syllables():
    """Syllables."""
    assert tn.syllables("warsztaty") == 3
    assert tn.syllables("niepełnosprawnością") >= 6


def test_sentences_split_lines_and_punctuation():
    """Sentences split lines and punctuation."""
    s = tn.sentences("• Pierwszy punkt\nDrugie zdanie. Trzecie zdanie!")
    assert s == ["Pierwszy punkt", "Drugie zdanie.", "Trzecie zdanie!"]


def test_taxonomy_aliases_are_unambiguous():
    """No two skills share an alias after stemming (a collision silently merges skills)."""
    from cvscore.lexicon import load_yaml, phrase_stems  # pylint: disable=import-outside-toplevel

    owners = {}
    for body in load_yaml("taxonomy").values():
        for entry in body["skills"]:
            for alias in entry["aliases"]:
                owners.setdefault(phrase_stems(str(alias)), set()).add(entry["name"])
    clashes = {k: v for k, v in owners.items() if len(v) > 1}
    assert not clashes, clashes
