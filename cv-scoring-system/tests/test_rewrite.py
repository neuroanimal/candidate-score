"""Rewrite drafts."""

from cvscore.rewrite import draft


def test_duty_to_action_feminine_and_masculine():
    """'Odpowiedzialna za przygotowanie' becomes an action with placeholders."""
    rw = draft("• Odpowiedzialna za przygotowanie materiałów dydaktycznych")
    assert rw.draft.startswith("Przygotowałam materiałów") or rw.draft.startswith("Przygotowałam")
    assert "[" in rw.draft
    assert draft("Odpowiedzialny za prowadzenie zajęć", "m").draft.startswith("Prowadziłem zajęć")


def test_weak_ownership():
    """'Pomagałam przy' becomes co-leading with the own part to fill in."""
    rw = draft("Pomagałam przy organizacji festiwalu")
    assert rw.draft.startswith("Współprowadziłam organizacji festiwalu") and "moja część" in rw.draft


def test_strong_bullet_untouched():
    """A bullet with action, number and result needs no draft."""
    assert draft("Zorganizowałam 3 wystawy prac uczestników, wernisaż dla 120 gości") is None
