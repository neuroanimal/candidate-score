"""Job-fit metrics: knock-outs, requirement coverage with evidence levels,
experience relevance per role, skills by kind, seniority, languages, keywords and
semantic similarity."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .. import similarity, textnorm
from ..analysis import LEVELS, CVAnalysis, JobOffer, Requirement
from ..lexicon import lexicons, phrase_set, regex_any, taxonomy
from ..profiles import Profile
from ..sections import KNOWN_CITIES, Entry, merged_months
from .base import MetricResult, make, pct, ramp, short
from .content import _SCOPE, _DATEISH, _is_result, _numbers

USED_SECTIONS = ("experience", "volunteering", "portfolio", "exhibitions", "awards", "publications")
CLAIM_SECTIONS = ("skills", "summary", "header", "languages", "interests")
LEVEL_NAMES = {0: "brak dowodu", 1: "wzmianka", 2: "w doświadczeniu", 3: "w doświadczeniu z wynikiem"}


@dataclass
class Evidence:
    """Evidence for one requirement."""

    req: Requirement
    level: int | None  # 0-3, None = not verifiable from a CV (soft skill, document)
    line: str
    how: str


class FitContext:
    """Pre-computed structures shared by fit metrics (built once per CV/offer pair)."""

    def __init__(self, cv: CVAnalysis, job: JobOffer, profile: Profile | None = None):
        self.cv, self.job = cv, job
        targets = profile.targets if profile is not None else {}
        # how fast experience "ages": tech stacks age faster than artistic or pedagogical craft
        self.fresh_mid = float(targets.get("fresh_mid", 0.7))
        self.fresh_old = float(targets.get("fresh_old", 0.4))
        self.recent_months = int(targets.get("recency_months", 36))
        self.cv_lines: list[tuple[str, str, Entry | None]] = []
        for sec in cv.sections:
            for line in sec.lines:
                entry = next((e for e in cv.entries if e.section == sec.key and (line == e.title or line in e.lines)), None)
                self.cv_lines.append((line, sec.key, entry))
        self.index = similarity.LineIndex([ln for ln, _, _ in self.cv_lines])
        self.line_meta = {ln: (sec, entry) for ln, sec, entry in self.cv_lines}
        self.skill_where: dict[str, list[tuple[str, Entry | None, str]]] = {}
        for hit, sec, entry in cv.skill_hits:
            self.skill_where.setdefault(hit.skill.name, []).append((sec, entry, hit.alias))
        self.cv_skills = set(self.skill_where)
        self.relevance = [self.role_relevance(e) for e in cv.jobs]  # needed by requirement_evidence
        self.evidence = [self.requirement_evidence(r) for r in job.requirements]

    # ---------------------------------------------------------- requirement
    def _soft_only(self, req: Requirement) -> bool:
        tax = taxonomy()
        kinds = {tax.skills[s].kind for s in req.skills}
        if req.skills and kinds <= {"soft"}:
            return True
        if not req.skills and phrase_set("cliches").any(req.text):
            return True
        return False

    def skill_level(self, name: str) -> tuple[int, str]:
        """Evidence level of a skill in the CV (0-3) and where it was found."""
        where = self.skill_where.get(name, [])
        if not where:
            return 0, ""
        level, how = 1, "zadeklarowana (" + where[0][0] + ")"
        for sec, entry, _alias in where:
            if sec in USED_SECTIONS or sec in ("education", "courses"):
                lvl = 2
                text = entry.text if entry is not None else ""
                if sec in USED_SECTIONS and text and (_is_result(text) or _numbers(text)):
                    lvl = 3
                if lvl > level:
                    level = lvl
                    how = f"{sec}: {short(entry.title, 60)}" if entry is not None else sec
        return level, how

    def requirement_evidence(self, req: Requirement) -> Evidence:
        """Level 0-3 for a requirement (Grok scale: none / mention / scope / result)."""
        if req.knockouts and all(lexicons()["knockouts"][k].get("document_only") for k in req.knockouts):
            return Evidence(req, None, "", "dokument do dostarczenia - nie wynika z CV")
        if req.language:
            lang, need = req.language
            have = self.cv.languages.get(lang, "missing")
            if have == "missing":
                return Evidence(req, 0, "", f"brak języka: {lang}")
            if need is None or have is None:
                return Evidence(req, 2, "", f"{lang}: poziom {have or 'nie podany'}")
            ok = LEVELS.index(have) >= LEVELS.index(need)
            return Evidence(req, 3 if ok else 1, "", f"{lang}: {have} vs wymagane {need}")
        hard = [k for k in req.knockouts if not lexicons()["knockouts"][k].get("document_only")]
        if hard:
            met = [k for k in hard if self.knockout_met(k)]
            if met:
                return Evidence(req, 3, "", "warunek spełniony: " + ", ".join(lexicons()["knockouts"][k]["label"] for k in met))
        if self._soft_only(req):
            return Evidence(req, None, "", "kompetencja miękka - do weryfikacji na rozmowie")
        rest = [s for s in req.stems if s not in ("stanowisk", "podobn", "branz", "zawodow", "pracy", "pokrewn")]
        if req.years and not req.skills and len(rest) <= 2:
            have = self.relevant_months() / 12
            lvl = 3 if have >= req.years else 1 if have >= req.years / 2 else 0
            return Evidence(req, lvl, "", f"trafny staż {have:.1f} z wymaganych {req.years} lat")
        best_level, best_line, how = 0, "", ""
        tax = taxonomy()
        hard_skills = [n for n in req.skills if tax.skills[n].kind != "soft"] or req.skills
        if hard_skills:
            levels = {name: self.skill_level(name) for name in hard_skills}
            vals = [lv for lv, _ in levels.values()]
            # all listed skills matter ("Photoshop, Illustrator, InDesign"), so blend mean and max
            best_level = int(round(0.4 * max(vals) + 0.6 * sum(vals) / len(vals)))
            top = max(levels, key=lambda n: levels[n][0])
            missing = [n for n, (lv, _) in levels.items() if lv == 0]
            how = f"{top}: {levels[top][1]}" + (f"; brak: {', '.join(missing)}" if missing else "")
        hits = self.index.best(req.text, top=3)
        for line, _norm, overlap in hits:
            sec, entry = self.line_meta.get(line, ("", None))
            # overlap is IDF-weighted: the distinctive words of the requirement must be there
            if overlap >= 0.65:
                lvl = 2 if sec in USED_SECTIONS or sec in ("education", "courses") else 1
                if lvl == 2 and sec in USED_SECTIONS and (_is_result(line) or _numbers(line)
                                                          or (entry is not None and _is_result(entry.text))):
                    lvl = 3
            elif overlap >= 0.4:
                lvl = 1
            else:
                lvl = 0
            if lvl > best_level:
                best_level, best_line, how = lvl, line, f"tekst ({pct(overlap)} wspólnych słów, {sec})"
        if best_level < 2:
            emb = similarity.embedding_best_match(req.text, [ln for ln, s, _ in self.cv_lines if len(ln.split()) >= 3])
            if emb is not None and emb[1] >= 0.62:
                sec = self.line_meta.get(emb[0], ("", None))[0]
                lvl = 2 if sec in USED_SECTIONS else 1
                if lvl > best_level:
                    best_level, best_line, how = lvl, emb[0], f"podobieństwo znaczeniowe {emb[1]:.2f}"
        if req.years:
            have = self.relevant_months() / 12
            if have < req.years:
                best_level = min(best_level, 1)
                how += f"; staż {have:.1f} z wymaganych {req.years} lat"
        return Evidence(req, best_level, best_line, how)

    # --------------------------------------------------------------- roles
    def _jd_core_skills(self) -> set[str]:
        return self.job.skills(("must", "duty", "nice")) - {
            s for s in self.job.skills() if taxonomy().skills[s].kind == "soft"}

    def _jd_stems(self) -> set[str]:
        st = []
        for r in self.job.requirements:
            if r.kind in ("must", "duty"):
                st.extend(r.stems)
        return {s for s in st if len(s) >= 4}

    def role_relevance(self, entry: Entry) -> dict:
        """Grok-style relevance of one role: problem, context, scale, ownership (0-3 each)."""
        text = entry.text
        tax = taxonomy()
        entry_skills = {h.skill.name for h in tax.find(text)}
        core = self._jd_core_skills()
        skill_share = len(core & entry_skills) / len(core) if core else 0.0
        jd_st = self._jd_stems()
        stem_share = len(jd_st & set(textnorm.content_stems(text))) / len(jd_st) if jd_st else 0.0
        share = max(skill_share, stem_share * 1.4)
        problem = 3 if share >= 0.35 else 2 if share >= 0.2 else 1 if share >= 0.08 else 0
        cos = _cosine(tax.domain_profile(text), self.job.domains)
        context = 3 if cos >= 0.8 else 2 if cos >= 0.5 else 1 if cos >= 0.2 else 0
        scope = len(_SCOPE.findall(_DATEISH.sub(" ", text)))
        scale = 3 if scope >= 2 else 2 if scope == 1 else 1 if _numbers(text) else 0
        strong = phrase_set("ownership_strong").count(text)
        weak = phrase_set("ownership_weak").count(text)
        own = 3 if strong and not weak else 2 if strong else 0.5 if weak else 1
        raw = (problem + context + scale + own) / 4
        if problem == 0:
            raw = min(raw, 0.75)
        since = entry.months_since_end(self.cv.today)
        fresh = 1.0 if since <= 24 else self.fresh_mid if since <= 48 else self.fresh_old
        length = 0.6 if entry.months < 6 and not entry.ongoing else 1.0
        return {"entry": entry, "problem": problem, "context": context, "scale": scale, "ownership": own,
                "raw": raw, "fresh": fresh, "length": length, "score": raw * fresh * length,
                "share": share, "cos": cos}

    def relevant_months(self) -> int:
        """Months of relevant experience (roles with problem >= 1 or context >= 2), overlaps once."""
        rel = [r["entry"] for r in self.relevance if r["problem"] >= 1 or r["context"] >= 2]
        return merged_months(rel)


    def knockout_met(self, key: str) -> bool:
        """True if the CV shows the knock-out condition (see lexicons.yaml ``knockouts``)."""
        spec = lexicons()["knockouts"][key]
        cv = self.cv
        if key == "portfolio_required" and (cv.contact.portfolio or cv.contact.other_links):
            return True
        if key == "higher_education":
            edu = "\n".join(s.text for s in cv.sections if s.key == "education")
            return bool(regex_any(spec["cv"], edu or cv.doc.text))
        return bool(regex_any(spec["cv"], cv.doc.text))


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


# ------------------------------------------------------------------- metrics


def knockouts(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Hard requirements; a missing one is flagged as critical."""
    job, cv = ctx.job, ctx.cv
    if not job.knockouts:
        return make("knockouts", None, value="brak warunków koniecznych w ofercie")
    spec = lexicons()["knockouts"]
    vals, ev, critical = [], [], False
    found_by_key: dict[str, list[str]] = {}
    for key, info in job.knockouts.items():
        found = regex_any(spec[key]["cv"], cv.doc.text)
        if key == "portfolio_required" and (cv.contact.portfolio or cv.contact.other_links):
            found = found or ["link"]
        if key == "higher_education":
            edu = "\n".join(s.text for s in cv.sections if s.key == "education")
            found = regex_any(spec[key]["cv"], edu) if edu else found
        found_by_key[key] = found
    # "wykształcenie wyższe kierunkowe LUB kwalifikacje terapeuty" - alternatives on one line:
    # meeting any of them meets all of them
    for key, info in job.knockouts.items():
        line = textnorm.fold(info["line"])
        if not found_by_key[key] and line and re.search(r"\b(lub|albo|or|badz)\b", line):
            alt = [k for k, i in job.knockouts.items() if k != key and i["line"] == info["line"] and found_by_key[k]]
            if alt:
                found_by_key[key] = [f"alternatywa: {job.knockouts[alt[0]]['label']}"]
    for key, info in job.knockouts.items():
        if info["document_only"]:
            ev.append(f"DO DOSTARCZENIA: {info['label']} (zaświadczenie, nie ocenia się z CV)")
            continue
        found = found_by_key[key]
        if found:
            vals.append(1.0)
            ev.append(f"SPEŁNIONE: {info['label']} ({found[0]})")
        elif info["soft"]:
            # a "nice to have" condition is not a knock-out: reported, scored in nice-to-have coverage
            ev.append(f"NIE WIADOMO: {info['label']} - mile widziane / warto dopisać, jeśli spełniasz")
        else:
            vals.append(0.0)
            critical = True
            ev.append(f"BRAK: {info['label']} - wymaganie: {short(info['line'] or '(w treści ogłoszenia)', 80)}")
    if not vals:
        return make("knockouts", None, value="brak twardych warunków do sprawdzenia w CV", evidence=ev)
    score = 100 * sum(vals) / len(vals)
    if critical:
        score = min(score, 30)
    return make("knockouts", score, value=f"{sum(1 for v in vals if v == 1)}/{len(vals)}", evidence=ev, critical=critical,
                suggestion="Jeśli spełniasz warunek, napisz to wprost w CV (np. 'Prawo jazdy kat. B', 'Przygotowanie "
                           "pedagogiczne - studia podyplomowe ..., 2019')." if score < 100 else "")


def _coverage(ctx: FitContext, kind: str, metric: str) -> MetricResult:
    evs = [e for e in ctx.evidence if e.req.kind == kind]
    if not evs:
        return make(metric, None, value="brak takich pozycji w ofercie")
    known = [e for e in evs if e.level is not None]
    if not known:
        return make(metric, None, value="tylko kompetencje miękkie/dokumenty",
                    evidence=[f"? {short(e.req.text, 90)} - {e.how}" for e in evs])
    avg = sum(e.level for e in known) / len(known)
    zeros = [e for e in known if e.level == 0]
    ev = [f"[{e.level if e.level is not None else '?'}] {short(e.req.text, 80)} - "
          f"{LEVEL_NAMES.get(e.level, e.how) if e.level is not None else e.how}"
          + (f" | {e.how}" if e.level else "") for e in evs]
    score = 100 * avg / 3
    if kind == "must" and zeros:
        score = min(score, 100 - 18 * len(zeros))
    sugg = ""
    if zeros:
        sugg = "Brak dowodów w CV dla: " + "; ".join(short(e.req.text, 60) for e in zeros[:5]) + \
               ". Jeśli masz to doświadczenie - opisz je przy konkretnym stanowisku lub projekcie."
    return make(metric, score, value=f"{avg:.2f}/3, braków: {len(zeros)}", evidence=ev, suggestion=sugg)


def must_have_coverage(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Average evidence level of must-have requirements."""
    return _coverage(ctx, "must", "must_have_coverage")


def nice_to_have_coverage(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Average evidence level of nice-to-have requirements."""
    return _coverage(ctx, "nice", "nice_to_have_coverage")


def duty_coverage(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Average evidence level of the job's responsibilities."""
    return _coverage(ctx, "duty", "duty_coverage")


def experience_relevance(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Best relevant role (not the sum of years), adjusted for recency and duration."""
    rel = sorted(ctx.relevance, key=lambda r: -r["score"])
    if not rel:
        return make("experience_relevance", None, value="brak datowanych stanowisk",
                    suggestion="Dodaj daty (miesiąc/rok - miesiąc/rok) przy stanowiskach i projektach.")
    best = rel[0]["score"]
    bonus = 0.3 if len(rel) > 1 and rel[1]["score"] >= 2 else 0.0
    total = min(3.0, best + bonus)
    ev = []
    for r in rel[:5]:
        e = r["entry"]
        ev.append(f"{short(e.title or '(bez tytułu)', 55)} [{e.start:%m.%Y}-{'obecnie' if e.ongoing else format(e.end, '%m.%Y')}]: "
                  f"problem {r['problem']}, kontekst {r['context']}, skala {r['scale']}, samodzielność {r['ownership']:g}"
                  f" -> {r['raw']:.2f} x świeżość {r['fresh']} x długość {r['length']} = {r['score']:.2f}")
    verdict = "trafne" if total >= 2.4 else "częściowo trafne" if total >= 1.6 else "sąsiednie/historyczne"
    return make("experience_relevance", 100 * total / 3, value=f"{total:.2f}/3 ({verdict})", evidence=ev,
                suggestion="Najbardziej pasującą rolę opisz konkretnie: jakie zadania z ogłoszenia wykonywałaś, "
                           "dla kogo, w jakiej skali i z jakim efektem." if total < 2.4 else "")


def _kind_match(ctx: FitContext, metric: str, kinds: set[str], domains: set[str] | None = None) -> MetricResult:
    tax = taxonomy()
    wanted = {s for s in ctx.job.skills() if tax.skills[s].kind in kinds and (domains is None or tax.skills[s].domain in domains)}
    if not wanted:
        return make(metric, None, value="oferta nie wymienia")
    levels = {s: ctx.skill_level(s)[0] for s in wanted}
    must = ctx.job.skills(("must",))
    weights = {s: (2.0 if s in must else 1.0) for s in wanted}
    score = 100 * sum(weights[s] * min(levels[s], 2) / 2 for s in wanted) / sum(weights.values())
    have = sorted(s for s in wanted if levels[s] > 0)
    miss = sorted(s for s in wanted if levels[s] == 0)
    return make(metric, score, value=f"{len(have)}/{len(wanted)}",
                evidence=[f"Są: {', '.join(f'{s} ({levels[s]})' for s in have) or '-'}", f"Brak: {', '.join(miss) or '-'}"],
                suggestion=("W CV nie widać: " + ", ".join(miss)) if miss else "")


def skill_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """All non-soft skills of the offer."""
    return _kind_match(ctx, "skill_match", {"tool", "technique", "knowledge", "population", "setting"})


def tool_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Software and platforms."""
    return _kind_match(ctx, "tool_match", {"tool"})


def technique_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Artistic and graphic techniques."""
    return _kind_match(ctx, "technique_match", {"technique"})


def population_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Target groups: children, adults, seniors, disabilities, neurological, psychiatric."""
    return _kind_match(ctx, "population_match", {"population"})


def setting_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Institution type: SDS, WTZ, DPS, foundation..."""
    return _kind_match(ctx, "setting_match", {"setting"})


QUALIFICATION_KO = ("higher_education", "pedagogical_qualification", "special_pedagogy",
                    "occupational_therapist", "art_therapy")


def qualification_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Formal qualifications required by the offer, confirmed in education/courses."""
    job, cv = ctx.job, ctx.cv
    spec = lexicons()["knockouts"]
    edu = "\n".join(s.text for s in cv.sections if s.key in ("education", "courses"))
    items = []  # (label, ok, kind, requirement line)
    for key in QUALIFICATION_KO:
        if key in job.knockouts:
            items.append((spec[key]["label"], bool(regex_any(spec[key]["cv"], edu or cv.doc.text)),
                          job.knockouts[key]["kind"], job.knockouts[key]["line"]))
    tax = taxonomy()
    edu_skills = taxonomy().names(edu)
    seen = {label for label, _, _, _ in items}
    for r in job.requirements:
        if r.kind not in ("must", "nice") or not re.search(
                r"(?i)wykszta|studi|kurs|kwalifik|ukończ|dyplom|certyfik|uprawnie", r.text):
            continue
        for s in r.skills:
            if (tax.skills[s].domain in ("education", "therapy_special", "fine_arts")
                    and tax.skills[s].kind == "knowledge" and s not in seen):
                seen.add(s)
                items.append((s, s in edu_skills, r.kind, r.text))
    # alternatives in one requirement ("studia ... LUB kurs ...") - one met is enough
    for i, (label, ok, kind, line) in enumerate(items):
        if not ok and line and re.search(r"\b(lub|albo|or|bądź)\b", line, re.IGNORECASE):
            if any(o for _, o, _, ln in items if ln == line):
                items[i] = (label + " (alternatywa spełniona)", True, kind, line)
    items = [(label, ok, kind) for label, ok, kind, _ in items]
    if not items:
        return make("qualification_match", None, value="oferta nie wymaga kwalifikacji formalnych")
    w = [(2 if k == "must" else 1) for _, _, k in items]
    score = 100 * sum(wi for (_, ok, _), wi in zip(items, w) if ok) / sum(w)
    return make("qualification_match", score, value=f"{sum(1 for _, ok, _ in items if ok)}/{len(items)}",
                evidence=[("OK: " if ok else "BRAK: ") + f"{name} ({kind})" for name, ok, kind in items],
                suggestion="Wpisz wymagane kwalifikacje w sekcji Wykształcenie/Kursy z nazwą uczelni i rokiem."
                if score < 100 else "")


def _jd_terms(job: JobOffer, limit: int = 25) -> list[tuple[str, str]]:
    """Most important stems of the offer with a readable surface form."""
    counts: Counter = Counter()
    surface: dict[str, str] = {}
    for r in job.requirements:
        weight = 2 if r.kind == "must" else 1
        for tok in textnorm.tokens(r.text):
            if tok in textnorm.STOPWORDS or len(tok) < 4 or tok.isdigit():
                continue
            st = textnorm.stem(tok)
            if st in textnorm.GENERIC_REQUIREMENT_WORDS:
                continue
            counts[st] += weight
            surface.setdefault(st, tok)
    skills_stems = set()
    for s in job.skills():
        skills_stems.update(textnorm.stems(s))
    ranked = sorted(counts, key=lambda st: (-(counts[st] + (3 if st in skills_stems else 0)), st))
    return [(st, surface[st]) for st in ranked[:limit]]


def keyword_coverage(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Share of the offer's key terms present in the CV (what an ATS keyword filter sees)."""
    terms = _jd_terms(ctx.job)
    if not terms:
        return make("keyword_coverage", None, value="brak terminów")
    cv_st = set(textnorm.stems(ctx.cv.doc.text))
    have = [s for s, _ in terms if s in cv_st]
    miss = [w for s, w in terms if s not in cv_st]
    ratio = len(have) / len(terms)
    return make("keyword_coverage", ramp(ratio, 0.15, 0.75), value=pct(ratio),
                evidence=[f"Brakujące terminy: {', '.join(miss[:15]) or '-'}"],
                suggestion="Użyj w CV słownictwa z ogłoszenia tam, gdzie zgodnie z prawdą opisuje Twoje doświadczenie: "
                           + ", ".join(miss[:8]) if ratio < 0.6 else "")


def semantic_similarity(ctx: FitContext, _profile: Profile) -> MetricResult:
    """TF-IDF on stems + character n-grams (+ embeddings if available)."""
    jd = "\n".join(r.text for r in ctx.job.requirements) or ctx.job.doc.text
    cv_text = ctx.cv.doc.text
    word = similarity.tfidf_similarity(cv_text, jd)
    char = similarity.char_similarity(cv_text, jd)
    emb = similarity.embedding_similarity(cv_text, jd)
    score = 0.55 * ramp(word, 0.03, 0.35) + 0.45 * ramp(char, 0.15, 0.6)
    ev = [f"TF-IDF (rdzenie słów): {word:.3f}", f"TF-IDF (n-gramy znakowe): {char:.3f}"]
    if emb is not None:
        score = 0.6 * score + 0.4 * ramp(emb, 0.35, 0.8)
        ev.append(f"Embeddingi ({similarity.EMBEDDING_MODEL}): {emb:.3f}")
    else:
        ev.append("Embeddingi: niedostępne (opcjonalne, zob. README)")
    return make("semantic_similarity", score, value=f"{word:.2f}/{char:.2f}" + (f"/{emb:.2f}" if emb is not None else ""),
                evidence=ev)


def bm25_relevance(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Mean best-line overlap for must-haves and duties (BM25-ranked)."""
    reqs = [r for r in ctx.job.requirements if r.kind in ("must", "duty")]
    if not reqs:
        return make("bm25_relevance", None, value="brak wymagań")
    best = []
    for r in reqs:
        hits = ctx.index.best(r.text, top=1)
        best.append((r, hits[0] if hits else ("", 0.0, 0.0)))
    avg = sum(h[2] for _, h in best) / len(best)
    ev = [f"{pct(h[2])} {short(r.text, 50)} <- {short(h[0], 60)}" for r, h in sorted(best, key=lambda x: -x[1][2])[:5]]
    return make("bm25_relevance", ramp(avg, 0.1, 0.6), value=pct(avg), evidence=ev)


def years_requirement(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Years of relevant experience vs required."""
    need = ctx.job.years_required
    if not need:
        return make("years_requirement", None, value="oferta nie określa stażu")
    have = ctx.relevant_months() / 12
    total = merged_months(ctx.cv.jobs) / 12
    return make("years_requirement", 100 * min(1.0, have / need), value=f"{have:.1f} / {need} lat",
                evidence=[f"Trafny staż: {have:.1f} lat (łącznie {total:.1f} lat w datowanych rolach)"],
                suggestion="Doświadczenie z umów zleceń, wolontariatu i projektów też jest stażem - dodaj je z datami."
                if have < need else "")


_LV = {"junior": 1, "mid": 2, "senior": 3, "lead": 4}


def _cv_level(ctx: FitContext) -> tuple[int, str]:
    from .content import _entry_level  # pylint: disable=import-outside-toplevel

    years = merged_months(ctx.cv.jobs) / 12
    latest = sorted(ctx.cv.jobs, key=lambda e: e.end, reverse=True)
    title_level = next((lv for lv in (_entry_level(e.title) for e in latest[:2]) if lv), None)
    by_years = 1 if years < 2 else 2 if years < 5 else 3
    level = max(title_level or 0, by_years) if title_level != 4 else 4
    return level, f"staż {years:.1f} lat, tytuł: {latest[0].title if latest else '-'}"


def seniority_fit(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Junior/mid/senior/lead of the offer vs the CV."""
    job_level = _LV.get(ctx.job.seniority or "")
    if job_level is None and ctx.job.years_required:
        job_level = 1 if ctx.job.years_required < 2 else 2 if ctx.job.years_required < 5 else 3
    if job_level is None:
        return make("seniority_fit", None, value="oferta nie określa poziomu")
    cv_level, why = _cv_level(ctx)
    diff = cv_level - job_level
    score = {0: 100, 1: 75, -1: 60, 2: 45}.get(diff, 20)
    names = {1: "junior", 2: "mid", 3: "senior", 4: "lead"}
    return make("seniority_fit", score, value=f"CV {names[cv_level]} vs oferta {names[job_level]}", evidence=[why],
                suggestion="Poziom CV jest niższy niż oczekiwany - pokaż samodzielne projekty i odpowiedzialność."
                if diff < 0 else ("Jesteś powyżej poziomu stanowiska - w liście wyjaśnij motywację." if diff >= 2 else ""))


def domain_alignment(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Cosine of skill-domain distributions."""
    cv_dom = taxonomy().domain_profile(ctx.cv.doc.text)
    cos = _cosine(cv_dom, ctx.job.domains)
    tax = taxonomy()

    def top(dist: dict[str, float]) -> str:
        return ", ".join(f"{tax.domains[k]} {v:.0%}" for k, v in list(dist.items())[:3])

    return make("domain_alignment", ramp(cos, 0.2, 0.85), value=f"{cos:.2f}",
                evidence=[f"Oferta: {top(ctx.job.domains) or '-'}", f"CV: {top(cv_dom) or '-'}"])


def skill_recency(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Required skills used in roles active in the last 36 months."""
    wanted = ctx._jd_core_skills()  # pylint: disable=protected-access
    found = {s for s in wanted if s in ctx.skill_where}
    if not found:
        return make("skill_recency", None, value="brak wspólnych umiejętności")
    recent = set()
    for s in found:
        for sec, entry, _ in ctx.skill_where[s]:
            if entry is not None and sec in USED_SECTIONS and entry.months_since_end(ctx.cv.today) <= ctx.recent_months:
                recent.add(s)
    ratio = len(recent) / len(found)
    return make("skill_recency", ramp(ratio, 0.1, 0.8), value=f"{len(recent)}/{len(found)}",
                evidence=[f"Świeże (ostatnie {ctx.recent_months} mies.): {', '.join(sorted(recent)) or '-'}",
                          f"Tylko starsze/bez daty: {', '.join(sorted(found - recent)) or '-'}"])


def language_requirements(ctx: FitContext, _profile: Profile) -> MetricResult:
    """Required language levels vs the CV."""
    req = ctx.job.languages
    if not req:
        return make("language_requirements", None, value="oferta nie wymaga języków")
    vals, ev = [], []
    for lang, need in req.items():
        have = ctx.cv.languages.get(lang, "missing")
        if have == "missing":
            vals.append(0.0)
            ev.append(f"BRAK: {lang} (wymagany {need or 'bez poziomu'})")
        elif need is None or have is None:
            vals.append(0.7 if have is None else 1.0)
            ev.append(f"{lang}: CV {have or 'bez poziomu'}, oferta {need or 'bez poziomu'}")
        else:
            ok = LEVELS.index(have) >= LEVELS.index(need)
            vals.append(1.0 if ok else 0.4)
            ev.append(("OK" if ok else "ZA NISKI") + f": {lang} {have} vs {need}")
    return make("language_requirements", 100 * sum(vals) / len(vals), value=f"{sum(1 for v in vals if v == 1)}/{len(vals)}",
                evidence=ev, suggestion="Podaj poziom języka w skali CEFR (A1-C2)." if min(vals) < 1 else "")


def tailoring(ctx: FitContext, _profile: Profile) -> MetricResult:
    """CV visibly aimed at this role, without keyword stuffing."""
    cv = ctx.cv
    head = (cv.headline() + "\n" + "\n".join(s.text for s in cv.sections if s.key in ("header", "summary")))[:1500]
    skills = "\n".join(s.text for s in cv.sections if s.key == "skills")
    title_st = [s for s in textnorm.content_stems(ctx.job.title, textnorm.GENERIC_REQUIREMENT_WORDS) if len(s) >= 4]
    head_st = set(textnorm.stems(head))
    title_hit = bool(title_st) and len(set(title_st) & head_st) / len(set(title_st)) >= 0.5
    terms = _jd_terms(ctx.job, 15)
    top_share = sum(1 for s, _ in terms if s in set(textnorm.stems(head + "\n" + skills))) / max(1, len(terms))
    counts = Counter(textnorm.content_stems(cv.doc.text))
    stuffed = [w for w, c in counts.items() if c >= 12 and len(w) >= 5]
    n_skills = len(taxonomy().names(skills))
    score = (40 if title_hit else 10) + 50 * min(1.0, top_share / 0.5) - (15 if stuffed else 0) - (10 if n_skills > 40 else 0) + 10
    ev = [f"Nazwa stanowiska w nagłówku/profilu: {'tak' if title_hit else 'nie'} ('{short(ctx.job.title, 60)}')",
          f"Kluczowe terminy oferty w nagłówku/profilu/umiejętnościach: {pct(top_share)}"]
    if stuffed:
        ev.append("Możliwe upychanie słów: " + ", ".join(stuffed[:5]))
    return make("tailoring", score, value=pct(top_share), evidence=ev,
                suggestion="Dopasuj nagłówek i profil do tej oferty (nazwa roli, 2-3 kluczowe kompetencje z ogłoszenia)."
                if score < 70 else "")


def location_match(ctx: FitContext, _profile: Profile) -> MetricResult:
    """City of the offer vs city in the CV (remote work makes it irrelevant)."""
    jd = ctx.job.doc.text
    if re.search(r"(?i)zdaln|remote|home office|100% online", jd):
        return make("location_match", 100, value="praca zdalna/hybrydowa", evidence=["Oferta dopuszcza pracę zdalną"])
    city = next((c for c in KNOWN_CITIES if re.search(rf"\b{c}\b", jd)), None)
    if not city or not ctx.cv.contact.city:
        return make("location_match", None, value=f"oferta: {city or '?'}, CV: {ctx.cv.contact.city or '?'}")
    same = textnorm.fold(city) == textnorm.fold(ctx.cv.contact.city)
    return make("location_match", 100 if same else 50, value=f"{ctx.cv.contact.city} vs {city}",
                evidence=["Ta sama miejscowość" if same else "Inna miejscowość - napisz o gotowości do dojazdu/przeprowadzki"])


ALL = [knockouts, must_have_coverage, nice_to_have_coverage, duty_coverage, experience_relevance, skill_match,
       tool_match, technique_match, population_match, setting_match, qualification_match, keyword_coverage,
       semantic_similarity, bm25_relevance, years_requirement, seniority_fit, domain_alignment, skill_recency,
       language_requirements, tailoring, location_match]
