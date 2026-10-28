"""Professional profiles (profiles.yaml): weights, expected sections and targets,
selected automatically from the job offer or given explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import load_yaml


@dataclass
class Profile:
    """Effective (possibly blended) profile."""

    name: str
    label: str
    metric_weights: dict[str, float] = field(default_factory=dict)
    group_weights: dict[str, float] = field(default_factory=dict)
    expected_sections: list[str] = field(default_factory=list)
    targets: dict[str, float] = field(default_factory=dict)
    mix: dict[str, float] = field(default_factory=dict)

    def weight(self, metric_id: str) -> float:
        """Multiplier for a metric (1.0 when not overridden)."""
        return float(self.metric_weights.get(metric_id, 1.0))

    def target(self, key: str, default: float | None = None) -> float | None:
        """Target value."""
        return self.targets.get(key, default)


def _raw() -> dict:
    return load_yaml("profiles")


def profile_names() -> list[str]:
    """Names of the defined profiles."""
    return list(_raw()["profiles"])


def get_profile(name: str) -> Profile:
    """A single profile merged over the defaults."""
    data = _raw()
    defaults = data["defaults"]
    body = data["profiles"][name]
    targets = dict(defaults["targets"])
    targets.update(body.get("targets", {}))
    return Profile(
        name=name,
        label=body.get("label", name),
        metric_weights=dict(body.get("metric_weights", {})),
        group_weights=dict(body.get("group_weights", defaults["group_weights"])),
        expected_sections=list(body.get("expected_sections", defaults["expected_sections"])),
        targets=targets,
        mix={name: 1.0},
    )


def domain_to_profile_scores(domains: dict[str, float]) -> dict[str, float]:
    """Aggregate skill-domain shares into profile shares."""
    scores: dict[str, float] = {}
    for name, body in _raw()["profiles"].items():
        share = sum(domains.get(d, 0.0) for d in body.get("domains", []))
        if share > 0:
            scores[name] = share
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def blend(mix: dict[str, float]) -> Profile:
    """Blend profiles by weight; targets and multipliers are weighted means,
    expected sections the union of profiles with weight >= 0.3."""
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items() if v > 0}
    parts = {k: get_profile(k) for k in mix}
    keys_w = {k for p in parts.values() for k in p.metric_weights}
    keys_t = {k for p in parts.values() for k in p.targets}
    keys_g = {k for p in parts.values() for k in p.group_weights}
    sections: list[str] = []
    for name, prof in parts.items():
        if mix[name] >= 0.3 or len(parts) == 1:
            sections += [s for s in prof.expected_sections if s not in sections]
    return Profile(
        name="+".join(parts),
        label=" + ".join(f"{p.label} ({mix[k]:.0%})" for k, p in parts.items()) if len(parts) > 1
        else next(iter(parts.values())).label,
        metric_weights={k: sum(mix[n] * p.weight(k) for n, p in parts.items()) for k in keys_w},
        group_weights={k: sum(mix[n] * p.group_weights.get(k, 0.0) for n, p in parts.items()) for k in keys_g},
        expected_sections=sections,
        targets={k: sum(mix[n] * p.targets.get(k, 0.0) for n, p in parts.items()) for k in keys_t},
        mix=mix,
    )


def detect(domains: dict[str, float], max_profiles: int = 2, min_share: float = 0.2) -> Profile:
    """Pick the profile for a job offer (or a CV, when there is no offer).

    Profiles whose domains hold at least ``min_share`` of the skill mentions are
    blended (at most ``max_profiles``); with no signal the ``general`` profile is used.
    """
    scores = domain_to_profile_scores(domains)
    if len(scores) > 1:
        scores.pop("general", None)  # "general" only when nothing more specific is found
    chosen = {k: v for k, v in list(scores.items())[:max_profiles] if v >= min_share}
    if not chosen:
        chosen = dict(list(scores.items())[:1]) or {"general": 1.0}
    return blend(chosen)


def resolve(name: str | None, domains: dict[str, float]) -> Profile:
    """Explicit name (``tech``, ``creative_arts+therapy``) or ``auto``/``None``."""
    if not name or name == "auto":
        return detect(domains)
    names = [n.strip() for n in name.split("+") if n.strip()]
    unknown = [n for n in names if n not in profile_names()]
    if unknown:
        raise ValueError(f"Nieznany profil: {', '.join(unknown)}. Dostępne: {', '.join(profile_names())}")
    return blend({n: 1.0 for n in names})
