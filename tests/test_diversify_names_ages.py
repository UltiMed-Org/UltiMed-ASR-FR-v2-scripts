"""Stdlib test for 99_hf_release/06_diversify_names_ages.py.

Checks that overused surnames and ages are rewritten after a title or a person, that
names and durations outside those contexts stay, and that a second run changes nothing.

Run: python tests/test_diversify_names_ages.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sync_pass_guard import load_sync_module  # noqa: E402

SCRIPT = "06_diversify_names_ages.py"


def test_rewrite() -> None:
    mod = load_sync_module(script=SCRIPT)
    pool = mod.load_pool()
    text = ("Monsieur Girard, 45 ans, et la petite Léa Bernard, patiente de 34 ans. Syndrome de "
            "Claude Bernard-Horner, le Moreau, fumeur depuis 45 ans.")
    out = mod.rewrite(text, "000014_0001_x.flac", pool)
    assert "Girard" not in out and "Léa Bernard" not in out and "45 ans," not in out, out
    assert "Claude Bernard-Horner, le Moreau, fumeur depuis 45 ans." in out, out
    assert mod.rewrite(out, "000014_0001_x.flac", pool) == out  # idempotent
    assert mod.rewrite(text, "000014_0001_x.flac", pool) == out  # same clip, same draw
    # a kept age stays kept on the next run, a moved one never lands on an overused age
    for i in range(300):
        age = mod.new_age(f"c{i}|45", 45)
        assert age == 45 or (age not in mod.OVERUSED_AGES and abs(age - 45) <= mod.AGE_SPREAD), age
    assert not set(mod.REPLACED) & set(mod.load_pool()[0])
    assert "Lefebvre" not in mod.rewrite("Madame Lefebvre.", "a.flac", pool)
    # first names: same sex, a surname after a title stays, a second run changes nothing
    firsts = mod.load_first_pools()
    text = "La petite Léa Bernard, 3 ans, et son frère Lucas. Monsieur Lucas, pomme d'Adam."
    out = mod.rewrite(text, "000014_0002_x.flac", pool, firsts)
    assert "Léa" not in out and "frère Lucas" not in out and "Monsieur Lucas" in out, out
    girl = out.split("petite ")[1].split(" ")[0]
    assert girl in firsts["2"][0], girl
    assert mod.rewrite(out, "000014_0002_x.flac", pool, firsts) == out
    print("test_rewrite: OK")


if __name__ == "__main__":
    test_rewrite()
