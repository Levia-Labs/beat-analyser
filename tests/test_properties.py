"""Properties of the pure functions at the trust boundary and in the CSV writer.

Example tests pin known cases; these search for the inputs nobody thought of. Profiles are
in conftest.py: `task test:run` runs 50 examples, `task props:deep` 1000.
"""

import ntpath
import posixpath
import string
from urllib.parse import urlsplit

from hypothesis import example, given
from hypothesis import strategies as st

import analysis
import notes
import storage
import youtube

ID_CHARS = string.ascii_letters + string.digits + "_-"
ID = st.text(alphabet=ID_CHARS, min_size=11, max_size=11)


# Random unicode is almost always refused outright; printable ASCII reaches the sanitiser's
# interesting cases (dots, separators, spaces, `..`).
FILENAME = st.text() | st.text(alphabet=string.printable, max_size=40)
# Names shaped like valid ones, plus noise, so the traversal property is not vacuous.
NAME = st.from_regex(r"[A-Za-z0-9.][A-Za-z0-9._/\\-]{0,20}", fullmatch=True) | st.text()


@given(FILENAME)
def test_project_name_is_stable(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    assert storage.project_name(f"{name}.wav") == name


@given(FILENAME)
@example("../../etc/passwd")
@example("..\\..\\boot.ini")
def test_project_name_stays_inside_the_root(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    # Judged by the path libraries of both platforms, not by re-reading storage's regex.
    for flavour in (posixpath, ntpath):
        assert flavour.dirname(flavour.normpath(flavour.join("uploads", name))) == "uploads"


@given(NAME)
@example("..")
@example("a/../..")
def test_valid_name_stays_inside_the_root(name):
    if not storage.is_valid_name(name):
        return
    for flavour in (posixpath, ntpath):
        assert flavour.dirname(flavour.normpath(flavour.join("uploads", name))) == "uploads"


ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
# Plain st.text() almost never spells a URL, so the property would only ever see ValueError.
URL = st.builds(
    "{}://{}{}".format,
    st.sampled_from(["http", "https", "ftp", "javascript", "HTTPS"]),
    st.sampled_from(
        [
            *sorted(ALLOWED_HOSTS),
            "YOUTUBE.com",
            "youtube.com.evil.test",
            "evil.test",
            "youtube.com@evil.test",  # scrub:allow - URL userinfo attack input, not an email
            "evil.test@youtube.com",  # scrub:allow - URL userinfo attack input, not an email
            "evil.test#youtube.com",
            "youtube.com:443",
        ]
    ),
    st.builds(
        str.format,
        st.sampled_from(["/watch?v={}", "/shorts/{}", "/{}", "/watch?x=1&v={}", "/embed/{}"]),
        ID | st.text(max_size=13),
    ),
)


@given(URL | st.text())
def test_video_id_is_an_id_from_an_allowed_host_or_refused(url):
    try:
        vid = youtube.video_id(url)
    except ValueError:
        return
    parts = urlsplit(url.strip())
    assert parts.scheme in ("http", "https")
    assert (parts.hostname or "").lower() in ALLOWED_HOSTS
    assert len(vid) == 11
    assert set(vid) <= set(ID_CHARS)


@given(ID)
def test_video_id_round_trips(vid):
    assert youtube.video_id(youtube.canonical_url(vid)) == vid
    assert youtube.video_id(f"https://youtu.be/{vid}") == vid


@given(st.floats(0, 1e5), st.floats(0, 1e5))
@example(0.03125, 0.0)  # an exact tie: "0.0312" parses to just under 0.0312
def test_format_time_is_monotonic_and_close(a, b):
    low, high = sorted((a, b))
    assert float(analysis.format_time(low)) <= float(analysis.format_time(high))
    # Half a unit in the fourth decimal, plus float noise from parsing the string back.
    assert abs(float(analysis.format_time(a)) - a) <= 5e-5 + 1e-9


# Three pitches over two seconds, short notes: dense enough that same-pitch gaps fall on
# both sides of MERGE_GAP_S, which is where merging happens.
NOTE = st.tuples(st.floats(0, 2), st.floats(0, 0.3), st.integers(60, 62), st.floats(0, 1)).map(
    lambda t: (t[0], t[0] + t[1], t[2], t[3])
)


@given(st.lists(NOTE, max_size=30))
def test_merge_repeats_separates_same_pitch_notes(found):
    last_end: dict[int, float] = {}
    for start, end, pitch, _ in notes.merge_repeats(found):
        if pitch in last_end:
            assert start - last_end[pitch] >= notes.MERGE_GAP_S
        last_end[pitch] = max(end, last_end.get(pitch, end))


@given(st.lists(NOTE, max_size=30))
def test_merge_repeats_loses_no_sounding_time(found):
    merged = notes.merge_repeats(found)
    for start, end, pitch, _ in found:
        assert any(p == pitch and s <= start and end <= e for s, e, p, _ in merged)


@given(st.lists(NOTE, max_size=30))
def test_merge_repeats_keeps_amplitude_in_range(found):
    amplitudes = [amp for *_, amp in found]
    for *_, amp in notes.merge_repeats(found):
        assert min(amplitudes) - 1e-9 <= amp <= max(amplitudes) + 1e-9
