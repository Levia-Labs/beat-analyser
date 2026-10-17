"""Properties of the pure functions at the trust boundary and in the CSV writer.

Example tests pin known cases; these search for the inputs nobody thought of. Profiles are
in conftest.py: `task test` runs 50 examples, `task test:deep` 1000.
"""

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


@given(st.text())
def test_project_name_is_valid_or_refused(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    assert storage.is_valid_name(name)
    assert "/" not in name and "\\" not in name


@given(st.text())
def test_project_name_is_stable(filename):
    try:
        name = storage.project_name(filename)
    except ValueError:
        return
    assert storage.project_name(f"{name}.wav") == name


@given(st.text())
def test_valid_name_cannot_traverse(name):
    if storage.is_valid_name(name):
        assert not name.startswith(".")
        assert "/" not in name and "\\" not in name
        assert len(name) <= 100


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
            "youtube.com@evil.test",
            "evil.test@youtube.com",
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


NOTE = st.tuples(st.floats(0, 100), st.floats(0, 5), st.integers(21, 108), st.floats(0, 1)).map(
    lambda t: (t[0], t[0] + t[1], t[2], t[3])
)


@given(st.lists(NOTE, max_size=40))
def test_merge_repeats_shortens_sorts_and_separates(found):
    merged = notes.merge_repeats(found)
    assert len(merged) <= len(found)
    assert merged == sorted(merged)
    last_end: dict[int, float] = {}
    for start, end, pitch, _ in merged:
        if pitch in last_end:
            assert start - last_end[pitch] >= notes.MERGE_GAP_S
        last_end[pitch] = max(end, last_end.get(pitch, end))
