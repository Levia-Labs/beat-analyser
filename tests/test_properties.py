"""Properties of the pure functions at the trust boundary and in the CSV writer.

Example tests pin known cases; these search for the inputs nobody thought of. Profiles are
in conftest.py: `task test:run` runs 50 examples, `task props:deep` 1000.
"""

import base64
import math
import ntpath
import posixpath
import string
from urllib.parse import urlsplit

import pytest
import webauth
from hypothesis import example, given
from hypothesis import strategies as st

import analysis
import catalog
import mcp_tools
import notes
import progress
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


@given(st.one_of(st.none(), st.text(max_size=300)), ID)
@example("../../../etc/passwd", "dQw4w9WgXcQ")
@example("-" * 5 + "x" * 200, "dQw4w9WgXcQ")
def test_title_name_is_a_valid_name(title, vid):
    assert storage.is_valid_name(youtube.project_name(title, vid))


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


EXTENSION = st.sampled_from([*sorted(storage.ALLOWED_EXTENSIONS), ".MP3", ".txt", ""])


@given(FILENAME, EXTENSION, st.binary(max_size=64))
def test_check_audio_returns_the_bytes_or_refuses(stem, extension, data):
    try:
        name, ext, decoded = mcp_tools.check_audio(
            stem + extension, base64.b64encode(data).decode(), 64
        )
    except ValueError:
        return
    assert storage.is_valid_name(name)
    assert ext in storage.ALLOWED_EXTENSIONS
    assert decoded == data


@given(st.binary(min_size=65, max_size=300), st.booleans())
def test_check_audio_refuses_more_than_the_limit(data, wrapped):
    encoded = base64.encodebytes(data) if wrapped else base64.b64encode(data)
    with pytest.raises(ValueError):
        mcp_tools.check_audio("a.wav", encoded.decode(), 64)


@given(st.binary(min_size=1, max_size=300))
def test_check_audio_accepts_wrapped_base64(data):
    assert mcp_tools.check_audio("a.wav", base64.encodebytes(data).decode(), 300)[2] == data


@given(st.text(max_size=40))
@example("0123456789abcdef0123456789abcdef")
def test_progress_ids_are_32_lowercase_hex(value):
    accepted = progress.is_valid_id(value)
    assert accepted == (len(value) == 32 and all(c in "0123456789abcdef" for c in value))


QUERY_ARGS = st.dictionaries(
    st.sampled_from(
        [
            "q",
            "key",
            "status",
            "bpm_min",
            "bpm_max",
            "dur_min",
            "dur_max",
            "sort",
            "order",
            "offset",
            "limit",
            "x",
        ]
    ),
    st.text(max_size=12),
)


@given(QUERY_ARGS)
def test_catalog_parse_returns_a_bounded_query_or_refuses(args):
    try:
        query = catalog.parse(args)
    except catalog.QueryError:
        return
    assert query.sort in catalog.SORTS
    assert query.order in catalog.ORDERS
    assert query.offset >= 0
    assert 1 <= query.limit <= catalog.MAX_LIMIT
    assert query.status in ("", "built", "pending")
    assert all(key in catalog.KEYS or key in catalog.MODES for key in query.keys)
    assert len(set(query.keys)) == len(query.keys)
    for bound in (query.bpm_min, query.bpm_max, query.dur_min, query.dur_max):
        assert bound is None or math.isfinite(bound)


@given(st.text(), st.text())
def test_token_comparison_is_equality_and_never_raises(a, b):
    assert webauth.same(a, b) == (a == b)


@given(st.text(min_size=1), st.text(min_size=1))
def test_session_value_is_64_hex_and_one_per_token(a, b):
    value = webauth.session_value(a)
    assert len(value) == 64
    assert set(value) <= set(string.hexdigits.lower())
    assert (value == webauth.session_value(b)) == (a == b)
