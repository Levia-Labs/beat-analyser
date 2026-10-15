import pytest

import youtube

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&t=42s",
        f"http://m.youtube.com/watch?list=x&v={VID}",
        f"https://music.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abc",
        f"https://www.youtube.com/shorts/{VID}",
        f"  https://www.youtube.com/watch?v={VID}  ",
    ],
)
def test_video_id_accepts(url):
    assert youtube.video_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        f"https://evil.example/?v={VID}",
        f"https://evil.example/?https://www.youtube.com/watch?v={VID}",
        f"https://www.youtube.com.evil.example/watch?v={VID}",
        f"ftp://www.youtube.com/watch?v={VID}",
        f"file:///etc/passwd?v={VID}",
        "https://www.youtube.com/watch?v=short",
        f"https://www.youtube.com/watch?v={VID}x",
        "https://www.youtube.com/watch?v=../../../etc",
    ],
)
def test_video_id_rejects(url):
    with pytest.raises(ValueError):
        youtube.video_id(url)


def test_canonical_url():
    assert youtube.canonical_url(VID) == f"https://www.youtube.com/watch?v={VID}"
