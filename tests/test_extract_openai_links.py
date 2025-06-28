import re

REGEX = r"https://learn\.microsoft\.com/[^\s\]\),]+"

def test_learn_link_regex():
    text = "See docs https://learn.microsoft.com/en-us/azure/some-path), more at https://learn.microsoft.com/en-us/other?param=1]."
    links = re.findall(REGEX, text)
    assert links == [
        "https://learn.microsoft.com/en-us/azure/some-path",
        "https://learn.microsoft.com/en-us/other?param=1"
    ]
