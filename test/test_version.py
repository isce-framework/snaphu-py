import re

import snaphu


def test_version_pep440():
    # Check that the version string is PEP440-compliant.
    # https://peps.python.org/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
    regex = re.compile(
        r"""
        v?
        (?:
            (?:(?P<epoch>[0-9]+)!)?                           # epoch
            (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
            (?P<pre>                                          # pre-release
                [-_\.]?
                (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
                [-_\.]?
                (?P<pre_n>[0-9]+)?
            )?
            (?P<post>                                         # post release
                (?:-(?P<post_n1>[0-9]+))
                |
                (?:
                    [-_\.]?
                    (?P<post_l>post|rev|r)
                    [-_\.]?
                    (?P<post_n2>[0-9]+)?
                )
            )?
            (?P<dev>                                          # dev release
                [-_\.]?
                (?P<dev_l>dev)
                [-_\.]?
                (?P<dev_n>[0-9]+)?
            )?
        )
        (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    assert regex.fullmatch(snaphu.__version__)


def test_version_tuple():
    # `__version_tuple__` should contain exactly 3 nonnegative integer release numbers,
    # plus optional string(s) representing pre/post/dev release segments and/or local
    # version label.
    major, minor, patch, *extra = snaphu.__version_tuple__
    assert all(isinstance(n, int) for n in [major, minor, patch])
    assert (major, minor, patch) >= (0, 0, 0)
    assert all(isinstance(s, str) for s in extra)
