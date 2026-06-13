"""Тесты определения сжатия, кодировки и разделителя в loader."""
import pytest

from preprocessing.loader import (
    _detect_compression,
    _detect_encoding,
    _detect_separator,
)


class TestDetectCompression:
    def test_gz(self):
        assert _detect_compression("data.csv.gz") == "gzip"

    def test_zip_uppercase(self):
        assert _detect_compression("DATA.ZIP") == "zip"

    def test_bz2(self):
        assert _detect_compression("x.bz2") == "bz2"

    def test_xz(self):
        assert _detect_compression("x.xz") == "xz"

    def test_plain_csv(self):
        assert _detect_compression("data.csv") is None

    def test_no_extension(self):
        assert _detect_compression("data") is None


class TestDetectEncoding:
    def test_ascii_is_utf8(self):
        enc, bom = _detect_encoding(b"col1,col2\n1,2\n")
        assert enc == "utf-8"
        assert bom is False

    def test_utf8_bom(self):
        enc, bom = _detect_encoding(b"\xef\xbb\xbfcol1,col2\n")
        assert enc == "utf-8-sig"
        assert bom is True

    def test_utf16_le_bom(self):
        enc, bom = _detect_encoding(b"\xff\xfec\x00o\x00l\x00")
        assert enc == "utf-16"
        assert bom is True

    def test_utf32_le_bom(self):
        enc, bom = _detect_encoding(b"\xff\xfe\x00\x00")
        assert enc == "utf-32"
        assert bom is True

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _detect_encoding(b"")


class TestDetectSeparator:
    def test_comma(self):
        assert _detect_separator("a,b,c\n1,2,3\n") == ","

    def test_semicolon(self):
        assert _detect_separator("a;b;c\n1;2;3\n") == ";"

    def test_tab(self):
        assert _detect_separator("a\tb\tc\n1\t2\t3\n") == "\t"

    def test_pipe(self):
        assert _detect_separator("a|b|c\n1|2|3\n") == "|"

    def test_single_column_raises(self):
        with pytest.raises(ValueError):
            _detect_separator("justonecolumn\nvalue1\nvalue2\n")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _detect_separator("   \n  \n")
