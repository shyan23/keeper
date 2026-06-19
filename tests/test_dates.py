import datetime as dt

from app.services.dates import parse_doc_date


def test_iso():
    assert parse_doc_date("2023-10-05") == dt.date(2023, 10, 5)


def test_day_first_slash():
    assert parse_doc_date("05/10/2023") == dt.date(2023, 10, 5)


def test_textual():
    assert parse_doc_date("5 Oct 2023") == dt.date(2023, 10, 5)
    assert parse_doc_date("October 5, 2023") == dt.date(2023, 10, 5)


def test_two_digit_year_day_first():
    assert parse_doc_date("05-10-23") == dt.date(2023, 10, 5)


def test_month_name_dashes():
    # DD-Mon-YY / DD-Mon-YYYY (e.g. "05-Oct-23") — common lab-report header format.
    assert parse_doc_date("05-Oct-23") == dt.date(2023, 10, 5)
    assert parse_doc_date("05-October-2023") == dt.date(2023, 10, 5)


def test_month_name_embedded_in_line():
    from app.services.dates import date_from_text
    assert date_from_text("Date: 05-Oct-23 Patient's Name : MRS. X") == dt.date(2023, 10, 5)


def test_unparseable_is_none():
    assert parse_doc_date("") is None
    assert parse_doc_date(None) is None
    assert parse_doc_date("not a date") is None
