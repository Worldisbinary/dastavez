from dastavez.verify import Citation, parse_citations, strip_citations, support, verify_sentences

PAGE = (
    "Under the PM-KISAN scheme an income support of Rs. 6000 per year is provided to all landholding farmer "
    "families. The amount is released in three equal instalments of Rs. 2000 each."
)
PAGES = {("pmkisan", 1): PAGE, ("pmkisan", 2): "Institutional landholders are excluded."}


def test_parse_and_strip_citations():
    s = "Farmers get Rs. 6000 per year [pmkisan p.1] [pmkisan_hi p.12]."
    assert parse_citations(s) == [Citation("pmkisan", 1), Citation("pmkisan_hi", 12)]
    assert strip_citations(s) == "Farmers get Rs. 6000 per year ."


def test_verbatim_sentence_is_supported():
    ok, score, reason = support("The amount is released in three equal instalments of Rs. 2000 each.", PAGE)
    assert ok and score == 1.0 and reason == "verbatim"


def test_paraphrase_with_overlap_is_supported():
    ok, _, reason = support(
        "An income support of Rs. 6000 per year is provided to landholding farmer families", PAGE
    )
    assert ok and reason in {"overlap", "verbatim"}


def test_wrong_number_is_unsupported():
    ok, _, reason = support("Income support of Rs. 8000 per year is provided to farmer families.", PAGE)
    assert not ok and reason.startswith("number_not_on_page")


def test_fabricated_quote_is_unsupported():
    ok, _, reason = support('The guidelines say "all tenant farmers are fully covered" by the scheme.', PAGE)
    assert not ok and reason == "quote_not_on_page"


def test_unrelated_claim_is_unsupported():
    ok, _, _ = support("Street vendors receive working capital loans from banks.", PAGE)
    assert not ok


def test_verify_sentences_keeps_supported_and_drops_the_rest():
    res = verify_sentences(
        [
            "An income support of Rs. 6000 per year is provided [pmkisan p.1].",
            "It is paid in four instalments of Rs. 1500 [pmkisan p.1].",  # wrong numbers
            "Institutional landholders are excluded [pmkisan p.1].",  # true, but cited to the wrong page
            "This sentence has no citation at all.",
            "Institutional landholders are excluded [pmkisan p.2].",
            "Something on a page that does not exist [pmkisan p.99].",
        ],
        PAGES,
    )
    kept = [c.sentence for c in res.supported]
    assert kept == [
        "An income support of Rs. 6000 per year is provided .",
        "Institutional landholders are excluded .",
    ]
    reasons = [c.reason for c in res.dropped]
    assert reasons[0].startswith("number_not_on_page")
    assert "no_citation" in reasons and "page_not_found" in reasons
    assert res.support_rate == 2 / 6
    assert res.supported[1].supporting == Citation("pmkisan", 2)
