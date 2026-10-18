from dastavez.routing import arbitrate, decide, quality_score

ENGLISH = (
    "The Pradhan Mantri Awas Yojana Urban 2.0 will be implemented for five years. The scheme supports "
    "construction of houses with a minimum carpet area of thirty square metres for eligible families in urban areas."
)
HINDI = (
    "प्रधानमंत्री आवास योजना शहरी 2.0 पांच वर्षों के लिए लागू की जाएगी। योजना के अंतर्गत पात्र परिवारों को "
    "न्यूनतम तीस वर्ग मीटर कारपेट क्षेत्र वाले आवास के निर्माण के लिए सहायता दी जाएगी।"
)
BROKEN_HINDI = (
    "धानमंी आवास योजना शहरी 2.0 आवासन और शहरी काय मंालय भारत सरकार योजना ि◌दशा ि◌नदश ि◌सत ि◌र सबके ि◌लए"
)
KRUTI_DEV = "vkoklu vkSj 'kgjh dk;Z ea=ky; çèkkuea=h vkokl ;kstuk & 'kgjh 2-0 lcds fy, vkokl fe'ku ;kstuk fn'kkfunsZ'k"
GARBLED_SCAN_LAYER = "REGD. NG:D. L.-33004/99 '~j. - ...~*.. ~V1lR ~C:: .: Q:he of ~ndio/ (GoLette .'~ EXTRAORD~~~Y 'q'flT n-~ 3-'-'3'Q'-~ (ii)_ ~V1lR ~C:: xzQ ~~kk"


def test_good_english_text_layer_is_kept():
    d = decide(ENGLISH, "en")
    assert not d.needs_ocr and d.reason == "text_layer_ok"


def test_good_hindi_text_layer_is_kept():
    d = decide(HINDI, "hi")
    assert not d.needs_ocr


def test_empty_page_goes_to_ocr_with_language():
    d = decide("   12 ", "hi")
    assert d.needs_ocr and d.reason == "empty_text_layer" and d.ocr_lang == "hin+eng"
    assert decide("", "en").ocr_lang == "eng"


def test_broken_devanagari_goes_to_ocr():
    d = decide(BROKEN_HINDI, "hi")
    assert d.needs_ocr and d.reason == "broken_devanagari" and d.ocr_lang == "hin+eng"


def test_legacy_font_in_hindi_doc_goes_to_ocr():
    d = decide(KRUTI_DEV, "hi")
    assert d.needs_ocr and d.reason == "script_mismatch_legacy_font"


def test_garbled_latin_goes_to_ocr():
    d = decide(GARBLED_SCAN_LAYER, "en")
    assert d.needs_ocr and d.reason == "garbled_text"


def test_undecodable_glyphs_go_to_ocr():
    d = decide(ENGLISH[:60] + "��", "en")
    assert d.needs_ocr and d.reason == "undecodable_glyphs"


def test_scan_with_existing_text_layer_is_reocrd():
    d = decide(ENGLISH, "en", image_coverage=0.97)
    assert d.needs_ocr and d.reason == "scanned_with_text_layer"


def test_quality_score_orders_texts_sensibly():
    assert quality_score(HINDI, "hi") > quality_score(BROKEN_HINDI, "hi")
    assert quality_score(HINDI, "hi") > quality_score(KRUTI_DEV, "hi")
    assert quality_score(ENGLISH, "en") > quality_score(GARBLED_SCAN_LAYER, "en")
    assert quality_score("", "en") == 0.0


def test_arbitrate_keeps_text_layer_when_ocr_is_worse():
    choice, q_text, q_ocr = arbitrate(ENGLISH, GARBLED_SCAN_LAYER, "en")
    assert choice == "text_layer" and q_text > q_ocr
    choice, _, _ = arbitrate(KRUTI_DEV, HINDI, "hi")
    assert choice == "ocr"
