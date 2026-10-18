from dastavez.textutil import (
    contains_devanagari,
    invalid_devanagari_sequences,
    latin_wordlike_ratio,
    normalize_for_eval,
    script_stats,
    split_sentences,
    tokenize,
)

CLEAN_HINDI = "प्रधानमंत्री आवास योजना शहरी 2.0 के अंतर्गत पात्र परिवारों को सहायता दी जाएगी।"
# what a PDF with a broken ToUnicode map yields for the same words (from the PMAY-U 2.0 Hindi PDF)
BROKEN_HINDI = "धानमंी आवास योजना – शहरी 2.0 आवासन और शहरी काय मंालय योजना ि◌दशा"
KRUTI_DEV = "vkoklu vkSj 'kgjh dk;Z ea=ky; çèkkuea=h vkokl ;kstuk & 'kgjh 2-0 lcds fy, vkokl fe'ku"


def test_contains_devanagari():
    assert contains_devanagari(CLEAN_HINDI)
    assert not contains_devanagari("PM-KISAN operational guidelines")


def test_clean_hindi_has_no_invalid_sequences():
    assert invalid_devanagari_sequences(CLEAN_HINDI) == 0
    assert invalid_devanagari_sequences("क्षेत्र, श्री, ज्ञान, कृषि, हिंदी, ड़ ढ़ फ़") == 0


def test_broken_hindi_is_detected():
    # "मंी": vowel sign after anusvara; "ि◌" style orphan matra
    assert invalid_devanagari_sequences(BROKEN_HINDI) >= 1
    assert invalid_devanagari_sequences("ि") == 1  # matra at word start
    assert invalid_devanagari_sequences("काे") == 1  # two vowel signs in a row
    assert invalid_devanagari_sequences("अा") == 1  # vowel sign after an independent vowel


def test_script_stats_ratios():
    st = script_stats("abc कखग")
    assert st.devanagari_letters == 3 and st.latin_letters == 3
    assert st.devanagari_ratio == 0.5
    assert st.printable_ratio == 1.0
    assert script_stats("ab�").pua_ratio == 0.5
    assert script_stats("   ").n_chars == 0


def test_wordlike_ratio_separates_english_from_legacy_font():
    en, n_en = latin_wordlike_ratio("The scheme provides income support to all landholding farmer families.")
    kd, n_kd = latin_wordlike_ratio(KRUTI_DEV)
    assert n_en >= 10 and en > 0.9
    assert n_kd >= 10 and kd < 0.5


def test_tokenize_keeps_devanagari_words_whole_and_maps_digits():
    toks = tokenize("प्रधानमंत्री आवास योजना में ३० वर्ग मीटर")
    assert "प्रधानमंत्री" in toks and "आवास" in toks
    assert "में" not in toks  # stopword
    assert "30" in toks
    assert tokenize("Rs. 1,50,000 per year") == ["rs", "150000", "year"]


def test_normalize_for_eval():
    assert normalize_for_eval("  “Hello”\n\nWorld – ३ ") == '"hello" world - 3'
    assert normalize_for_eval("a​b") == "ab"


def test_split_sentences_handles_danda_and_abbreviations():
    assert split_sentences("पहला वाक्य। दूसरा वाक्य।") == ["पहला वाक्य।", "दूसरा वाक्य।"]
    assert split_sentences("Loan of Rs. 10,000 is given. Repay in 12 months.") == [
        "Loan of Rs. 10,000 is given.",
        "Repay in 12 months.",
    ]
